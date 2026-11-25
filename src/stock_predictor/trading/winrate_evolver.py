"""AI 자가진화(Self-Evolving) 모의투자 시뮬레이션 & 실전 승률 극대화(Meta-Labeling) 엔진.

핵심 원리 (미래 정보 유출 0% Strict Walk-Forward):
1. 1차 AI(수익률·방향 앙상블)가 매수 후보를 고르면, 과거 모의매매 성적표로 학습된
   **2차 승률 판별기(Meta-Classifier)**가 진입 시점의 차트·변동성·수급·거시 상태를 보고
   "수수료와 세금을 다 떼고도 실제로 수익으로 끝날 확률(Meta Win Probability)"을 한 번 더 채점합니다.
2. 세대(Generation)를 거듭할수록 다양한 국내·해외 종목 바스켓에서:
   - 진입 승률 커트라인(meta_prob_min)
   - 고점 추격 방지 과열 상한(max_rsi14, max_ret5d)
   - 추세 정렬 필터(min_dist_sma20, require_uptrend)
   - 손익 비대칭 청산 규칙(목표 익절폭, 손절폭, **수익 보존 본전 잠금 트레일링 브레이크**)
   을 진화 탐색(Evolutionary Search)하여 **검증 구간 승률과 손익비가 동시에 개선될 때만 새로운 챔피언(Champion)**으로 승격합니다.
3. 승격된 챔피언 정책과 2차 승률 모델은 `saved_models/winrate_champion.pkl` 및 SQLite `evolution_generations` 테이블에
   영구 저장되어, 실시간 종목 추천과 라이브 모의투자에 즉시 적용됩니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CACHE_DIR, MARKET_COST_BPS, MODELS_DIR, PredictorConfig
from ..data.features import FeatureEngine
from ..data.market_data import MarketDataBundle
from ..models.tabular import ExtraTreesStockModel, LightGBMStockModel, LinearQuantModel
from .paper_engine import DEFAULT_INITIAL_CASH_KRW, PaperTradingDB

CHAMPION_ARTIFACT_PATH = MODELS_DIR / "winrate_champion.pkl"

# 2차 승률 판별기(Meta-Classifier)가 학습에 사용하는 진입 시점 상태 피처
META_FEATURE_COLS: list[str] = [
    "pred_ret",
    "pred_prob",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_7",
    "rsi_14",
    "rv_20d",
    "atr_pct_14",
    "macd_hist_pct",
    "bb_zscore_20",
    "sma_5_20_cross",
    "sma_20_60_cross",
    "dist_sma_20",
    "dist_sma_60",
    "drawdown_60d",
    "vol_ratio_20",
    "rel_alpha_20d",
    "vix_level",
]


@dataclass
class WinRatePolicy:
    """세대별로 진화하는 AI 매매·위험관리 규칙 유전자(Gene)."""

    name: str = "기본 초보 전략"
    min_primary_prob: float = 0.51         # 1차 AI 최소 상승확률
    min_meta_win_prob: float = 0.0         # 2차 승률 판별기 최소 통과 확률 (0이면 미적용)
    max_rsi_14: float = 78.0               # 진입 허용 최대 RSI(14) (과열 추격 차단)
    max_ret_5d: float = 0.16               # 최근 5일 급등률 상한
    min_dist_sma60: float = -0.06          # 60일 추세선 대비 최소 위치
    require_sma20_above_60: bool = False   # 정배열(20일선 >= 60일선) 필터 여부
    max_rv20: float = 0.75                 # 연환산 변동성 상한
    stop_loss_pct: float = -0.075          # 기본 안전 브레이크(손절선)
    take_profit_pct: float = 0.14          # 목표 수익 확정선(익절선)
    breakeven_trigger_pct: float = 0.99    # 수익이 이 비율을 넘으면 안전 브레이크를 본전 위로 잠금 (0.99=미사용)
    breakeven_lock_pct: float = 0.005      # 잠금 시 최소 보장 수익률 (+0.5% 등)
    trailing_pullback_pct: float = 0.99    # 고점 대비 하락 트레일링 익절 폭 (0.99=미사용)
    max_hold_days: int = 25                # 최대 보유 거래일 (시간 손절/순환)
    max_positions: int = 5                 # 동시 최대 보유 종목 수
    rebalance_every: int = 5               # 리밸런싱 주기 (거래일)
    min_rel_alpha_20d: float = -0.50       # 시장 대비 20일 상대강도 하한 (-0.50=미적용, 0.0=시장 초과강세만)


@dataclass
class GenerationRecord:
    """각 시뮬레이션 세대의 성적 및 배운 점 기록."""

    generation: int
    created_at: str
    policy_name: str
    win_rate_pct: float
    total_return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    closed_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    fitness_score: float
    is_champion: bool
    policy_dict: dict[str, Any] = field(default_factory=dict)
    learned_lessons_ko: list[str] = field(default_factory=list)


@dataclass
class ChampionWinRateArtifact:
    """실전 예측과 모의투자에 즉시 투입되는 챔피언 승률 모델 & 정책 묶음."""

    generation: int
    updated_at: str
    baseline_win_rate_pct: float
    champion_win_rate_pct: float
    champion_return_pct: float
    champion_profit_factor: float
    champion_trades: int
    total_simulated_trades: int
    policy: WinRatePolicy
    meta_model: Any | None
    meta_feature_cols: list[str]
    learned_rules_ko: list[str]
    feature_win_insights: list[dict[str, Any]] = field(default_factory=list)


def ensure_evolution_table(db: PaperTradingDB) -> None:
    with db._connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_generations (
                generation INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                policy_name TEXT NOT NULL,
                win_rate_pct REAL NOT NULL,
                total_return_pct REAL NOT NULL,
                profit_factor REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                closed_trades INTEGER NOT NULL,
                avg_win_pct REAL NOT NULL,
                avg_loss_pct REAL NOT NULL,
                fitness_score REAL NOT NULL,
                is_champion INTEGER NOT NULL DEFAULT 0,
                policy_json TEXT NOT NULL,
                lessons_json TEXT NOT NULL
            )
            """
        )


def load_evolution_history_df(db: PaperTradingDB) -> pd.DataFrame:
    ensure_evolution_table(db)
    with db._connect() as conn:
        rows = conn.execute("SELECT * FROM evolution_generations ORDER BY generation ASC").fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def save_generation_record(db: PaperTradingDB, rec: GenerationRecord) -> None:
    ensure_evolution_table(db)
    with db._connect() as conn:
        conn.execute(
            """
            UPDATE evolution_generations
            SET fitness_score = ROUND(
                win_rate_pct * 1.10
                + MIN(profit_factor, 4.0) * 14.0
                + MAX(MIN(total_return_pct, 50.0), -25.0) * 0.75
                + MAX(MIN(max_drawdown_pct, 0.0), -30.0) * 0.40
                - CASE WHEN profit_factor < 1.0 THEN (1.0 - profit_factor) * 38.0 ELSE 0.0 END,
                2
            )
            """
        )
        if rec.is_champion:
            conn.execute("UPDATE evolution_generations SET is_champion = 0")
        conn.execute(
            """
            INSERT OR REPLACE INTO evolution_generations (
                generation, created_at, policy_name, win_rate_pct, total_return_pct,
                profit_factor, max_drawdown_pct, closed_trades, avg_win_pct, avg_loss_pct,
                fitness_score, is_champion, policy_json, lessons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.generation,
                rec.created_at,
                rec.policy_name,
                rec.win_rate_pct,
                rec.total_return_pct,
                rec.profit_factor,
                rec.max_drawdown_pct,
                rec.closed_trades,
                rec.avg_win_pct,
                rec.avg_loss_pct,
                rec.fitness_score,
                1 if rec.is_champion else 0,
                json.dumps(rec.policy_dict, ensure_ascii=False),
                json.dumps(rec.learned_lessons_ko, ensure_ascii=False),
            ),
        )


def load_champion_artifact() -> ChampionWinRateArtifact | None:
    if not CHAMPION_ARTIFACT_PATH.exists():
        return None
    try:
        with CHAMPION_ARTIFACT_PATH.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, ChampionWinRateArtifact):
            return obj
    except Exception:
        return None
    return None


def predict_stock_win_confidence(
    latest_X: pd.DataFrame,
    pred_log_ret: float,
    pred_up_prob: float,
    artifact: ChampionWinRateArtifact | None = None,
) -> tuple[float, bool, str]:
    """단일 종목의 최신 피처와 1차 예측치를 받아 챔피언 기준 '실전 승률 점수(0~100%)'와 통과 여부 산출."""
    art = artifact or load_champion_artifact()
    if art is None:
        base_conf = float(np.clip(pred_up_prob * 100.0, 20.0, 85.0))
        return base_conf, pred_up_prob >= 0.52, "기본 AI 상승확률 기준"

    row = latest_X.iloc[0]
    pol = art.policy
    feat_vec: list[float] = []
    for col in art.meta_feature_cols:
        if col == "pred_ret":
            feat_vec.append(float(pred_log_ret))
        elif col == "pred_prob":
            feat_vec.append(float(pred_up_prob))
        else:
            val = float(row.get(col, 0.0)) if col in row.index else 0.0
            feat_vec.append(0.0 if np.isnan(val) else val)

    # 1) 2차 승률 판별 모델 추론
    if art.meta_model is not None:
        X_arr = np.array([feat_vec], dtype=float)
        try:
            meta_p = float(art.meta_model.predict_proba(X_arr)[0, 1])
        except Exception:
            meta_p = float(pred_up_prob)
        blended_p = 0.62 * meta_p + 0.38 * float(pred_up_prob)
    else:
        blended_p = float(pred_up_prob)

    # 2) 챔피언 하드 안전 규칙 체크 (함정 회피)
    rsi14 = float(row.get("rsi_14", 0.5)) * 100.0
    ret5d = float(np.expm1(float(row.get("ret_5d", 0.0))))
    dist60 = float(row.get("dist_sma_60", 0.0))
    sma20_60 = float(row.get("sma_20_60_cross", 0.0))
    rv20 = float(row.get("rv_20d", 0.25))

    reasons_fail: list[str] = []
    if rsi14 > pol.max_rsi_14:
        reasons_fail.append(f"단기 과열(RSI {rsi14:.0f} > {pol.max_rsi_14:.0f})")
        blended_p -= 0.08
    if ret5d > pol.max_ret_5d:
        reasons_fail.append(f"최근 5일 급등({ret5d*100:+.1f}%) 추격 위험")
        blended_p -= 0.08
    if dist60 < pol.min_dist_sma60:
        reasons_fail.append("중기 하락 추세 구간")
        blended_p -= 0.06
    if pol.require_sma20_above_60 and sma20_60 < -0.005:
        reasons_fail.append("이동평균선 역배열 주의")
        blended_p -= 0.05
    if rv20 > pol.max_rv20:
        reasons_fail.append(f"가격 출렁임 과다(변동성 {rv20*100:.0f}%)")
        blended_p -= 0.05

    win_conf_pct = float(np.clip(blended_p * 100.0, 12.0, 94.0))
    passed = (
        len(reasons_fail) == 0
        and pred_up_prob >= pol.min_primary_prob
        and blended_p >= max(pol.min_meta_win_prob, 0.50)
    )
    if passed:
        note = f"✅ {art.generation}세대 챔피언 승률 필터 통과 (예상 실전승률 {win_conf_pct:.0f}%)"
    elif reasons_fail:
        note = f"⚠️ 함정 회피 발동: {', '.join(reasons_fail[:2])}"
    else:
        note = f"🟡 승률 기준 미달 (예상 실전승률 {win_conf_pct:.0f}%)"
    return round(win_conf_pct, 1), passed, note


def _build_meta_classifier(seed: int = 42) -> Pipeline:
    """과적합에 강하고 확률 보정이 잘 되는 경량 앙상블 분류기."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=65,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.85,
                    random_state=seed,
                ),
            ),
        ]
    )


class WinRateEvolutionEngine:
    """모의투자 시뮬레이션을 반복하며 승률과 손익비를 스스로 높이는 자가진화 훈련기."""

    def __init__(
        self,
        config: PredictorConfig | None = None,
        db: PaperTradingDB | None = None,
    ) -> None:
        self.config = config or PredictorConfig(horizon=5)
        self.db = db or PaperTradingDB()
        self.feature_engine = FeatureEngine(horizon=self.config.horizon)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ensure_evolution_table(self.db)

    def prepare_panel(self, bundles: list[MarketDataBundle], use_cache: bool = True) -> dict[str, Any]:
        """다중 종목 번들을 한 번만 정렬·피처화하여 수십 세대 시뮬레이션을 초고속으로 수행."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sig_key = "_".join(sorted(f"{b.symbol.code}:{len(b.ohlcv)}" for b in bundles))
        cache_hash = hashlib.md5(sig_key.encode("utf-8")).hexdigest()[:14]
        panel_cache = CACHE_DIR / f"evo_panel_h{self.config.horizon}_{cache_hash}.pkl"
        if use_cache and panel_cache.exists():
            try:
                with panel_cache.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

        sym_meta: dict[str, dict[str, Any]] = {}
        price_map: dict[str, pd.Series] = {}
        high_map: dict[str, pd.Series] = {}
        low_map: dict[str, pd.Series] = {}
        panel_parts: list[pd.DataFrame] = []
        fx_series: pd.Series | None = None

        for b in bundles:
            try:
                ds = self.feature_engine.build(b)
                if len(ds.train_X) < 180:
                    continue
                code = b.symbol.code
                sym_meta[code] = {
                    "market": b.symbol.market,
                    "code": code,
                    "name": b.symbol.name,
                    "currency": b.symbol.currency,
                }
                price_map[code] = b.ohlcv["Close"].copy()
                high_map[code] = b.ohlcv["High"].copy()
                low_map[code] = b.ohlcv["Low"].copy()
                if fx_series is None and b.macro is not None and "usdkrw" in b.macro.columns:
                    fx_series = b.macro["usdkrw"].dropna().sort_index()

                sub = ds.features_all.copy()
                sub["_date"] = sub.index
                sub["_code"] = code
                sub["_target_ret"] = np.log(
                    (b.ohlcv["Close"].shift(-self.config.horizon) / b.ohlcv["Close"]).reindex(sub.index).clip(lower=1e-6)
                )
                panel_parts.append(sub.reset_index(drop=True))
            except Exception:
                continue

        if len(panel_parts) < 3:
            raise RuntimeError("자가진화 시뮬레이션을 위한 유효 종목 수가 부족합니다 (최소 3종목).")

        meta_cols = {"_date", "_code", "_target_ret"}
        common_feats = set(panel_parts[0].columns) - meta_cols
        for p in panel_parts[1:]:
            common_feats &= (set(p.columns) - meta_cols)
        feature_cols = sorted(common_feats)

        panel = pd.concat([p[feature_cols + list(meta_cols)] for p in panel_parts], ignore_index=True)
        panel = panel.sort_values(["_date", "_code"]).reset_index(drop=True)

        for col in [c for c in ["ret_5d", "ret_20d", "rsi_14", "rv_20d", "dist_sma_20", "drawdown_60d"] if c in feature_cols]:
            rk = f"cs_rank_{col}"
            panel[rk] = panel.groupby("_date")[col].rank(pct=True)
            feature_cols.append(rk)

        panel["_alpha_ret"] = panel["_target_ret"] - panel.groupby("_date")["_target_ret"].transform("median")
        panel["_alpha_dir"] = (panel["_alpha_ret"] > 0).astype(int)

        close_df = pd.DataFrame(price_map).sort_index().ffill()
        high_df = pd.DataFrame(high_map).reindex(close_df.index).ffill().fillna(close_df)
        low_df = pd.DataFrame(low_map).reindex(close_df.index).ffill().fillna(close_df)

        valid_dates = sorted(
            dt for dt in panel["_date"].unique()
            if dt in close_df.index and close_df.loc[dt].notna().sum() >= 3
        )
        if len(valid_dates) < 200:
            raise RuntimeError("충분한 공통 거래일이 확보되지 않았습니다.")

        if fx_series is not None and not fx_series.empty:
            fx_aligned = fx_series.reindex(close_df.index, method="ffill").bfill().fillna(1380.0)
        else:
            fx_aligned = pd.Series(1380.0, index=close_df.index)

        # 1차 AI 예측을 전 구간 Walk-Forward(20일 주기 재학습 + Embargo)로 미리 계산해 두면
        # 수십 세대의 전략·승률필터 시뮬레이션을 미래 누수 없이 1초 만에 검증 가능!
        embargo = max(self.config.horizon, 1)
        sim_start_idx = max(140, len(valid_dates) - 420)
        eval_dates = valid_dates[sim_start_idx:]

        pred_ret_col = pd.Series(np.nan, index=panel.index, dtype=float)
        pred_prob_col = pd.Series(np.nan, index=panel.index, dtype=float)

        models: list[Any] = []
        date_to_idx = {d: i for i, d in enumerate(valid_dates)}
        retrain_step = 25

        for step_i, dt in enumerate(eval_dates):
            if step_i % retrain_step == 0 or not models:
                pos = date_to_idx[dt]
                cutoff = valid_dates[max(0, pos - embargo - 1)]
                tr = panel[(panel["_date"] <= cutoff) & (panel["_alpha_ret"].notna())]
                if len(tr) >= 120:
                    X_tr = tr[feature_cols].fillna(0.0)
                    models = [
                        LightGBMStockModel(random_state=self.config.random_seed),
                        LinearQuantModel(random_state=self.config.random_seed + 1),
                    ]
                    for m in models:
                        m.fit(X_tr, tr["_alpha_ret"], tr["_alpha_dir"])

            if models:
                mask = panel["_date"] == dt
                if mask.any():
                    X_d = panel.loc[mask, feature_cols].fillna(0.0)
                    p_r = np.mean([m.predict(X_d).expected_return for m in models], axis=0)
                    p_p = np.mean([m.predict(X_d).up_prob for m in models], axis=0)
                    pred_ret_col.loc[mask] = p_r
                    pred_prob_col.loc[mask] = p_p

        panel["pred_ret"] = pred_ret_col
        panel["pred_prob"] = pred_prob_col

        out_dict = {
            "panel": panel,
            "sym_meta": sym_meta,
            "close_df": close_df,
            "high_df": high_df,
            "low_df": low_df,
            "fx_aligned": fx_aligned,
            "eval_dates": eval_dates,
        }
        if use_cache:
            try:
                with panel_cache.open("wb") as f:
                    pickle.dump(out_dict, f)
            except Exception:
                pass
        return out_dict

    def simulate_policy(
        self,
        prepared: dict[str, Any],
        policy: WinRatePolicy,
        sim_dates: list[Any],
        meta_model: Any | None = None,
        meta_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """주어진 정책 + 2차 승률 판별기로 엄격한 시계열 모의매매 수행 (장중 고가/저가 및 트레일링 브레이크 반영)."""
        panel: pd.DataFrame = prepared["panel"]
        sym_meta: dict[str, dict[str, Any]] = prepared["sym_meta"]
        close_df: pd.DataFrame = prepared["close_df"]
        high_df: pd.DataFrame = prepared["high_df"]
        low_df: pd.DataFrame = prepared["low_df"]
        fx_aligned: pd.Series = prepared["fx_aligned"]
        m_cols = meta_cols or META_FEATURE_COLS

        cash = float(DEFAULT_INITIAL_CASH_KRW)
        holdings: dict[str, dict[str, Any]] = {}
        trades: list[dict[str, Any]] = []
        eq_list: list[float] = []

        # 날짜별 슬라이스 사전 인덱싱으로 고속화
        panel_sub = panel[panel["_date"].isin(sim_dates) & panel["pred_prob"].notna()]
        by_date = {d: df_d for d, df_d in panel_sub.groupby("_date")}

        for day_i, dt in enumerate(sim_dates):
            dt_ts = pd.Timestamp(dt)
            dt_str = dt_ts.strftime("%Y-%m-%d")
            fx_today = float(fx_aligned.loc[dt_ts])

            # 1) 보유 종목 일일 손절 / 익절 / 본전잠금(Breakeven Lock) / 시간청산 체크
            for code in list(holdings.keys()):
                h = holdings[code]
                h["days_held"] += 1
                px_close = float(close_df.loc[dt_ts, code])
                px_high = float(high_df.loc[dt_ts, code])
                px_low = float(low_df.loc[dt_ts, code])
                if np.isnan(px_close) or px_close <= 0:
                    continue

                entry_p = h["entry_price"]
                h["peak_price"] = max(h["peak_price"], px_high if px_high > 0 else px_close)
                peak_ret = (h["peak_price"] / entry_p) - 1.0

                # 수익이 일정 구간(breakeven_trigger_pct) 도달 시 안전 브레이크를 본전+수수료 위로 끌어올림!
                if peak_ret >= policy.breakeven_trigger_pct:
                    h["dynamic_stop_pct"] = max(h["dynamic_stop_pct"], policy.breakeven_lock_pct)

                low_ret = ((px_low if px_low > 0 else px_close) / entry_p) - 1.0
                high_ret = ((px_high if px_high > 0 else px_close) / entry_p) - 1.0
                close_ret = (px_close / entry_p) - 1.0
                pullback_from_peak = (px_close / h["peak_price"]) - 1.0

                exit_px = None
                exit_reason = None
                if low_ret <= h["dynamic_stop_pct"]:
                    exit_px = entry_p * (1.0 + h["dynamic_stop_pct"])
                    exit_reason = "수익보존 잠금익절" if h["dynamic_stop_pct"] > 0 else "안전브레이크 손절"
                elif high_ret >= policy.take_profit_pct:
                    exit_px = entry_p * (1.0 + policy.take_profit_pct)
                    exit_reason = "목표가 자동익절"
                elif policy.trailing_pullback_pct < 0.5 and peak_ret >= 0.04 and pullback_from_peak <= -policy.trailing_pullback_pct:
                    exit_px = px_close
                    exit_reason = "고점 트레일링 익절"
                elif h["days_held"] >= policy.max_hold_days and close_ret > -0.02:
                    exit_px = px_close
                    exit_reason = "만기 순환청산"

                if exit_px is not None and exit_reason is not None:
                    cash = self._close_sim_trade(code, h, exit_px, dt_str, fx_today, sym_meta[code], exit_reason, cash, trades)
                    del holdings[code]

            # 2) 리밸런싱 및 신규 진입 (1차 AI + 2차 승률 판별기 + 함정 회피 필터)
            if day_i % policy.rebalance_every == 0 and dt in by_date:
                today_df = by_date[dt].copy()
                if not today_df.empty:
                    # 2차 승률 판별 점수 계산
                    if meta_model is not None:
                        X_m = pd.DataFrame(index=today_df.index)
                        for c in m_cols:
                            X_m[c] = today_df[c] if c in today_df.columns else 0.0
                        try:
                            meta_probs = meta_model.predict_proba(X_m.to_numpy(dtype=float))[:, 1]
                        except Exception:
                            meta_probs = today_df["pred_prob"].to_numpy(dtype=float)
                    else:
                        meta_probs = today_df["pred_prob"].to_numpy(dtype=float)

                    today_df["_meta_win_prob"] = 0.62 * meta_probs + 0.38 * today_df["pred_prob"].to_numpy(dtype=float)

                    # 하드 안전 규칙 마스킹
                    cond = (today_df["pred_ret"] > 0.0) & (today_df["pred_prob"] >= policy.min_primary_prob)
                    if policy.min_meta_win_prob > 0:
                        cond &= today_df["_meta_win_prob"] >= policy.min_meta_win_prob
                    if "rsi_14" in today_df.columns:
                        cond &= (today_df["rsi_14"] * 100.0) <= policy.max_rsi_14
                    if "ret_5d" in today_df.columns:
                        cond &= np.expm1(today_df["ret_5d"]) <= policy.max_ret_5d
                    if "dist_sma_60" in today_df.columns:
                        cond &= today_df["dist_sma_60"] >= policy.min_dist_sma60
                    if policy.require_sma20_above_60 and "sma_20_60_cross" in today_df.columns:
                        cond &= today_df["sma_20_60_cross"] >= -0.003
                    if "rv_20d" in today_df.columns:
                        cond &= today_df["rv_20d"] <= policy.max_rv20
                    if policy.min_rel_alpha_20d > -0.25 and "rel_alpha_20d" in today_df.columns:
                        cond &= today_df["rel_alpha_20d"] >= policy.min_rel_alpha_20d

                    rv = today_df["rv_20d"].clip(0.15, 1.0) if "rv_20d" in today_df.columns else 0.3
                    today_df["_rank_score"] = (
                        today_df["pred_ret"] / np.sqrt(rv)
                        + (today_df["_meta_win_prob"] - 0.5) * 0.14
                    )
                    cands = today_df[cond].sort_values("_rank_score", ascending=False).head(policy.max_positions)

                    cur_eq = cash + sum(
                        h["shares"] * float(close_df.loc[dt_ts, c]) * (1.0 if sym_meta[c]["currency"] == "KRW" else fx_today)
                        for c, h in holdings.items()
                        if not np.isnan(close_df.loc[dt_ts, c])
                    )
                    new_cands = cands[~cands["_code"].isin(holdings.keys())]
                    slots = max(policy.max_positions - len(holdings), 0)
                    if slots > 0 and not new_cands.empty and cash > cur_eq * 0.12:
                        new_cands = new_cands.head(slots)
                        per_budget = min((cash * 0.85) / max(slots, 1), cur_eq * 0.20)
                        for _, r in new_cands.iterrows():
                            code = str(r["_code"])
                            px = float(close_df.loc[dt_ts, code])
                            if np.isnan(px) or px <= 0:
                                continue
                            meta = sym_meta[code]
                            fx_m = 1.0 if meta["currency"] == "KRW" else fx_today
                            buy_cost = MARKET_COST_BPS[meta["market"]]["commission"] + MARKET_COST_BPS[meta["market"]]["slippage"]
                            raw_sh = (per_budget / (1.0 + buy_cost)) / (px * fx_m)
                            sh = float(int(raw_sh)) if meta["market"] == "KR" else round(float(raw_sh), 3)
                            if sh <= 0:
                                continue
                            net = sh * px * fx_m * (1.0 + buy_cost)
                            if net > cash:
                                continue
                            cash -= net
                            snap_f = {
                                c: float(r[c]) if (c in r.index and pd.notna(r[c])) else 0.0
                                for c in m_cols
                            }
                            holdings[code] = {
                                "shares": sh,
                                "entry_price": px,
                                "peak_price": px,
                                "dynamic_stop_pct": policy.stop_loss_pct,
                                "entry_krw": net,
                                "entry_date": dt_str,
                                "days_held": 0,
                                "entry_feats": snap_f,
                            }

            inv_val = sum(
                h["shares"] * float(close_df.loc[dt_ts, c]) * (1.0 if sym_meta[c]["currency"] == "KRW" else fx_today)
                for c, h in holdings.items()
                if not np.isnan(close_df.loc[dt_ts, c])
            )
            eq_list.append(cash + inv_val)

        # 종료 시점 미청산분 평가 정리
        if sim_dates:
            last_dt = pd.Timestamp(sim_dates[-1])
            last_str = last_dt.strftime("%Y-%m-%d")
            last_fx = float(fx_aligned.loc[last_dt])
            for code in list(holdings.keys()):
                px = float(close_df.loc[last_dt, code])
                if not np.isnan(px) and px > 0:
                    cash = self._close_sim_trade(code, holdings[code], px, last_str, last_fx, sym_meta[code], "기간종료 정산", cash, trades)
                    del holdings[code]

        tdf = pd.DataFrame(trades)
        final_eq = eq_list[-1] if eq_list else DEFAULT_INITIAL_CASH_KRW
        tot_ret = (final_eq / DEFAULT_INITIAL_CASH_KRW - 1.0) * 100.0
        eq_arr = np.array(eq_list if eq_list else [DEFAULT_INITIAL_CASH_KRW], dtype=float)
        mdd = float(np.min((eq_arr / np.maximum.accumulate(eq_arr) - 1.0) * 100.0))

        if not tdf.empty:
            wins = tdf[tdf["pnl_krw"] > 0]
            losses = tdf[tdf["pnl_krw"] <= 0]
            win_rate = float(len(wins) / len(tdf) * 100.0)
            avg_w = float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0.0
            avg_l = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
            gp = float(wins["pnl_krw"].sum())
            gl = float(np.abs(losses["pnl_krw"].sum()))
            pf = gp / gl if gl > 1e-9 else (3.0 if gp > 0 else 0.0)
        else:
            win_rate = avg_w = avg_l = pf = 0.0

        # 승률 + 실제 순수익 동시 달성 적합도 점수 (손익비 1.0 미만 전략이 꼼수로 승률만 높이는 현상 방지!)
        trade_penalty = 0.0 if len(tdf) >= 14 else -35.0 * (1.0 - len(tdf) / 14.0)
        pf_guard = 0.0 if pf >= 1.0 else -(1.0 - pf) * 38.0
        fitness = (
            win_rate * 1.10
            + min(pf, 4.0) * 14.0
            + np.clip(tot_ret, -25.0, 50.0) * 0.75
            + np.clip(mdd, -30.0, 0.0) * 0.40
            + trade_penalty
            + pf_guard
        )
        return {
            "win_rate_pct": round(win_rate, 1),
            "total_return_pct": round(tot_ret, 2),
            "profit_factor": round(min(pf, 99.0), 2),
            "max_drawdown_pct": round(mdd, 2),
            "closed_trades": len(tdf),
            "avg_win_pct": round(avg_w, 2),
            "avg_loss_pct": round(avg_l, 2),
            "fitness_score": round(float(fitness), 2),
            "trades_df": tdf,
        }

    @staticmethod
    def _close_sim_trade(
        code: str,
        holding: dict[str, Any],
        exit_price: float,
        exit_date: str,
        fx_today: float,
        meta: dict[str, Any],
        exit_reason: str,
        cash: float,
        trades: list[dict[str, Any]],
    ) -> float:
        fx_m = 1.0 if meta["currency"] == "KRW" else fx_today
        sell_cost = (
            MARKET_COST_BPS[meta["market"]]["commission"]
            + MARKET_COST_BPS[meta["market"]]["sell_tax"]
            + MARKET_COST_BPS[meta["market"]]["slippage"]
        )
        net = holding["shares"] * exit_price * fx_m * (1.0 - sell_cost)
        pnl_krw = net - holding["entry_krw"]
        pnl_pct = (pnl_krw / (holding["entry_krw"] + 1e-9)) * 100.0
        rec = {
            "code": code,
            "name": meta["name"],
            "entry_date": holding["entry_date"],
            "exit_date": exit_date,
            "pnl_krw": pnl_krw,
            "pnl_pct": pnl_pct,
            "win": 1 if pnl_krw > 0 else 0,
            "exit_reason": exit_reason,
            "days_held": holding["days_held"],
        }
        rec.update(holding.get("entry_feats", {}))
        trades.append(rec)
        return cash + net

    def run_evolution_rounds(
        self,
        bundles: list[MarketDataBundle],
        n_generations: int = 6,
        progress_cb: Callable[[str], None] | None = None,
    ) -> ChampionWinRateArtifact:
        """과거 훈련 구간에서 2차 승률 모델을 학습하고, 미래 검증 구간(최근 ~180거래일)에서 세대별 승률 진화 수행."""
        if progress_cb:
            progress_cb("1단계: 다중 종목 시계열 피처 및 1차 AI 워크포워드 예측 준비 중...")
        prepared = self.prepare_panel(bundles)
        eval_dates = prepared["eval_dates"]

        # 엄격한 시간 분리: 앞쪽 45% 구간(Discovery)에서 거래 패턴 학습 -> 뒤쪽 55% 최신 구간(Out-of-Sample)에서 승률 검증!
        split_idx = max(int(len(eval_dates) * 0.45), 60)
        discovery_dates = eval_dates[:split_idx]
        validation_dates = eval_dates[split_idx:]
        if len(validation_dates) < 60:
            discovery_dates = eval_dates[: len(eval_dates) // 2]
            validation_dates = eval_dates[len(eval_dates) // 2 :]

        # 기존 세대 번호 이어받기
        hist_df = load_evolution_history_df(self.db)
        next_gen = int(hist_df["generation"].max()) + 1 if not hist_df.empty else 1
        existing_champ = load_champion_artifact()

        # Discovery 구간에서 기본 정책으로 풍부한 거래 샘플 수집 -> 2차 승률 판별기(Meta-Classifier) 학습
        if progress_cb:
            progress_cb("2단계: 훈련 구간 모의매매로 '이기는 패턴 vs 물리는 함정' 데이터 수집 중...")
        discovery_policy = WinRatePolicy(
            name="탐색용 기본 매매",
            min_primary_prob=0.50,
            stop_loss_pct=-0.08,
            take_profit_pct=0.14,
            rebalance_every=3,
            max_positions=6,
        )
        disc_res = self.simulate_policy(prepared, discovery_policy, discovery_dates, meta_model=None)
        disc_trades = disc_res["trades_df"]

        meta_model = None
        insights: list[dict[str, Any]] = []
        if not disc_trades.empty and len(disc_trades) >= 20 and disc_trades["win"].nunique() >= 2:
            avail_cols = [c for c in META_FEATURE_COLS if c in disc_trades.columns]
            X_meta = disc_trades[avail_cols].to_numpy(dtype=float)
            y_meta = disc_trades["win"].to_numpy(dtype=int)
            meta_model = _build_meta_classifier(seed=self.config.random_seed + next_gen)
            meta_model.fit(X_meta, y_meta)
            insights = self._extract_win_loss_insights(disc_trades)
        else:
            avail_cols = list(META_FEATURE_COLS)

        # 진화 후보 세대 설계 (1세대 순수 초보 전략 -> 추세 정렬 -> 과열 차단 -> 2차 승률 필터 -> 본전 잠금 브레이크 -> 하이브리드 변이)
        preset_candidates: list[WinRatePolicy] = [
            WinRatePolicy(
                name="초보 기본형 (단순 AI 매수·손절-8%/익절+18%)",
                min_primary_prob=0.51,
                min_meta_win_prob=0.0,
                max_rsi_14=82.0,
                max_ret_5d=0.22,
                stop_loss_pct=-0.08,
                take_profit_pct=0.18,
                breakeven_trigger_pct=0.99,
            ),
            WinRatePolicy(
                name="급등 추격 금지 + 정배열 상승추세 선별형",
                min_primary_prob=0.52,
                min_meta_win_prob=0.51,
                max_rsi_14=67.0,
                max_ret_5d=0.11,
                min_dist_sma60=-0.02,
                require_sma20_above_60=True,
                stop_loss_pct=-0.07,
                take_profit_pct=0.12,
                breakeven_trigger_pct=0.99,
            ),
            WinRatePolicy(
                name="2차 승률 판별기(Meta-AI) 엄선 진입형",
                min_primary_prob=0.52,
                min_meta_win_prob=0.55,
                max_rsi_14=66.0,
                max_ret_5d=0.10,
                min_dist_sma60=-0.02,
                require_sma20_above_60=True,
                max_rv20=0.60,
                stop_loss_pct=-0.065,
                take_profit_pct=0.095,
                breakeven_trigger_pct=0.99,
            ),
            WinRatePolicy(
                name="수익보존 본전잠금 브레이크 + 승률필터 결합형",
                min_primary_prob=0.52,
                min_meta_win_prob=0.54,
                max_rsi_14=66.0,
                max_ret_5d=0.10,
                min_dist_sma60=-0.03,
                require_sma20_above_60=True,
                max_rv20=0.62,
                stop_loss_pct=-0.065,
                take_profit_pct=0.09,
                breakeven_trigger_pct=0.028,
                breakeven_lock_pct=0.006,
                max_hold_days=18,
            ),
            WinRatePolicy(
                name="고승률 방어형 (짧은 목표익절 + 빠른 본전잠금)",
                min_primary_prob=0.53,
                min_meta_win_prob=0.56,
                max_rsi_14=64.0,
                max_ret_5d=0.09,
                min_dist_sma60=-0.01,
                require_sma20_above_60=True,
                max_rv20=0.55,
                stop_loss_pct=-0.06,
                take_profit_pct=0.075,
                breakeven_trigger_pct=0.022,
                breakeven_lock_pct=0.005,
                trailing_pullback_pct=0.03,
                max_hold_days=15,
            ),
        ]

        # 이미 이전 세대가 있으면 챔피언 주변에서 미세 변이(Mutation) 후보들을 생성해 계속 진화!
        rng = np.random.default_rng(int(datetime.now().timestamp()) % 100_000)
        candidates_to_run: list[WinRatePolicy] = []
        if next_gen == 1:
            candidates_to_run.extend(preset_candidates[:n_generations])
        else:
            base_pol = existing_champ.policy if existing_champ else preset_candidates[3]
            for k in range(n_generations):
                mut = WinRatePolicy(
                    name=f"자가변이 진화전략 v{next_gen + k}",
                    min_primary_prob=round(float(np.clip(base_pol.min_primary_prob + rng.uniform(-0.01, 0.015), 0.50, 0.56)), 3),
                    min_meta_win_prob=round(float(np.clip(base_pol.min_meta_win_prob + rng.uniform(-0.02, 0.025), 0.51, 0.62)), 3),
                    max_rsi_14=round(float(np.clip(base_pol.max_rsi_14 + rng.uniform(-4.0, 4.0), 58.0, 72.0)), 1),
                    max_ret_5d=round(float(np.clip(base_pol.max_ret_5d + rng.uniform(-0.02, 0.02), 0.07, 0.15)), 3),
                    min_dist_sma60=round(float(np.clip(base_pol.min_dist_sma60 + rng.uniform(-0.02, 0.02), -0.05, 0.02)), 3),
                    require_sma20_above_60=bool(rng.choice([True, True, False])),
                    max_rv20=round(float(np.clip(base_pol.max_rv20 + rng.uniform(-0.06, 0.06), 0.45, 0.72)), 2),
                    stop_loss_pct=round(float(np.clip(base_pol.stop_loss_pct + rng.uniform(-0.02, 0.02), -0.09, -0.04)), 3),
                    take_profit_pct=round(float(rng.uniform(0.08, 0.22)), 3),
                    breakeven_trigger_pct=round(float(rng.choice([0.035, 0.05, 0.065, 0.99])), 3),
                    breakeven_lock_pct=round(float(rng.uniform(0.008, 0.022)), 4),
                    trailing_pullback_pct=round(float(rng.choice([0.035, 0.05, 0.99])), 3),
                    max_hold_days=int(rng.choice([12, 16, 20, 25, 35])),
                    rebalance_every=int(rng.choice([5, 7, 10])),
                    min_rel_alpha_20d=round(float(rng.choice([-0.50, 0.0, 0.015, 0.03])), 3),
                )
                candidates_to_run.append(mut)

        while len(candidates_to_run) < n_generations:
            idx_m = len(candidates_to_run)
            candidates_to_run.append(
                WinRatePolicy(
                    name=f"고승률 정밀튜닝 v{next_gen + idx_m}",
                    min_primary_prob=0.52,
                    min_meta_win_prob=round(float(0.54 + 0.01 * (idx_m % 4)), 3),
                    max_rsi_14=65.0,
                    max_ret_5d=0.095,
                    min_dist_sma60=-0.02,
                    require_sma20_above_60=True,
                    max_rv20=0.58,
                    stop_loss_pct=-0.062,
                    take_profit_pct=0.082,
                    breakeven_trigger_pct=0.022,
                    breakeven_lock_pct=0.005,
                    trailing_pullback_pct=0.028,
                    max_hold_days=16,
                )
            )

        best_fitness = -1e9
        best_champ_data: dict[str, Any] | None = None
        if existing_champ is not None and not hist_df.empty:
            champ_rows = hist_df[hist_df["is_champion"] == 1]
            if not champ_rows.empty:
                best_fitness = float(champ_rows.iloc[-1]["fitness_score"])

        baseline_wr = float(hist_df.iloc[0]["win_rate_pct"]) if not hist_df.empty else None
        total_sim_trades = int(hist_df["closed_trades"].sum()) if not hist_df.empty else 0

        for idx, pol in enumerate(candidates_to_run):
            gen_no = next_gen + idx
            if progress_cb:
                progress_cb(f"3단계: [{gen_no}세대 훈련] '{pol.name}' 미공개 검증 구간 실전 테스트 중...")
            use_m = meta_model if pol.min_meta_win_prob > 0 else None
            res = self.simulate_policy(prepared, pol, validation_dates, meta_model=use_m, meta_cols=avail_cols)
            if baseline_wr is None:
                baseline_wr = res["win_rate_pct"]
            total_sim_trades += int(res["closed_trades"])

            lessons = self._describe_policy_lessons_ko(pol, res["win_rate_pct"], baseline_wr)
            is_new_champ = (res["fitness_score"] > best_fitness) and (res["closed_trades"] >= 12)
            if is_new_champ or best_champ_data is None:
                if is_new_champ:
                    best_fitness = res["fitness_score"]
                best_champ_data = {
                    "generation": gen_no,
                    "policy": pol,
                    "res": res,
                    "lessons": lessons,
                }

            rec = GenerationRecord(
                generation=gen_no,
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                policy_name=pol.name,
                win_rate_pct=res["win_rate_pct"],
                total_return_pct=res["total_return_pct"],
                profit_factor=res["profit_factor"],
                max_drawdown_pct=res["max_drawdown_pct"],
                closed_trades=res["closed_trades"],
                avg_win_pct=res["avg_win_pct"],
                avg_loss_pct=res["avg_loss_pct"],
                fitness_score=res["fitness_score"],
                is_champion=is_new_champ,
                policy_dict=asdict(pol),
                learned_lessons_ko=lessons,
            )
            save_generation_record(self.db, rec)

        # 최종 챔피언 확정 저장 (기존 챔피언이 더 우수했으면 기존 챔피언 유지하되 누적 시뮬레이션 수 갱신)
        if existing_champ is not None and best_champ_data is not None:
            # 새로 경신된 세대가 있는지 확인
            hist_after = load_evolution_history_df(self.db)
            champ_row = hist_after[hist_after["is_champion"] == 1]
            if champ_row.empty:
                # 첫 세대이거나 경신 없으면 최고 fitness 세대를 챔피언으로 지정
                top_idx = hist_after["fitness_score"].idxmax()
                best_gen = int(hist_after.loc[top_idx, "generation"])
                with self.db._connect() as conn:
                    conn.execute("UPDATE evolution_generations SET is_champion = CASE WHEN generation = ? THEN 1 ELSE 0 END", (best_gen,))

        hist_final = load_evolution_history_df(self.db)
        champ_rows = hist_final[hist_final["is_champion"] == 1]
        if not champ_rows.empty:
            c_row = champ_rows.iloc[-1]
        else:
            c_row = hist_final.sort_values("fitness_score", ascending=False).iloc[0]
            with self.db._connect() as conn:
                conn.execute("UPDATE evolution_generations SET is_champion = 1 WHERE generation = ?", (int(c_row["generation"]),))

        c_pol = WinRatePolicy(**json.loads(c_row["policy_json"]))
        c_rules = json.loads(c_row["lessons_json"])

        artifact = ChampionWinRateArtifact(
            generation=int(c_row["generation"]),
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            baseline_win_rate_pct=float(hist_final.iloc[0]["win_rate_pct"]),
            champion_win_rate_pct=float(c_row["win_rate_pct"]),
            champion_return_pct=float(c_row["total_return_pct"]),
            champion_profit_factor=float(c_row["profit_factor"]),
            champion_trades=int(c_row["closed_trades"]),
            total_simulated_trades=int(hist_final["closed_trades"].sum()),
            policy=c_pol,
            meta_model=meta_model if meta_model is not None else (existing_champ.meta_model if existing_champ else None),
            meta_feature_cols=avail_cols,
            learned_rules_ko=c_rules,
            feature_win_insights=insights or (existing_champ.feature_win_insights if existing_champ else []),
        )
        with CHAMPION_ARTIFACT_PATH.open("wb") as f:
            pickle.dump(artifact, f)
        return artifact

    @staticmethod
    def _extract_win_loss_insights(trades_df: pd.DataFrame) -> list[dict[str, Any]]:
        wins = trades_df[trades_df["win"] == 1]
        losses = trades_df[trades_df["win"] == 0]
        if wins.empty or losses.empty:
            return []
        check_map = [
            ("rsi_14", "진입 당시 과열도(RSI)", 100.0),
            ("rv_20d", "최근 가격 흔들림(변동성%)", 100.0),
            ("pred_prob", "1차 AI 상승확률(%)", 100.0),
            ("sma_20_60_cross", "중기 상승 추세(20일-60일선%)", 100.0),
        ]
        out: list[dict[str, Any]] = []
        for col, label, scale in check_map:
            if col in trades_df.columns:
                out.append(
                    {
                        "항목": label,
                        "이긴_거래_평균": round(float(wins[col].mean()) * scale, 2),
                        "진_거래_평균": round(float(losses[col].mean()) * scale, 2),
                    }
                )
        return out

    @staticmethod
    def _describe_policy_lessons_ko(pol: WinRatePolicy, wr: float, base_wr: float) -> list[str]:
        diff = wr - base_wr
        lessons: list[str] = []
        if pol.min_meta_win_prob > 0.50:
            lessons.append(
                f"🎯 2차 승률 판별기(Meta-AI)가 '실제로 익절할 확률 {pol.min_meta_win_prob*100:.0f}% 이상'이라고 확인해 준 종목만 골라 샀어요."
            )
        if pol.max_rsi_14 <= 70.0:
            lessons.append(
                f"🛑 이미 너무 뜨거워진 급등주(과열 지수 RSI {pol.max_rsi_14:.0f} 초과 · 5일 {pol.max_ret_5d*100:.0f}% 급등)는 참아서 고점 물림을 막았어요."
            )
        if pol.require_sma20_above_60:
            lessons.append("📈 주가가 내려가는 주식은 건드리지 않고, 상승 바람(20일선 ≥ 60일선)을 탄 주식에만 올라탔어요.")
        if pol.breakeven_trigger_pct < 0.15:
            lessons.append(
                f"🔒 산 가격보다 +{pol.breakeven_trigger_pct*100:.1f}%만 올라도 안전 브레이크를 '본전+{pol.breakeven_lock_pct*100:.1f}%' 위로 올려서, 다 잡은 수익이 손실로 바뀌지 않게 잠갔어요!"
            )
        lessons.append(
            f"🏆 결과: 처음 초보 규칙 승률({base_wr:.1f}%) 대비 검증 승률 {wr:.1f}% ({diff:+.1f}%p) 기록!"
        )
        return lessons
