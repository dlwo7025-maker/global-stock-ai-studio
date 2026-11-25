"""Qlib 스타일 다중 종목 통합(Cross-Sectional) 알파 학습 & 자동 피처 선별 엔진.

핵심 장점 (단일 종목 모델 대비 예측력 개선 원리):
1. 데이터 규모 20배+ 확장: 대표 20종목 × 5년 = ~20,000+ 샘플로 통합 학습해 과적합 대폭 억제.
2. Excess Alpha 타겟: 개별 종목 절대수익률 대신 '유니버스 평균(시장) 대비 초과수익률'을 예측해
   시장 전체 흔들림 노이즈를 제거.
3. Cross-Sectional 상대 순위 피처: 매일 전체 종목 사이에서 각 종목의 모멘텀·변동성·RSI·이격도
   백분위 순위(0~1)를 계산해 주도주/과매도 패턴을 포착.
4. Purged 자동 피처 선별: Out-of-Sample 신호가 약한 노이즈 피처를 자동 제거하고 Top-K 핵심 피처만 학습.
5. 정직한 Top-K 알파 포트폴리오 Walk-Forward 백테스트: 매일 예측 상위 K개 종목 균등투자 vs 시장 평균 비교.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import MARKET_COST_BPS, MODELS_DIR, PredictorConfig
from ..data.features import FeatureEngine
from ..data.market_data import MarketDataBundle
from .tabular import (
    ExtraTreesStockModel,
    LightGBMStockModel,
    LinearQuantModel,
    XGBoostStockModel,
)

# 일별 Cross-Sectional 백분위 순위를 생성할 핵심 상대 팩터
CS_RANK_SOURCE_COLS: list[str] = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "dist_sma_20",
    "dist_sma_60",
    "rsi_14",
    "rv_20d",
    "vol_ratio_5_20",
    "drawdown_60d",
    "rel_alpha_20d",
]


@dataclass
class CrossSectionalMetrics:
    """통합 알파 모델의 Purged Walk-Forward OOF 성과 및 Top-K 포트폴리오 벤치마크."""

    market: str
    horizon: int
    n_symbols: int
    n_samples: int
    n_selected_features: int
    trained_at: str
    symbols_used: list[str]
    selected_features: list[str]
    # 예측력 지표
    mean_daily_rank_ic: float       # 일별 Cross-Sectional Rank IC 평균 (0.03 이상 우수, 0.05+ 탁월)
    ic_ir: float                    # IC Information Ratio (평균 IC / IC 표준편차)
    excess_dir_accuracy: float      # 시장 초과수익 방향 적중률 (Outperform Hit Ratio)
    top_bottom_spread_pct: float    # 예측 상위 20% vs 하위 20% 평균 H일 수익률 스프레드 (%)
    # Top-K 롱-온리 알파 포트폴리오 vs 유니버스 동일가중 시장평균 (거래비용·세금 반영)
    top_k: int
    portfolio_total_return: float
    market_total_return: float
    excess_total_return: float
    portfolio_sharpe: float
    market_sharpe: float
    portfolio_mdd: float
    market_mdd: float
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class CrossSectionalArtifact:
    """저장 및 재사용 가능한 학습 완료 Cross-Sectional 알파 모델 묶음."""

    market: str
    horizon: int
    models: list[Any]
    model_weights: dict[str, float]
    selected_features: list[str]
    rank_reference: dict[str, np.ndarray]  # 단일 종목 추론 시 백분위 변환용 정렬 참조 배열
    metrics: CrossSectionalMetrics
    top_feature_importance: list[tuple[str, float]] = field(default_factory=list)


def model_artifact_path(market: str, horizon: int) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"cs_alpha_{market.upper()}_h{horizon}.pkl"


def load_cross_sectional_artifact(market: str, horizon: int) -> CrossSectionalArtifact | None:
    path = model_artifact_path(market, horizon)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, CrossSectionalArtifact) else None
    except Exception:
        return None


def save_cross_sectional_artifact(artifact: CrossSectionalArtifact) -> Path:
    path = model_artifact_path(artifact.market, artifact.horizon)
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    return path


def predict_single_with_artifact(
    artifact: CrossSectionalArtifact,
    latest_X: pd.DataFrame,
    bm_expected_drift: float = 0.0,
) -> dict[str, float]:
    """학습된 Cross-Sectional 통합 알파 모델로 단일 종목의 최신 피처 추론."""
    row = latest_X.copy()

    # 저장된 유니버스 분포 참조표를 이용해 Cross-Sectional 백분위 순위 피처 복원
    for col in CS_RANK_SOURCE_COLS:
        rk_col = f"cs_rank_{col}"
        if col in row.columns and col in artifact.rank_reference:
            ref = artifact.rank_reference[col]
            val = float(row[col].iloc[0])
            if np.isnan(val) or len(ref) == 0:
                row[rk_col] = 0.5
            else:
                pos = float(np.searchsorted(ref, val, side="right")) / len(ref)
                row[rk_col] = np.clip(pos, 0.0, 1.0)
        else:
            row[rk_col] = 0.5

    # 누락 피처 안전 보정
    for feat in artifact.selected_features:
        if feat not in row.columns:
            row[feat] = 0.0
    X_in = row[artifact.selected_features].fillna(0.0)

    alpha_ret = 0.0
    outperform_prob = 0.0
    w_sum = 0.0
    for model in artifact.models:
        w = artifact.model_weights.get(model.name, 1.0 / len(artifact.models))
        pred = model.predict(X_in)
        alpha_ret += w * float(pred.expected_return[0])
        outperform_prob += w * float(pred.up_prob[0])
        w_sum += w

    if w_sum > 0:
        alpha_ret /= w_sum
        outperform_prob /= w_sum

    return {
        "expected_alpha_log_return": float(alpha_ret),
        "expected_total_log_return": float(alpha_ret + bm_expected_drift),
        "outperform_prob": float(np.clip(outperform_prob, 0.05, 0.95)),
    }


class CrossSectionalAlphaTrainer:
    """국내·해외 다중 종목 통합 알파 모델 학습 및 OOF Top-K 포트폴리오 검증기."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config
        self.feature_engine = FeatureEngine(horizon=config.horizon)

    def _create_panel_models(self) -> list[Any]:
        return [
            LightGBMStockModel(random_state=self.config.random_seed),
            XGBoostStockModel(random_state=self.config.random_seed + 11),
            LinearQuantModel(random_state=self.config.random_seed + 22),
            ExtraTreesStockModel(random_state=self.config.random_seed + 33),
        ]

    def train_from_bundles(
        self,
        bundles: list[MarketDataBundle],
        market: str,
        save: bool = True,
        progress_cb: Callable[[str], None] | None = None,
    ) -> CrossSectionalArtifact:
        """여러 종목의 MarketDataBundle로 통합 패널 구축 -> 피처 선별 -> Walk-Forward 검증 -> 최종 학습."""
        if len(bundles) < 4:
            raise ValueError(f"통합 알파 학습에는 최소 4개 이상의 종목 데이터가 필요합니다 (현재: {len(bundles)}개).")

        if progress_cb:
            progress_cb(f"[{market}] {len(bundles)}개 종목 퀀트 피처 패널 구축 중...")

        panel_frames: list[pd.DataFrame] = []
        symbols_used: list[str] = []

        for b in bundles:
            try:
                ds = self.feature_engine.build(b)
                if len(ds.train_X) < 120:
                    continue
                sub = ds.train_X.copy()
                sub["_date"] = sub.index
                sub["_symbol"] = b.symbol.code
                sub["_close"] = ds.train_close.to_numpy(dtype=float)
                sub["_daily_next_ret"] = b.ohlcv["Close"].pct_change().shift(-1).reindex(sub.index).fillna(0.0).to_numpy(dtype=float)
                sub["_raw_target_ret"] = ds.train_y_ret.to_numpy(dtype=float)
                panel_frames.append(sub.reset_index(drop=True))
                symbols_used.append(b.symbol.code)
            except Exception:
                continue

        if len(panel_frames) < 4:
            raise RuntimeError("유효한 피처 데이터셋을 만든 종목 수가 부족합니다.")

        # 공통 피처 교집합 추출 (모든 종목에 안정적으로 존재하는 피처만 사용)
        meta_cols = {"_date", "_symbol", "_close", "_daily_next_ret", "_raw_target_ret"}
        common_feats = set(panel_frames[0].columns) - meta_cols
        for pf in panel_frames[1:]:
            common_feats &= (set(pf.columns) - meta_cols)
        base_feature_cols = sorted(common_feats)

        panel = pd.concat(
            [pf[base_feature_cols + list(meta_cols)] for pf in panel_frames],
            ignore_index=True,
        )
        panel = panel.sort_values(["_date", "_symbol"]).reset_index(drop=True)

        # 동일 날짜에 최소 4개 이상 종목이 존재하는 거래일만 유지
        date_counts = panel.groupby("_date")["_symbol"].transform("count")
        panel = panel.loc[date_counts >= min(4, len(panel_frames))].reset_index(drop=True)

        # 1) 일별 Cross-Sectional 백분위 순위 피처 생성 (0~1)
        cs_cols: list[str] = []
        for col in CS_RANK_SOURCE_COLS:
            if col in panel.columns:
                rk_col = f"cs_rank_{col}"
                panel[rk_col] = panel.groupby("_date")[col].rank(pct=True)
                cs_cols.append(rk_col)

        all_feature_cols = base_feature_cols + cs_cols

        # 2) Excess Alpha 타겟: 당일 유니버스 중앙값(시장) 대비 초과 H일 수익률
        day_median_ret = panel.groupby("_date")["_raw_target_ret"].transform("median")
        panel["_alpha_ret"] = panel["_raw_target_ret"] - day_median_ret
        panel["_alpha_dir"] = (panel["_alpha_ret"] > 0.0).astype(int)

        # 시간순 고유 거래일 기준 Purged Walk-Forward 분할
        unique_dates = np.sort(panel["_date"].unique())
        n_dates = len(unique_dates)
        embargo = max(self.config.horizon, 1)
        min_train_dates = min(max(int(n_dates * 0.45), 140), n_dates - 60)
        avail_val_dates = n_dates - min_train_dates - embargo
        n_splits = max(2, min(4, avail_val_dates // 40))
        fold_dates = max(avail_val_dates // n_splits, 20)

        date_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_splits):
            val_s = min_train_dates + embargo + k * fold_dates
            val_e = n_dates if k == n_splits - 1 else min(val_s + fold_dates, n_dates)
            tr_e = val_s - embargo
            if tr_e >= 80 and val_e - val_s >= 15:
                date_splits.append((unique_dates[:tr_e], unique_dates[val_s:val_e]))

        # 3) Purged 자동 피처 선별 (첫 번째 학습 구간 데이터만 사용해 누수 방지!)
        if progress_cb:
            progress_cb(f"[{market}] 고신호 퀀트 팩터 자동 선별 중 ({len(all_feature_cols)}개 -> Top {self.config.top_k_features}개)...")
        first_tr_mask = panel["_date"].isin(date_splits[0][0])
        selected_features = self._select_top_features(
            panel.loc[first_tr_mask, all_feature_cols],
            panel.loc[first_tr_mask, "_alpha_ret"],
            panel.loc[first_tr_mask, "_alpha_dir"],
            top_k=min(self.config.top_k_features, len(all_feature_cols)),
        )

        # 4) Purged Walk-Forward Out-of-Sample 검증
        if progress_cb:
            progress_cb(f"[{market}] {len(panel):,}행 패널 데이터 Purged Walk-Forward OOF 검증 중...")
        model_names = [m.name for m in self._create_panel_models()]
        oof_preds: dict[str, pd.Series] = {
            n: pd.Series(np.nan, index=panel.index, dtype=float) for n in model_names
        }
        oof_probs: dict[str, pd.Series] = {
            n: pd.Series(np.nan, index=panel.index, dtype=float) for n in model_names
        }

        for tr_dates, va_dates in date_splits:
            tr_mask = panel["_date"].isin(tr_dates)
            va_mask = panel["_date"].isin(va_dates)
            X_tr = panel.loc[tr_mask, selected_features]
            yr_tr = panel.loc[tr_mask, "_alpha_ret"]
            yd_tr = panel.loc[tr_mask, "_alpha_dir"]
            X_va = panel.loc[va_mask, selected_features]

            for model in self._create_panel_models():
                model.fit(X_tr, yr_tr, yd_tr)
                pred = model.predict(X_va)
                oof_preds[model.name].loc[va_mask] = pred.expected_return
                oof_probs[model.name].loc[va_mask] = pred.up_prob

        # 5) 모델별 일별 Cross-Sectional Rank IC로 가중치 계산
        valid_oof_mask = oof_preds[model_names[0]].notna()
        oof_panel = panel.loc[valid_oof_mask, ["_date", "_symbol", "_raw_target_ret", "_alpha_ret", "_alpha_dir", "_daily_next_ret"]].copy()

        model_ic_scores: dict[str, float] = {}
        for name in model_names:
            oof_panel[f"pred_{name}"] = oof_preds[name].loc[valid_oof_mask].to_numpy()
            oof_panel[f"prob_{name}"] = oof_probs[name].loc[valid_oof_mask].to_numpy()
            d_ics = self._compute_daily_ics(oof_panel, f"pred_{name}", "_alpha_ret")
            model_ic_scores[name] = float(np.mean(d_ics)) if len(d_ics) > 0 else 0.0

        ic_arr = np.array([model_ic_scores[n] for n in model_names], dtype=float)
        ic_arr = ic_arr - np.max(ic_arr)
        soft_w = np.exp(ic_arr * 25.0)
        soft_w = soft_w / np.sum(soft_w)
        unif_w = np.ones_like(soft_w) / len(soft_w)
        weights = {n: float(0.6 * sw + 0.4 * uw) for n, sw, uw in zip(model_names, soft_w, unif_w)}

        # OOF 백테스트에는 미래 가중치 누수를 막기 위해 균등 평균 적용
        oof_panel["ens_pred"] = np.mean([oof_panel[f"pred_{n}"].to_numpy() for n in model_names], axis=0)
        oof_panel["ens_prob"] = np.mean([oof_panel[f"prob_{n}"].to_numpy() for n in model_names], axis=0)

        daily_ics = self._compute_daily_ics(oof_panel, "ens_pred", "_alpha_ret")
        mean_ic = float(np.mean(daily_ics)) if len(daily_ics) > 0 else 0.0
        ic_std = float(np.std(daily_ics)) if len(daily_ics) > 1 else 1.0
        ic_ir = float(mean_ic / (ic_std + 1e-9))
        excess_dir_acc = float(np.mean((oof_panel["ens_prob"] >= 0.5).astype(int) == oof_panel["_alpha_dir"]))

        # 상위 20% vs 하위 20% H일 수익률 스프레드
        spreads: list[float] = []
        for _, grp in oof_panel.groupby("_date"):
            if len(grp) < 5:
                continue
            q = max(1, len(grp) // 5)
            grp_s = grp.sort_values("ens_pred")
            bot = float(grp_s["_raw_target_ret"].iloc[:q].mean())
            top = float(grp_s["_raw_target_ret"].iloc[-q:].mean())
            spreads.append(top - bot)
        spread_pct = float(np.mean(spreads) * 100.0) if spreads else 0.0

        # 6) Top-K 알파 롱-온리 포트폴리오 백테스트 (거래비용·세금 반영)
        top_k = min(5, max(2, len(symbols_used) // 4))
        port_stats = self._simulate_topk_portfolio(oof_panel, top_k=top_k, market=market)

        # 7) 전체 패널 데이터로 최종 앙상블 모델 적합
        if progress_cb:
            progress_cb(f"[{market}] 전체 {len(panel):,}행 최종 Cross-Sectional 앙상블 모델 학습 중...")
        final_models = self._create_panel_models()
        X_full = panel[selected_features]
        yr_full = panel["_alpha_ret"]
        yd_full = panel["_alpha_dir"]

        imp_acc: dict[str, float] = {f: 0.0 for f in selected_features}
        imp_w_sum = 0.0
        for model in final_models:
            model.fit(X_full, yr_full, yd_full)
            f_imp = model.get_feature_importance(selected_features)
            if f_imp:
                w = weights[model.name]
                for fname, val in f_imp.items():
                    imp_acc[fname] = imp_acc.get(fname, 0.0) + w * val
                imp_w_sum += w

        top_imp = (
            sorted([(k, v / imp_w_sum) for k, v in imp_acc.items()], key=lambda x: x[1], reverse=True)[:15]
            if imp_w_sum > 0
            else []
        )

        # 최근 60거래일 피처 분포 참조표 저장 (단일 종목 추론 시 백분위 변환용)
        recent_cutoff = unique_dates[max(0, len(unique_dates) - 60)]
        recent_panel = panel.loc[panel["_date"] >= recent_cutoff]
        rank_ref: dict[str, np.ndarray] = {}
        for col in CS_RANK_SOURCE_COLS:
            if col in recent_panel.columns:
                vals = recent_panel[col].dropna().to_numpy(dtype=float)
                if len(vals) > 0:
                    rank_ref[col] = np.sort(vals)

        metrics = CrossSectionalMetrics(
            market=market,
            horizon=self.config.horizon,
            n_symbols=len(symbols_used),
            n_samples=len(panel),
            n_selected_features=len(selected_features),
            trained_at=datetime.now().isoformat(timespec="seconds"),
            symbols_used=symbols_used,
            selected_features=selected_features,
            mean_daily_rank_ic=mean_ic,
            ic_ir=ic_ir,
            excess_dir_accuracy=excess_dir_acc,
            top_bottom_spread_pct=spread_pct,
            top_k=top_k,
            portfolio_total_return=port_stats["port_total"],
            market_total_return=port_stats["mkt_total"],
            excess_total_return=port_stats["port_total"] - port_stats["mkt_total"],
            portfolio_sharpe=port_stats["port_sharpe"],
            market_sharpe=port_stats["mkt_sharpe"],
            portfolio_mdd=port_stats["port_mdd"],
            market_mdd=port_stats["mkt_mdd"],
            equity_curve=port_stats["curve"],
        )

        artifact = CrossSectionalArtifact(
            market=market,
            horizon=self.config.horizon,
            models=final_models,
            model_weights=weights,
            selected_features=selected_features,
            rank_reference=rank_ref,
            metrics=metrics,
            top_feature_importance=top_imp,
        )
        if save:
            save_cross_sectional_artifact(artifact)
        return artifact

    def _select_top_features(
        self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series, top_k: int
    ) -> list[str]:
        """LightGBM 중요도 + 단변량 Spearman |IC| 결합으로 고신호 피처 자동 선별."""
        feats = list(X.columns)
        if len(feats) <= top_k:
            return feats
        lgb_sel = LightGBMStockModel(random_state=self.config.random_seed)
        lgb_sel.fit(X, y_ret, y_dir)
        lgb_imp = lgb_sel.get_feature_importance(feats)

        y_arr = y_ret.to_numpy(dtype=float)
        scores: list[tuple[str, float]] = []
        for f in feats:
            x_arr = X[f].to_numpy(dtype=float)
            ic_abs = 0.0
            if np.std(x_arr) > 1e-12:
                val = spearmanr(x_arr, y_arr).statistic
                if not np.isnan(val):
                    ic_abs = abs(float(val))
            combined = 0.65 * lgb_imp.get(f, 0.0) + 0.35 * ic_abs
            scores.append((f, combined))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [f for f, _ in scores[:top_k]]

    @staticmethod
    def _compute_daily_ics(df: pd.DataFrame, pred_col: str, target_col: str) -> list[float]:
        ics: list[float] = []
        for _, grp in df.groupby("_date"):
            if len(grp) < 4:
                continue
            p = grp[pred_col].to_numpy(dtype=float)
            y = grp[target_col].to_numpy(dtype=float)
            if np.std(p) < 1e-12 or np.std(y) < 1e-12:
                continue
            r = spearmanr(p, y).statistic
            if not np.isnan(r):
                ics.append(float(r))
        return ics

    def _simulate_topk_portfolio(
        self, oof_panel: pd.DataFrame, top_k: int, market: str
    ) -> dict[str, Any]:
        """예측 상위 Top-K 종목 균등투자(H일 중첩 트렌치 + 거래비용) vs 시장 전체 동일가중 비교."""
        costs = MARKET_COST_BPS.get(market, MARKET_COST_BPS["US"])
        buy_c = costs["commission"] + costs["slippage"]
        sell_c = costs["commission"] + costs["sell_tax"] + costs["slippage"]

        pred_piv = oof_panel.pivot(index="_date", columns="_symbol", values="ens_pred").sort_index()
        ret_piv = oof_panel.pivot(index="_date", columns="_symbol", values="_daily_next_ret").sort_index().fillna(0.0)

        # 매일 상위 top_k 종목에 1/top_k 비중 배정
        raw_w = pd.DataFrame(0.0, index=pred_piv.index, columns=pred_piv.columns)
        for dt, row in pred_piv.iterrows():
            valid = row.dropna()
            if len(valid) == 0:
                continue
            k = min(top_k, len(valid))
            winners = valid.nlargest(k).index
            raw_w.loc[dt, winners] = 1.0 / k

        # H일 보유 중첩 트렌치로 회전율 완화 및 정확한 일별 비중 산출
        h = max(self.config.horizon, 1)
        held_w = raw_w.rolling(window=h, min_periods=1).mean()
        w_diff = held_w.diff().fillna(held_w)
        daily_cost = (w_diff.clip(lower=0.0) * buy_c + (-w_diff).clip(lower=0.0) * sell_c).sum(axis=1)

        port_daily = (held_w * ret_piv).sum(axis=1) - daily_cost
        mkt_daily = ret_piv.mean(axis=1)

        port_eq = (1.0 + port_daily).cumprod()
        mkt_eq = (1.0 + mkt_daily).cumprod()
        port_dd = port_eq / port_eq.cummax() - 1.0
        mkt_dd = mkt_eq / mkt_eq.cummax() - 1.0

        p_std = float(port_daily.std())
        m_std = float(mkt_daily.std())
        p_sharpe = float((port_daily.mean() / p_std) * np.sqrt(252)) if p_std > 1e-9 else 0.0
        m_sharpe = float((mkt_daily.mean() / m_std) * np.sqrt(252)) if m_std > 1e-9 else 0.0

        curve = pd.DataFrame(
            {
                "portfolio_equity": port_eq,
                "market_equity": mkt_eq,
                "portfolio_dd": port_dd,
                "market_dd": mkt_dd,
            },
            index=pred_piv.index,
        )
        return {
            "port_total": float(port_eq.iloc[-1] - 1.0),
            "mkt_total": float(mkt_eq.iloc[-1] - 1.0),
            "port_sharpe": p_sharpe,
            "mkt_sharpe": m_sharpe,
            "port_mdd": float(port_dd.min()),
            "mkt_mdd": float(mkt_dd.min()),
            "curve": curve,
        }
