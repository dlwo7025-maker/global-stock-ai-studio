"""Qlib Alpha & TA-Lib 스타일 퀀트 피처 엔지니어링 모듈.

- 다중 기간 모멘텀, 캔들 구조, 추세(EMA/SMA/MACD/ADX), 오실레이터(RSI/Stoch/CCI/BB)
- 변동성(ATR, Garman-Klass, 변동성 레짐), 거래량·유동성(OBV, VWAP, Amihud)
- 거시경제(S&P500, 나스닥, VIX, 금리, 환율) 및 벤치마크 상대강도/베타
- 한국 시장(KR) 종목에 대한 미국 거시지표 1거래일 시차(Lag) 적용으로 Look-Ahead Bias 완벽 방지
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market_data import MarketDataBundle

US_MACRO_KEYS = {"sp500", "nasdaq", "vix", "us10y", "dxy"}


@dataclass
class FeatureDataset:
    """모델 학습·검증 및 최신 시점 예측을 위해 정렬된 피처 데이터셋."""

    features_all: pd.DataFrame       # 전체 기간 피처 (최신 거래일 포함)
    feature_names: list[str]
    train_X: pd.DataFrame            # 타겟이 존재하는 학습/검증용 피처
    train_y_ret: pd.Series           # 향후 H거래일 로그수익률 타겟
    train_y_dir: pd.Series           # 향후 H거래일 상승 여부 (0/1)
    train_close: pd.Series           # train_X 시점의 종가
    train_future_close: pd.Series    # train_X + H 시점의 실제 종가
    latest_X: pd.DataFrame           # 가장 최근 거래일(실전 미래 예측 입력, 1행)
    latest_date: pd.Timestamp
    latest_close: float
    latest_atr_pct: float
    latest_rv20: float
    horizon: int


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, eps: float = 1e-9) -> pd.Series:
    return a / (b + eps)


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=close.index).ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100.0 * pd.Series(minus_dm, index=close.index).ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean() / (atr + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return adx, plus_di - minus_di, atr


class FeatureEngine:
    """국내·해외 공통 퀀트 피처 생성 엔진."""

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon

    def build(self, bundle: MarketDataBundle) -> FeatureDataset:
        df = bundle.ohlcv.copy()
        idx = df.index
        o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

        feat = pd.DataFrame(index=idx)

        # 1. 다중 기간 로그수익률 & 모멘텀
        log_c = np.log(c.clip(lower=1e-6))
        daily_ret = log_c.diff()
        for w in [1, 2, 3, 5, 10, 20, 60, 120]:
            feat[f"ret_{w}d"] = log_c - log_c.shift(w)

        # 모멘텀 가속도 (단기 vs 중기)
        feat["mom_accel_5_20"] = feat["ret_5d"] - (feat["ret_20d"] / 4.0)
        feat["mom_accel_20_60"] = feat["ret_20d"] - (feat["ret_60d"] / 3.0)

        # 2. 캔들 가격 행동 (OHLC Microstructure)
        hl_range = (h - l).clip(lower=1e-9)
        feat["hl_range_pct"] = (h - l) / (c + 1e-9)
        feat["gap_ret"] = np.log((o / (c.shift(1) + 1e-9)).clip(lower=1e-6))
        feat["candle_body_ratio"] = (c - o) / hl_range
        feat["upper_shadow_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / hl_range
        feat["lower_shadow_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / hl_range
        feat["close_pos_in_day"] = (c - l) / hl_range

        # 3. 이동평균 & 추세 (Trend)
        smas = {w: c.rolling(w, min_periods=w).mean() for w in [5, 10, 20, 60, 120]}
        for w, sma in smas.items():
            feat[f"dist_sma_{w}"] = _safe_div(c - sma, sma)
        feat["sma_5_20_cross"] = _safe_div(smas[5] - smas[20], smas[20])
        feat["sma_20_60_cross"] = _safe_div(smas[20] - smas[60], smas[60])
        feat["sma_60_120_cross"] = _safe_div(smas[60] - smas[120], smas[120])

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        feat["macd_pct"] = _safe_div(macd, c)
        feat["macd_hist_pct"] = _safe_div(macd - macd_sig, c)

        adx, di_diff, atr = _compute_adx(h, l, c, window=14)
        feat["adx_14"] = adx / 100.0
        feat["di_diff_14"] = di_diff / 100.0
        feat["atr_14_pct"] = _safe_div(atr, c)

        # 4. 오실레이터 & 평균회귀 (Mean Reversion)
        feat["rsi_7"] = _compute_rsi(c, 7) / 100.0
        feat["rsi_14"] = _compute_rsi(c, 14) / 100.0

        low14 = l.rolling(14, min_periods=14).min()
        high14 = h.rolling(14, min_periods=14).max()
        stoch_k = _safe_div(c - low14, high14 - low14)
        feat["stoch_k_14"] = stoch_k
        feat["stoch_d_14"] = stoch_k.rolling(3).mean()

        tp = (h + l + c) / 3.0
        tp_sma = tp.rolling(20, min_periods=20).mean()
        tp_mad = (tp - tp_sma).abs().rolling(20, min_periods=20).mean()
        feat["cci_20"] = ((tp - tp_sma) / (0.015 * tp_mad + 1e-9)).clip(-400, 400) / 100.0

        bb_std = c.rolling(20, min_periods=20).std()
        feat["bb_zscore_20"] = _safe_div(c - smas[20], bb_std).clip(-5, 5)
        feat["bb_width_20"] = _safe_div(4.0 * bb_std, smas[20])

        # 5. 변동성 & 낙폭(Drawdown)
        rv5 = daily_ret.rolling(5, min_periods=5).std() * np.sqrt(252)
        rv20 = daily_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
        rv60 = daily_ret.rolling(60, min_periods=60).std() * np.sqrt(252)
        feat["rv_5d"] = rv5
        feat["rv_20d"] = rv20
        feat["rv_60d"] = rv60
        feat["vol_regime_5_60"] = _safe_div(rv5, rv60).clip(0, 5)

        # Garman-Klass 변동성 (OHLC 정보 활용)
        log_hl = np.log((h / (l + 1e-9)).clip(lower=1e-6))
        log_co = np.log((c / (o + 1e-9)).clip(lower=1e-6))
        gk_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        feat["gk_vol_20"] = np.sqrt(gk_var.rolling(20, min_periods=20).mean().clip(lower=0.0) * 252)

        # 60일 / 252일 고점 대비 낙폭 및 저점 대비 위치
        roll_max_60 = c.rolling(60, min_periods=20).max()
        roll_min_60 = c.rolling(60, min_periods=20).min()
        roll_max_252 = c.rolling(252, min_periods=60).max()
        roll_min_252 = c.rolling(252, min_periods=60).min()
        feat["drawdown_60d"] = _safe_div(c - roll_max_60, roll_max_60)
        feat["pos_in_60d_range"] = _safe_div(c - roll_min_60, roll_max_60 - roll_min_60)
        feat["drawdown_252d"] = _safe_div(c - roll_max_252, roll_max_252)
        feat["pos_in_252d_range"] = _safe_div(c - roll_min_252, roll_max_252 - roll_min_252)

        # 6. 거래량 & 유동성 팩터
        vol_sma5 = v.rolling(5, min_periods=5).mean()
        vol_sma20 = v.rolling(20, min_periods=20).mean()
        vol_std20 = v.rolling(20, min_periods=20).std()
        feat["vol_ratio_5_20"] = _safe_div(vol_sma5, vol_sma20).clip(0, 10)
        feat["vol_zscore_20"] = _safe_div(v - vol_sma20, vol_std20).clip(-5, 5)

        obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
        obv_change = obv - obv.shift(10)
        feat["obv_norm_10"] = _safe_div(obv_change, vol_sma20 * 10.0).clip(-5, 5)

        dollar_vol = (c * v).replace(0, np.nan)
        vwap_20 = (tp * v).rolling(20, min_periods=10).sum() / (v.rolling(20, min_periods=10).sum() + 1e-9)
        feat["dist_vwap_20"] = _safe_div(c - vwap_20, vwap_20).clip(-0.5, 0.5)
        feat["amihud_illiq_20"] = (daily_ret.abs() / (np.log1p(dollar_vol.fillna(0.0)) + 1.0)).rolling(20, min_periods=10).mean()

        # 수익률 왜도/첨도 (꼬리 리스크)
        feat["ret_skew_20"] = daily_ret.rolling(20, min_periods=20).skew().clip(-5, 5)

        # 7. 거시경제 & 벤치마크 상대강도 (Look-ahead Bias 엄격 차단!)
        if bundle.macro is not None and not bundle.macro.empty:
            macro_aligned = self._align_macro_without_leakage(bundle.macro, idx, bundle.symbol.market)
            bm_key = bundle.symbol.benchmark_key
            if bm_key in macro_aligned.columns:
                bm = macro_aligned[bm_key]
                bm_log = np.log(bm.clip(lower=1e-6))
                bm_ret1 = bm_log.diff()
                bm_ret5 = bm_log - bm_log.shift(5)
                bm_ret20 = bm_log - bm_log.shift(20)
                feat["bm_ret_1d"] = bm_ret1
                feat["bm_ret_5d"] = bm_ret5
                feat["bm_ret_20d"] = bm_ret20
                feat["rel_alpha_5d"] = feat["ret_5d"] - bm_ret5
                feat["rel_alpha_20d"] = feat["ret_20d"] - bm_ret20
                cov60 = daily_ret.rolling(60, min_periods=30).cov(bm_ret1)
                var60 = bm_ret1.rolling(60, min_periods=30).var()
                feat["beta_60d"] = _safe_div(cov60, var60).clip(-3, 4)

            if "nasdaq" in macro_aligned.columns:
                nq_log = np.log(macro_aligned["nasdaq"].clip(lower=1e-6))
                feat["nasdaq_ret_1d"] = nq_log.diff()
                feat["nasdaq_ret_5d"] = nq_log - nq_log.shift(5)

            if "vix" in macro_aligned.columns:
                vix = macro_aligned["vix"]
                feat["vix_level"] = (vix / 100.0).clip(0, 1.5)
                feat["vix_ret_5d"] = np.log((vix / (vix.shift(5) + 1e-9)).clip(lower=1e-6))

            if "us10y" in macro_aligned.columns:
                us10y = macro_aligned["us10y"]
                feat["us10y_diff_20d"] = (us10y - us10y.shift(20)) / 100.0

            if "usdkrw" in macro_aligned.columns:
                fx = macro_aligned["usdkrw"]
                fx_log = np.log(fx.clip(lower=1e-6))
                feat["usdkrw_ret_5d"] = fx_log - fx_log.shift(5)
                feat["usdkrw_ret_20d"] = fx_log - fx_log.shift(20)

        # 8. 국내 주식 외국인·기관 수급 팩터 (가능한 경우)
        if bundle.investor_flow is not None and not bundle.investor_flow.empty:
            flow = bundle.investor_flow.reindex(idx).fillna(0.0)
            dv20 = (c * v).rolling(20, min_periods=5).mean().replace(0, np.nan)
            if "foreign_net" in flow.columns:
                feat["kr_foreign_flow_5d"] = _safe_div(flow["foreign_net"].rolling(5).sum(), dv20 * 5.0).clip(-2, 2)
                feat["kr_foreign_flow_20d"] = _safe_div(flow["foreign_net"].rolling(20).sum(), dv20 * 20.0).clip(-2, 2)
            if "inst_net" in flow.columns:
                feat["kr_inst_flow_5d"] = _safe_div(flow["inst_net"].rolling(5).sum(), dv20 * 5.0).clip(-2, 2)
                feat["kr_inst_flow_20d"] = _safe_div(flow["inst_net"].rolling(20).sum(), dv20 * 20.0).clip(-2, 2)

        # 9. 캘린더 순환 피처
        feat["dow_sin"] = np.sin(2.0 * np.pi * idx.dayofweek / 5.0)
        feat["dow_cos"] = np.cos(2.0 * np.pi * idx.dayofweek / 5.0)
        feat["month_sin"] = np.sin(2.0 * np.pi * idx.month / 12.0)
        feat["month_cos"] = np.cos(2.0 * np.pi * idx.month / 12.0)

        # 무한대/결측 정리
        feat = feat.replace([np.inf, -np.inf], np.nan)

        # 타겟 생성: 향후 H거래일 로그수익률
        future_close = c.shift(-self.horizon)
        target_ret = np.log((future_close / (c + 1e-9)).clip(lower=1e-6))
        target_dir = (target_ret > 0.0).astype(int)

        # 초기 워밍업 구간(피처 결측이 많은 앞부분) 제거
        # 컬럼 중 대부분이 NaN인 피처는 드롭
        valid_cols = [col for col in feat.columns if feat[col].notna().mean() > 0.7]
        feat = feat[valid_cols].copy()
        feat = feat.ffill()

        # 여전히 NaN이 남는 초기 행 제거
        valid_mask = feat.notna().all(axis=1)
        feat_clean = feat.loc[valid_mask].copy()

        if len(feat_clean) < 100:
            # 드물게 일부 피처 때문에 너무 많이 잘리면 중앙값으로 보완
            feat_clean = feat.dropna(thresh=int(len(valid_cols) * 0.8)).copy()
            feat_clean = feat_clean.fillna(feat_clean.median(numeric_only=True)).fillna(0.0)

        # 최신 시점(실전 예측용)
        latest_date = feat_clean.index[-1]
        latest_X = feat_clean.iloc[[-1]].copy()
        latest_close = float(c.loc[latest_date])
        latest_atr_pct = float(feat_clean.loc[latest_date, "atr_14_pct"]) if "atr_14_pct" in feat_clean.columns else 0.02
        latest_rv20 = float(feat_clean.loc[latest_date, "rv_20d"]) if "rv_20d" in feat_clean.columns else 0.25

        # 학습/검증 가능 구간 (미래 타겟이 존재하는 구간)
        trainable_idx = feat_clean.index.intersection(target_ret.dropna().index)
        train_X = feat_clean.loc[trainable_idx].copy()
        train_y_ret = target_ret.loc[trainable_idx].copy()
        train_y_dir = target_dir.loc[trainable_idx].copy()
        train_close = c.loc[trainable_idx].copy()
        train_future_close = future_close.loc[trainable_idx].copy()

        return FeatureDataset(
            features_all=feat_clean,
            feature_names=list(feat_clean.columns),
            train_X=train_X,
            train_y_ret=train_y_ret,
            train_y_dir=train_y_dir,
            train_close=train_close,
            train_future_close=train_future_close,
            latest_X=latest_X,
            latest_date=latest_date,
            latest_close=latest_close,
            latest_atr_pct=max(latest_atr_pct, 0.005),
            latest_rv20=max(latest_rv20, 0.08),
            horizon=self.horizon,
        )

    @staticmethod
    def _align_macro_without_leakage(
        macro: pd.DataFrame, target_index: pd.DatetimeIndex, market: str
    ) -> pd.DataFrame:
        """시차를 고려해 거시지표를 종목 거래일에 안전하게 정렬.

        한국(KR) 장 마감(KST 15:30)은 미국 장 개장(KST 23:30)보다 먼저 발생하므로,
        한국 날짜 t에 미국 날짜 t 종가를 붙이면 미래 정보 누수가 발생한다.
        따라서 KR 시장에서는 미국 지표를 반드시 1거래일 지연(shift(1))한다.
        """
        aligned_cols: dict[str, pd.Series] = {}
        for col in macro.columns:
            s = macro[col].dropna().sort_index()
            if s.empty:
                continue
            if market == "KR" and col in US_MACRO_KEYS:
                # 1거래일 이전 미국 마감 데이터만 참조
                s_shifted = s.shift(1).dropna()
                aligned = s_shifted.reindex(target_index, method="ffill")
            else:
                aligned = s.reindex(target_index, method="ffill")
            aligned_cols[col] = aligned
        return pd.DataFrame(aligned_cols, index=target_index)
