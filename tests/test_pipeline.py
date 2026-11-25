"""오프라인 합성 데이터 기반 통합 파이프라인 & Look-Ahead Bias 방지 검증 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import PredictorConfig
from stock_predictor.data.features import FeatureEngine
from stock_predictor.data.market_data import MarketDataBundle, ResolvedSymbol, resolve_symbol
from stock_predictor.pipeline import GlobalStockPipeline


def _make_synthetic_bundle(market: str = "KR", n_days: int = 520, seed: int = 42) -> MarketDataBundle:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)

    # 추세 + 평균회귀 + 노이즈가 섞인 현실적인 주가 시계열 생성
    macro_shock = rng.normal(0.0004, 0.011, size=n_days)
    stock_noise = rng.normal(0.0002, 0.014, size=n_days)
    log_ret = 0.6 * macro_shock + stock_noise
    close = (70000.0 if market == "KR" else 150.0) * np.exp(np.cumsum(log_ret))
    high = close * (1.0 + np.abs(rng.normal(0.008, 0.006, size=n_days)))
    low = close * (1.0 - np.abs(rng.normal(0.008, 0.006, size=n_days)))
    open_p = np.roll(close, 1)
    open_p[0] = close[0]
    volume = rng.integers(500_000, 5_000_000, size=n_days).astype(float)

    ohlcv = pd.DataFrame(
        {"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    macro = pd.DataFrame(
        {
            "sp500": 5000.0 * np.exp(np.cumsum(macro_shock)),
            "nasdaq": 16000.0 * np.exp(np.cumsum(macro_shock * 1.2)),
            "kospi": 2600.0 * np.exp(np.cumsum(macro_shock * 0.8 + rng.normal(0, 0.005, n_days))),
            "vix": np.clip(18.0 + np.cumsum(rng.normal(0, 0.6, n_days)), 11.0, 45.0),
            "us10y": np.clip(4.2 + np.cumsum(rng.normal(0, 0.03, n_days)), 2.5, 6.0),
            "usdkrw": np.clip(1350.0 + np.cumsum(rng.normal(0, 3.5, n_days)), 1150.0, 1550.0),
        },
        index=dates,
    )
    flow = None
    if market == "KR":
        flow = pd.DataFrame(
            {
                "foreign_net": rng.normal(0, 3e10, size=n_days),
                "inst_net": rng.normal(0, 2e10, size=n_days),
            },
            index=dates,
        )

    sym = ResolvedSymbol(
        raw_input="005930" if market == "KR" else "NVDA",
        market=market,
        code="005930" if market == "KR" else "NVDA",
        yf_ticker="005930.KS" if market == "KR" else "NVDA",
        name="삼성전자(합성)" if market == "KR" else "NVIDIA(합성)",
        currency="KRW" if market == "KR" else "USD",
        benchmark_key="kospi" if market == "KR" else "sp500",
    )
    return MarketDataBundle(
        symbol=sym,
        ohlcv=ohlcv,
        macro=macro,
        investor_flow=flow,
        fundamentals={"trailingPE": 14.5, "priceToBook": 1.3, "returnOnEquity": 0.16, "revenueGrowth": 0.18},
        news=[{"title": "신규 AI 반도체 수주 확대 및 실적 성장 기대감", "publisher": "TestNews", "published": "2026-01-01"}],
        data_source="synthetic",
    )


def test_symbol_resolution() -> None:
    s_kr = resolve_symbol("삼성전자")
    assert s_kr.market == "KR" and s_kr.code == "005930" and s_kr.currency == "KRW"
    s_kq = resolve_symbol("247540")
    assert s_kq.market == "KR" and s_kq.yf_ticker == "247540.KQ"
    s_us = resolve_symbol("nvda")
    assert s_us.market == "US" and s_us.code == "NVDA" and s_us.currency == "USD"


def test_kr_macro_no_lookahead_bias() -> None:
    """KR 종목에서 미국 거시지표가 정확히 1거래일 지연(Lag)되어 미래 누수가 없는지 검증."""
    dates = pd.bdate_range("2025-01-02", periods=5)
    macro = pd.DataFrame(
        {
            "sp500": [100.0, 101.0, 102.0, 103.0, 999.0],  # 마지막 날 급등
            "kospi": [2500.0, 2510.0, 2520.0, 2530.0, 2540.0],
        },
        index=dates,
    )
    aligned_kr = FeatureEngine._align_macro_without_leakage(macro, dates, market="KR")
    # KR 마지막 날(dates[-1])에는 미국 전일(103.0)만 보여야 하며 당일 999.0이 보이면 안 됨!
    assert aligned_kr.loc[dates[-1], "sp500"] == 103.0
    assert aligned_kr.loc[dates[-1], "kospi"] == 2540.0

    aligned_us = FeatureEngine._align_macro_without_leakage(macro, dates, market="US")
    assert aligned_us.loc[dates[-1], "sp500"] == 999.0


def test_full_pipeline_kr_and_us() -> None:
    cfg = PredictorConfig(
        horizon=5,
        n_splits=3,
        min_train_rows=200,
        use_deep_learning=True,
        use_cross_sectional=False,
        record_predictions=False,
        use_cache=False,
    )
    pipeline = GlobalStockPipeline(config=cfg)

    for mkt in ("KR", "US"):
        bundle = _make_synthetic_bundle(market=mkt)
        report = pipeline.analyze_bundle(bundle)
        assert report.forecast.current_price > 0
        assert report.forecast.target_price_low < report.forecast.target_price_median < report.forecast.target_price_high
        assert 0.0 < report.forecast.up_probability < 1.0
        assert len(report.forecast.model_metrics) == 5
        assert abs(sum(m.weight for m in report.forecast.model_metrics) - 1.0) < 1e-6
        assert len(report.verdict.opinions) == 5
        assert report.backtest.n_days >= 50
        assert -1.0 <= report.backtest.strategy_mdd <= 0.0
        summary = report.to_summary_dict()
        assert "최종의견" in summary and "OOF_방향적중률(%)" in summary


if __name__ == "__main__":
    test_symbol_resolution()
    test_kr_macro_no_lookahead_bias()
    test_full_pipeline_kr_and_us()
    print("ALL OFFLINE TESTS PASSED!")
