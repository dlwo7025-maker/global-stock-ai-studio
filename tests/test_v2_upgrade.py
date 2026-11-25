"""v2 업그레이드 종합 오프라인 검증 테스트.

검증 항목:
1. 기업 재정·실적 5대 기둥 채점(우량주 vs 부실주 점수·등급·시그널 분별력)
2. 실시간 이슈·뉴스 6대 이벤트 분류, 시간감쇠, 호재/악재 쇼크 스코어링
3. 다중 종목 통합(Cross-Sectional) 알파 모델 학습, 자동 피처 선별, Top-K 포트폴리오 OOF 검증 및 단일 종목 추론
4. SQLite 예측 기록 -> 만기 도래 자동 채점 -> 모델별 실전 적중률 기반 자가보정(Online Calibration)
5. 전체 하이브리드 융합(단일 ML/DL + 통합 알파 + 기업 재무 + 실시간 이슈) 기여도 분해 검증
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import PredictorConfig
from stock_predictor.data.fundamentals import FundamentalAnalyzer
from stock_predictor.data.market_data import MarketDataBundle, ResolvedSymbol
from stock_predictor.data.news_intelligence import NewsIntelligenceEngine
from stock_predictor.models.cross_sectional import (
    CrossSectionalAlphaTrainer,
    predict_single_with_artifact,
)
from stock_predictor.pipeline import GlobalStockPipeline
from stock_predictor.storage.memory_db import PredictionMemoryDB


def _make_panel_bundles(market: str = "US", n_stocks: int = 6, n_days: int = 380) -> list[MarketDataBundle]:
    rng = np.random.default_rng(123 if market == "US" else 456)
    dates = pd.bdate_range("2024-06-01", periods=n_days)
    mkt_shock = rng.normal(0.0004, 0.010, size=n_days)
    macro = pd.DataFrame(
        {
            "sp500": 5000.0 * np.exp(np.cumsum(mkt_shock)),
            "nasdaq": 16000.0 * np.exp(np.cumsum(mkt_shock * 1.15)),
            "kospi": 2600.0 * np.exp(np.cumsum(mkt_shock * 0.85)),
            "vix": np.clip(18.0 + np.cumsum(rng.normal(0, 0.5, n_days)), 12.0, 40.0),
            "us10y": np.clip(4.1 + np.cumsum(rng.normal(0, 0.02, n_days)), 2.8, 5.5),
            "usdkrw": np.clip(1360.0 + np.cumsum(rng.normal(0, 3.0, n_days)), 1200.0, 1500.0),
        },
        index=dates,
    )

    bundles: list[MarketDataBundle] = []
    for i in range(n_stocks):
        # 각 종목에 고유 모멘텀/평균회귀 신호를 약간 섞어 학습 가능성 검증
        idio = rng.normal(0.0001 * (i - n_stocks / 2), 0.013, size=n_days)
        ret = 0.6 * mkt_shock + idio
        close = (50000.0 if market == "KR" else 100.0) * (1.0 + 0.1 * i) * np.exp(np.cumsum(ret))
        high = close * (1.0 + np.abs(rng.normal(0.007, 0.005, n_days)))
        low = close * (1.0 - np.abs(rng.normal(0.007, 0.005, n_days)))
        open_p = np.roll(close, 1)
        open_p[0] = close[0]
        vol = rng.integers(300_000, 4_000_000, size=n_days).astype(float)

        code = f"00593{i}" if market == "KR" else f"SYM{i}"
        sym = ResolvedSymbol(
            raw_input=code,
            market=market,
            code=code,
            yf_ticker=f"{code}.KS" if market == "KR" else code,
            name=f"테스트종목{i}",
            currency="KRW" if market == "KR" else "USD",
            benchmark_key="kospi" if market == "KR" else "sp500",
        )
        bundles.append(
            MarketDataBundle(
                symbol=sym,
                ohlcv=pd.DataFrame({"Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates),
                macro=macro,
                data_source="synthetic_panel",
            )
        )
    return bundles


def test_fundamental_scoring() -> None:
    fa = FundamentalAnalyzer()
    strong = fa.score_from_metrics(
        {
            "forward_pe": 11.5,
            "pbr": 1.1,
            "roe": 0.24,
            "operating_margin": 0.28,
            "revenue_growth": 0.30,
            "earnings_growth": 0.35,
            "debt_to_equity": 32.0,
            "current_ratio": 2.4,
            "fcf_yield": 0.06,
            "avg_eps_surprise": 0.12,
            "target_upside": 0.25,
        }
    )
    weak = fa.score_from_metrics(
        {
            "forward_pe": 65.0,
            "pbr": 9.5,
            "roe": -0.08,
            "operating_margin": -0.05,
            "revenue_growth": -0.12,
            "earnings_growth": -0.30,
            "debt_to_equity": 260.0,
            "current_ratio": 0.7,
            "fcf_yield": -0.04,
            "avg_eps_surprise": -0.10,
            "target_upside": -0.15,
        }
    )
    assert strong.overall_score > 80.0 and strong.fundamental_signal > 0.5
    assert weak.overall_score < 32.0 and weak.fundamental_signal < -0.4
    assert len(strong.strengths) >= 3 and len(weak.red_flags) >= 3


def test_news_event_intelligence() -> None:
    engine = NewsIntelligenceEngine(use_llm=False)
    bull_news = engine.analyze_articles(
        [
            {"title": "삼성전자 어닝 서프라이즈 및 HBM 대규모 수주 계약 체결", "publisher": "한국경제", "published": "2026-11-23T09:00:00Z"},
            {"title": "증권가 목표가 상향 잇따라... 사상 최대 실적 기대감", "publisher": "매일경제", "published": "2026-11-23T11:00:00Z"},
        ]
    )
    bear_news = engine.analyze_articles(
        [
            {"title": "대규모 유상증자 결정 및 어닝 쇼크 우려에 급락", "publisher": "이데일리", "published": "2026-11-23T09:00:00Z"},
            {"title": "SEC probe and antitrust lawsuit hit outlook after earnings miss", "publisher": "Reuters", "published": "2026-11-23T10:00:00Z"},
        ]
    )
    assert bull_news.sentiment_score > 0.4 and bull_news.event_shock_score > 0.4
    assert bear_news.sentiment_score < -0.4 and bear_news.event_shock_score < -0.4
    assert len(bull_news.key_catalysts) >= 1 and len(bear_news.key_risks) >= 1


def test_cross_sectional_and_self_learning_loop() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PredictionMemoryDB(db_path=Path(tmpdir) / "test_mem.db")
        cfg = PredictorConfig(
            horizon=5,
            n_splits=2,
            min_train_rows=140,
            top_k_features=20,
            use_deep_learning=True,
            use_cross_sectional=True,
            use_fundamental_overlay=True,
            use_news_overlay=True,
            use_cache=False,
        )

        # 1) Cross-Sectional 6종목 패널 통합 학습 & 저장 없이 아티팩트 검증
        bundles = _make_panel_bundles(market="KR", n_stocks=6, n_days=340)
        trainer = CrossSectionalAlphaTrainer(config=cfg)
        artifact = trainer.train_from_bundles(bundles, market="KR", save=False)
        assert artifact.metrics.n_symbols == 6
        assert artifact.metrics.n_selected_features <= 20
        assert not artifact.metrics.equity_curve.empty

        # 2) 심층 재무 + 뉴스를 부착한 종목을 통합 파이프라인으로 분석 (Cross-Sectional 아티팩트 주입)
        target_b = bundles[0]
        target_b.fundamental_profile = FundamentalAnalyzer().score_from_metrics(
            {"forward_pe": 12.0, "roe": 0.22, "revenue_growth": 0.25, "debt_to_equity": 40.0, "avg_eps_surprise": 0.09}
        )
        target_b.news_report = NewsIntelligenceEngine().analyze_articles(
            [{"title": "어닝 서프라이즈 및 대규모 수주 발표, 목표가 상향", "publisher": "Test", "published": "2026-11-23T08:00:00Z"}]
        )

        pipeline = GlobalStockPipeline(config=cfg, memory_db=db)
        report = pipeline.analyze_bundle(target_b, cs_artifact=artifact)
        assert report.forecast.cross_sectional_used is True
        assert report.forecast.signal_breakdown["기업_재정실적_보정(%)"] > 0
        assert report.forecast.signal_breakdown["실시간_이슈뉴스_보정(%)"] > 0

        # 3) SQLite 예측 기록 -> 만기 채점 -> 자가보정 배수 확인
        # 채점 테스트를 위해 6건의 과거 만기 예측 기록 후 자동 채점
        for d in range(6):
            db.log_prediction(
                market="KR",
                code="005930",
                name="테스트",
                horizon=5,
                as_of_date=pd.Timestamp(f"2026-10-0{d+1}"),
                target_date=pd.Timestamp(f"2026-10-1{d+1}"),
                base_price=100.0,
                pred_price=104.0,
                pred_pct_ret=0.04,
                up_prob=0.65,
                fund_score=75.0,
                news_score=0.4,
                committee_score=40.0,
                action="매수",
                model_entries=[("lightgbm", 0.04, 0.70), ("xgboost", -0.02, 0.35)],
            )
        scored = db.evaluate_due_predictions(
            price_resolver=lambda m, c, dt: 105.0,
            as_of_today="2026-11-24",
        )
        assert scored >= 6
        mults = db.get_model_calibration_multipliers("KR", min_samples=5)
        assert mults["lightgbm"] > 1.0  # 100% 적중 -> 가중치 상승
        assert mults["xgboost"] < 1.0   # 0% 적중 -> 가중치 하락


if __name__ == "__main__":
    test_fundamental_scoring()
    test_news_event_intelligence()
    test_cross_sectional_and_self_learning_loop()
    print("ALL V2 UPGRADE TESTS PASSED!")
