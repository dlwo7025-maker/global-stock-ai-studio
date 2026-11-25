"""폴리마켓·예측시장·큰손 대안 데이터 해석 및 AI 모델/포트폴리오 반영 단위 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stock_predictor.config import PredictorConfig
from stock_predictor.data.alt_intelligence import (
    StockAltSignal,
    _interpret_bet_for_beginners,
)
from stock_predictor.pipeline import GlobalStockPipeline
from tests.test_v2_upgrade import _make_panel_bundles


def test_beginner_bet_interpretation() -> None:
    topic_r, exp_r, imp_r, sc_r = _interpret_bet_for_beginners(
        "US Recession in 2026?", "Will the US enter a recession?", 62.0
    )
    assert "경기침체" in topic_r
    assert imp_r == "악재"
    assert sc_r < -0.2

    topic_c, exp_c, imp_c, sc_c = _interpret_bet_for_beginners(
        "Fed interest rate decision", "25 bps rate cut in December?", 74.0
    )
    assert "금리" in topic_c
    assert imp_c == "호재"
    assert sc_c > 0.2


def test_alt_signal_influences_forecast_and_committee() -> None:
    bundles = _make_panel_bundles(market="US", n_stocks=1, n_days=320)
    base_bundle = bundles[0]

    cfg = PredictorConfig(
        horizon=5,
        n_splits=3,
        use_deep_learning=False,
        use_cross_sectional=False,
        use_memory_calibration=False,
        record_predictions=False,
    )
    pipe = GlobalStockPipeline(config=cfg)

    # 1) 강한 긍정 폴리마켓·큰손 시그널 (+0.85)
    base_bundle.alt_signal = StockAltSignal(
        combined_signal=0.85,
        easy_bullets_ko=["👔 임원 자사주 매수 포착", "🔮 폴리마켓 금리인하 호재 75%"],
    )
    pos_rep = pipe.analyze_bundle(base_bundle)

    # 2) 강한 부정 폴리마켓·큰손 시그널 (-0.85)
    base_bundle.alt_signal = StockAltSignal(
        combined_signal=-0.85,
        easy_bullets_ko=["🔮 폴리마켓 경기침체 우려 상승", "🏦 큰손 순매도"],
    )
    neg_rep = pipe.analyze_bundle(base_bundle)

    assert pos_rep.forecast.expected_pct_return > neg_rep.forecast.expected_pct_return
    assert pos_rep.forecast.up_probability > neg_rep.forecast.up_probability
    assert pos_rep.verdict.composite_score > neg_rep.verdict.composite_score
    assert "예측시장_큰손민심_보정(%)" in pos_rep.forecast.signal_breakdown
    assert pos_rep.forecast.signal_breakdown["예측시장_큰손민심_보정(%)"] > 0
    assert neg_rep.forecast.signal_breakdown["예측시장_큰손민심_보정(%)"] < 0


if __name__ == "__main__":
    test_beginner_bet_interpretation()
    test_alt_signal_influences_forecast_and_committee()
    print("ALL ALT INTELLIGENCE TESTS PASSED!")
