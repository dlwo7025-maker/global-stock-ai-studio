"""모의투자(Paper Trading) DB · 라이브 매수/매도 · 워크포워드 시뮬레이터 오프라인 통합 테스트."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (ROOT_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tests.test_v2_upgrade import _make_panel_bundles
from stock_predictor.config import PredictorConfig
from stock_predictor.trading.paper_engine import (
    HistoricalPaperSimulator,
    LivePaperTrader,
    PaperTradingDB,
)


def test_live_paper_trading_and_rationale() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperTradingDB(db_path=Path(tmpdir) / "paper.db")
        trader = LivePaperTrader(db=db, fx_usdkrw=1400.0)

        # 국내 + 해외 자동 매수 테스트
        screen_df = pd.DataFrame(
            [
                {
                    "시장": "해외(US)", "종목명": "Micron", "티커": "MU", "통화": "USD",
                    "현재가": 100.0, "5일_예상가": 108.0, "하한_10%": 92.0,
                    "상승확률(%)": 72.0, "위원회_점수": 34.0, "최종의견": "매수",
                    "권장비중(%)": 35.0, "재무등급": "A+ (최우량)", "이슈감성": 0.5,
                },
                {
                    "시장": "국내(KR)", "종목명": "HD현대중공업", "티커": "329180", "통화": "KRW",
                    "현재가": 200000.0, "5일_예상가": 210000.0, "하한_10%": 185000.0,
                    "상승확률(%)": 61.0, "위원회_점수": 20.0, "최종의견": "매수",
                    "권장비중(%)": 25.0, "재무등급": "B+ (양호)", "이슈감성": 0.6,
                },
            ]
        )
        orders = trader.execute_ai_portfolio_from_screening(screen_df, fx_usdkrw=1400.0)
        assert len(orders) == 2

        # 가격 변동 후 평가손익 & 매도 검증
        trader.refresh_prices({"MU": 110.0, "329180": 210000.0}, fx_usdkrw=1400.0)
        summary = trader.get_portfolio_summary(fx_usdkrw=1400.0)
        assert summary["n_positions"] == 2
        assert summary["unrealized_pnl_krw"] > 0
        assert all(len(str(r)) > 10 for r in summary["positions_df"]["투자근거"])

        sell_res = trader.sell("US", "MU", price_local=110.0, sell_ratio=1.0, fx_usdkrw=1400.0, reason="목표가 도달 익절")
        assert sell_res["realized_pnl_krw"] > 0
        assert trader.get_portfolio_summary()["n_positions"] == 1


def test_historical_paper_simulator() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperTradingDB(db_path=Path(tmpdir) / "sim.db")
        bundles = _make_panel_bundles("US", n_stocks=4, n_days=360)
        sim = HistoricalPaperSimulator(
            config=PredictorConfig(horizon=5),
            initial_capital_krw=100_000_000.0,
            max_positions=3,
            rebalance_every=5,
            retrain_every=30,
        )
        rep = sim.run(bundles, sim_days=80, db=db)
        assert rep.n_days == 80
        assert rep.final_equity_krw > 0
        assert not rep.equity_curve.empty
        assert rep.closed_trades >= 1
        assert "entry_rationale" in rep.trade_log.columns
        loaded = db.load_latest_simulation()
        assert loaded is not None and loaded.run_id == rep.run_id


if __name__ == "__main__":
    test_live_paper_trading_and_rationale()
    test_historical_paper_simulator()
    print("ALL PAPER TRADING TESTS PASSED!")
