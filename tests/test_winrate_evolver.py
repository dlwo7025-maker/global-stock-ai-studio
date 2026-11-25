"""AI 승률 자가진화(Self-Evolving) 모의투자 시뮬레이션 및 2차 승률 판별기 단위 테스트."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stock_predictor.config import PredictorConfig
from stock_predictor.trading.paper_engine import PaperTradingDB
from stock_predictor.trading.winrate_evolver import (
    WinRateEvolutionEngine,
    load_evolution_history_df,
    predict_stock_win_confidence,
)
from tests.test_v2_upgrade import _make_panel_bundles


def test_winrate_evolution_end_to_end() -> None:
    bundles = _make_panel_bundles(market="US", n_stocks=6, n_days=460)
    with tempfile.TemporaryDirectory() as tmpdir:
        db = PaperTradingDB(db_path=Path(tmpdir) / "evo_test.db")
        cfg = PredictorConfig(horizon=5, use_deep_learning=False, record_predictions=False)
        engine = WinRateEvolutionEngine(config=cfg, db=db)
        champ = engine.run_evolution_rounds(bundles=bundles, n_generations=4)

        hist = load_evolution_history_df(db)
        assert len(hist) == 4
        assert int(hist["is_champion"].sum()) == 1
        assert champ.champion_win_rate_pct >= 0.0
        assert len(champ.learned_rules_ko) >= 1

        ds = engine.feature_engine.build(bundles[0])
        conf_pct, passed, note = predict_stock_win_confidence(
            ds.latest_X, pred_log_ret=0.02, pred_up_prob=0.62, artifact=champ
        )
        assert 10.0 <= conf_pct <= 95.0
        assert isinstance(passed, bool)
        assert len(note) > 3


if __name__ == "__main__":
    test_winrate_evolution_end_to_end()
    print("ALL WINRATE EVOLUTION TESTS PASSED!")
