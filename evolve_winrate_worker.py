"""AI 승률 자가진화(Self-Evolving) 백그라운드 시뮬레이션 훈련 워커.

지속적으로 국내·해외 다중 종목 과거·최신 데이터 위에서 모의투자 시뮬레이션을 반복하며,
수수료·세금 차감 후 실제 익절 승률이 더 높은 매매 규칙과 2차 승률 판별기(Meta-AI)를 찾아
`saved_models/winrate_champion.pkl` 및 SQLite `evolution_generations` 테이블을 갱신합니다.

사용 예시:
    python evolve_winrate_worker.py --once --generations 7 --refresh-screening
    python evolve_winrate_worker.py --interval 300 --generations 3
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import OUTPUT_DIR, PredictorConfig
from stock_predictor.data.market_data import MarketDataLoader
from stock_predictor.pipeline import GlobalStockPipeline
from stock_predictor.trading.paper_engine import PaperTradingDB
from stock_predictor.trading.starter_pack import export_starter_pack_files
from stock_predictor.trading.winrate_evolver import WinRateEvolutionEngine

load_dotenv(ROOT_DIR / ".env")

EVOLUTION_UNIVERSE: list[str] = [
    "삼성전자",
    "SK하이닉스",
    "KB금융",
    "삼성바이오로직스",
    "현대차",
    "한화에어로스페이스",
    "HD현대중공업",
    "두산에너빌리티",
    "NVDA",
    "AAPL",
    "MSFT",
    "META",
    "COST",
    "V",
    "LLY",
    "OKLO",
]

CORE_SCREEN_SYMBOLS: list[str] = [
    "삼성전자",
    "SK하이닉스",
    "KB금융",
    "삼성바이오로직스",
    "한화에어로스페이스",
    "HD현대중공업",
    "NVDA",
    "AAPL",
    "MSFT",
    "META",
    "COST",
    "V",
    "LLY",
    "OKLO",
]


def run_one_cycle(generations: int = 6, refresh_screening: bool = False) -> None:
    cfg = PredictorConfig(horizon=5, use_cache=True)
    loader = MarketDataLoader(use_cache=True)
    db = PaperTradingDB()
    evolver = WinRateEvolutionEngine(config=cfg, db=db)

    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] 🧬 AI 승률 자가진화 훈련 사이클 시작 (목표 {generations}세대)...", flush=True)
    bundles = []
    for sym in EVOLUTION_UNIVERSE:
        try:
            bundles.append(loader.load_bundle(sym, lookback_years=5, include_deep_info=False))
        except Exception as exc:
            print(f"  [스킵] {sym}: {exc}", flush=True)

    champ = evolver.run_evolution_rounds(
        bundles=bundles,
        n_generations=generations,
        progress_cb=lambda msg: print(f"  {msg}", flush=True),
    )
    print(
        f"=== 🏆 챔피언 결과 (제{champ.generation}세대) ===\n"
        f"  • 초보 시절(1세대) 승률: {champ.baseline_win_rate_pct:.1f}%\n"
        f"  • 현재 챔피언 검증 승률: {champ.champion_win_rate_pct:.1f}% ({champ.champion_win_rate_pct - champ.baseline_win_rate_pct:+.1f}%p 상승!)\n"
        f"  • 검증 수익률 / 손익비: {champ.champion_return_pct:+.2f}% / PF {champ.champion_profit_factor:.2f} (누적 연습 매매 {champ.total_simulated_trades}회)",
        flush=True,
    )
    for r in champ.learned_rules_ko:
        print(f"    - {r}", flush=True)

    # 만기 도래 예측 자동 채점도 함께 수행 (온라인 가중치 자가보정)
    pipe = GlobalStockPipeline(config=cfg)
    try:
        newly, sc = pipe.evaluate_past_predictions()
        if newly > 0:
            print(f"  [자가채점] 만기 도래 예측 {newly}건 신규 채점 완료 (누적 {sc.evaluated_count}건)", flush=True)
    except Exception:
        pass

    if refresh_screening:
        print("  [추천 갱신] 새 챔피언 승률 필터를 적용해 주요 종목 추천표를 업데이트합니다...", flush=True)
        df, _, errs = pipe.screen(CORE_SCREEN_SYMBOLS)
        if not df.empty:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"prediction_evolved_{datetime.now():%Y%m%d_%H%M%S}.csv"
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            # 오래된 중복 추천 CSV 정리 (최근 5개만 유지해 저장소 용량 경량화)
            old_csvs = sorted(OUTPUT_DIR.glob("prediction_evolved_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old_f in old_csvs[5:]:
                try:
                    old_f.unlink()
                except Exception:
                    pass
            print(f"  [완료] {out_path.name} 저장 ({len(df)}종목, 오류 {len(errs)}건)", flush=True)

    # 매 사이클마다 '10만 원 실전 바로 시작 패키지(.md / .csv / .zip)' 자동 최신화 저장!
    pack = export_starter_pack_files(budget_krw=100_000.0, db=db)
    st_info = pack["stats"]
    print(
        f"  [📦 실전 패키지 갱신] 누적 {st_info['total_generations']}세대({st_info['total_simulated_trades']:,}회 매매) · "
        f"실시간틱 {st_info['live_position_ticks']:,}건 -> data/starter_pack/ai_accumulated_data_pack.zip 최신화 완료!",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 승률 자가진화 모의투자 시뮬레이션 워커")
    parser.add_argument("--generations", type=int, default=5, help="한 사이클당 진화 세대 수")
    parser.add_argument("--interval", type=int, default=300, help="연속 실행 시 대기 주기(초)")
    parser.add_argument("--once", action="store_true", help="1사이클만 즉시 실행 후 종료")
    parser.add_argument("--refresh-screening", action="store_true", help="훈련 직후 종목 추천 CSV 갱신")
    args = parser.parse_args()

    cycle_idx = 0
    while True:
        try:
            cycle_idx += 1
            should_refresh = (args.refresh_screening and cycle_idx == 1) or (not args.once and cycle_idx % 10 == 0)
            run_one_cycle(
                generations=max(args.generations, 1),
                refresh_screening=should_refresh,
            )
        except Exception as exc:
            print(f"[진화 워커 오류] {exc}", flush=True)

        if args.once:
            break
        time.sleep(max(args.interval, 30))


if __name__ == "__main__":
    main()
