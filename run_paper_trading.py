"""국내·해외 AI 모의투자 실행기 (1년 워크포워드 시뮬레이션 + 1억 원 라이브 모의 계좌 집행).

사용 예시:
    python run_paper_trading.py --simulate --open-live
    python run_paper_trading.py --simulate --days 240
    python run_paper_trading.py --open-live --reset-live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import OUTPUT_DIR, PredictorConfig
from stock_predictor.data.market_data import MarketDataLoader
from stock_predictor.trading.live_monitor import fetch_realtime_fx_usdkrw
from stock_predictor.trading.paper_engine import (
    DEFAULT_INITIAL_CASH_KRW,
    HistoricalPaperSimulator,
    LivePaperTrader,
    PaperTradingDB,
)

load_dotenv(ROOT_DIR / ".env")
console = Console()

DEFAULT_PAPER_SYMBOLS = [
    "삼성전자", "SK하이닉스", "한화에어로스페이스", "HD현대중공업",
    "두산에너빌리티", "HD현대일렉트릭", "알테오젠",
    "NVDA", "MU", "AAPL", "PLTR", "OKLO", "IONQ", "LLY",
]


def _get_latest_fx(loader: MarketDataLoader) -> float:
    return fetch_realtime_fx_usdkrw(fallback=1380.0)


def _load_latest_screening_csv() -> pd.DataFrame:
    if not OUTPUT_DIR.exists():
        return pd.DataFrame()
    csvs = sorted(OUTPUT_DIR.glob("prediction_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return pd.DataFrame()
    df = pd.read_csv(csvs[0], encoding="utf-8-sig")
    if "위원회_점수" in df.columns:
        df = df.sort_values("위원회_점수", ascending=False).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="국내·해외 AI 모의투자 시뮬레이션 및 1억 원 라이브 계좌 집행")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_PAPER_SYMBOLS, help="시뮬레이션 대상 종목 리스트")
    parser.add_argument("--simulate", action="store_true", help="과거 워크포워드 멀티종목 AI 모의투자 시뮬레이션 실행")
    parser.add_argument("--days", type=int, default=240, help="시뮬레이션 거래일 수 (기본: 240일 ≈ 1년)")
    parser.add_argument("--open-live", action="store_true", help="최신 AI 스크리닝 기반 1억 원 라이브 모의계좌 첫 매수 집행")
    parser.add_argument("--reset-live", action="store_true", help="라이브 모의 계좌를 초기 1억 원으로 초기화 후 집행")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CASH_KRW, help="초기 모의투자 자본금 (원, 기본 1억)")
    args = parser.parse_args()

    if not args.simulate and not args.open_live:
        args.simulate = True
        args.open_live = True

    db = PaperTradingDB()
    loader = MarketDataLoader(use_cache=True)

    # 1) 과거 1년 Strict Walk-Forward 멀티종목 AI 모의투자 시뮬레이션 + 학습 데이터셋 생성
    if args.simulate:
        console.print(f"\n[bold cyan]>>> 국내·해외 {len(args.symbols)}종목 워크포워드 AI 모의투자 시뮬레이션 시작 (초기자본 {args.capital:,.0f}원, 기간 {args.days}거래일)...[/bold cyan]")
        bundles = []
        for idx, sym in enumerate(args.symbols, start=1):
            console.print(f"[dim]({idx}/{len(args.symbols)}) {sym} 가격·거시 데이터 로딩...[/dim]")
            try:
                bundles.append(loader.load_bundle(sym, lookback_years=5, include_deep_info=False))
            except Exception as exc:
                console.print(f"[yellow]스킵 ({sym}): {exc}[/yellow]")

        sim = HistoricalPaperSimulator(
            config=PredictorConfig(horizon=5),
            initial_capital_krw=args.capital,
            max_positions=5,
            rebalance_every=10,
            retrain_every=20,
            stop_loss_pct=-0.08,
            take_profit_pct=0.22,
        )
        rep = sim.run(bundles, sim_days=args.days, db=db, progress_cb=lambda msg: console.print(f"[dim]{msg}[/dim]"))

        panel_body = (
            f"기간: [bold]{rep.start_date} ~ {rep.end_date}[/bold] ({rep.n_days}거래일 · {rep.n_symbols}종목 통합)\n"
            f"초기 자본: {rep.initial_capital_krw:,.0f}원  →  최종 평가액: [bold cyan]{rep.final_equity_krw:,.0f}원[/bold cyan]\n"
            f"AI 모의투자 수익률: [bold]{rep.total_return_pct:+.2f}%[/bold] (CAGR {rep.cagr_pct:+.2f}%, Sharpe {rep.sharpe_ratio:.2f}, MDD {rep.max_drawdown_pct:.2f}%)\n"
            f"벤치마크(동일가중): {rep.benchmark_return_pct:+.2f}% (Sharpe {rep.benchmark_sharpe:.2f}, MDD {rep.benchmark_mdd_pct:.2f}%)  |  초과수익: {rep.excess_return_pct:+.2f}%p\n"
            f"청산 거래: {rep.closed_trades}건  |  승률: [bold]{rep.win_rate_pct:.1f}%[/bold]  |  손익비(PF): {rep.profit_factor:.2f}  |  총 수수료·세금: {rep.total_fees_krw:,.0f}원\n"
            f"생성된 자가학습 데이터셋: [green]{rep.learning_dataset_path}[/green]"
        )
        console.print(Panel(panel_body, title="AI 멀티종목 워크포워드 모의투자 결과", border_style="cyan", box=box.ROUNDED))

        if not rep.symbol_pnl.empty:
            tbl = Table(title="종목별 모의투자 누적 손익 기여도", box=box.SIMPLE_HEAVY)
            for c in rep.symbol_pnl.columns:
                tbl.add_column(c, justify="right" if c not in {"market", "code", "name"} else "left")
            for _, r in rep.symbol_pnl.iterrows():
                tbl.add_row(
                    str(r["market"]), str(r["code"]), str(r["name"]),
                    str(int(r["거래횟수"])), f"{r['승률']:.1f}%",
                    f"{r['누적손익_원']:+,.0f}원", f"{r['평균수익률']:+.2f}%", f"{r['평균보유일']:.1f}일",
                )
            console.print(tbl)

    # 2) 오늘 기준 1억 원 라이브 모의투자 계좌 첫 매수 집행
    if args.open_live:
        fx_now = _get_latest_fx(loader)
        if args.reset_live:
            db.reset_account(initial_cash_krw=args.capital, fx_usdkrw=fx_now)
        trader = LivePaperTrader(db=db, fx_usdkrw=fx_now)
        screen_df = _load_latest_screening_csv()
        if screen_df.empty:
            console.print("[yellow]저장된 최신 AI 스크리닝 CSV가 없어 라이브 매수를 건너뜁니다.[/yellow]")
            return

        console.print(f"\n[bold green]>>> 1억 원 라이브 모의투자 계좌 AI 포트폴리오 집행 (환율 1 USD = {fx_now:,.1f}원)...[/bold green]")
        orders = trader.execute_ai_portfolio_from_screening(screen_df, fx_usdkrw=fx_now)
        summary = trader.get_portfolio_summary(fx_usdkrw=fx_now)

        console.print(f"체결 완료: [bold]{len(orders)}건[/bold] | 총평가액: {summary['total_equity_krw']:,.0f}원 | 주식투자금: {summary['invested_krw']:,.0f}원 | 보유현금: {summary['cash_krw']:,.0f}원")
        pos_df = summary["positions_df"]
        if not pos_df.empty:
            pt = Table(title="라이브 모의투자 보유 포트폴리오 & 매수 근거", box=box.ROUNDED)
            for col in ["시장", "종목명", "수량", "매입가", "평가금액(원)", "비중", "재무", "AI 매수 근거"]:
                pt.add_column(col)
            for _, p in pos_df.iterrows():
                px_s = f"{p['매입단가']:,.0f}원" if p["통화"] == "KRW" else f"${p['매입단가']:,.2f}"
                pt.add_row(
                    str(p["시장"]), str(p["종목명"]), str(p["보유수량"]), px_s,
                    f"{p['평가금액(원)']:,.0f}원", f"{p['비중(%)']:.1f}%", str(p["재무등급"]),
                    str(p["투자근거"]),
                )
            console.print(pt)


if __name__ == "__main__":
    main()
