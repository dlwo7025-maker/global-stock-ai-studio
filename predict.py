"""국내·해외 주식 통합 AI 예측 · 다중종목 알파 학습 · 심층 재무/이슈 분석 · 자가채점 CLI 도구.

사용 예시:
    # 1) 국내(KR) + 해외(US) 20종목 통합 Cross-Sectional 알파 모델 사전 학습
    python predict.py --train-universe KR US --horizon 5

    # 2) 단일 / 복수 종목 심층 예측 (차트 + 통합알파 + 회사 재무 + 실시간 이슈)
    python predict.py 삼성전자 NVDA --horizon 5 --save-json

    # 3) 만기 도래한 과거 예측 자동 채점 및 자가학습 성적표 확인
    python predict.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
from stock_predictor.models.cross_sectional import CrossSectionalArtifact
from stock_predictor.pipeline import GlobalStockPipeline, StockAnalysisReport

load_dotenv(ROOT_DIR / ".env")
console = Console()


def _fmt_price(price: float, currency: str) -> str:
    return f"{price:,.0f}원" if currency == "KRW" else f"${price:,.2f}"


def _action_style(action: str) -> str:
    if "강력 매수" in action:
        return "bold bright_green"
    if "매수" in action:
        return "bold green"
    if "강력 매도" in action:
        return "bold bright_red"
    if "매도" in action:
        return "bold red"
    return "bold yellow"


def print_universe_artifact(artifact: CrossSectionalArtifact) -> None:
    m = artifact.metrics
    mkt_label = "국내(KRX)" if m.market == "KR" else "해외(US)"
    body = (
        f"[bold cyan]{mkt_label} 다중종목 통합(Cross-Sectional) 알파 모델 학습 완료[/bold cyan]\n"
        f"학습 규모: {m.n_symbols}개 종목 · {m.n_samples:,}행 패널 · 자동 선별 피처 {m.n_selected_features}개 · 예측 기간 {m.horizon}거래일\n"
        f"일별 Cross-Sectional Rank IC: [bold]{m.mean_daily_rank_ic:+.4f}[/bold] (ICIR {m.ic_ir:+.2f})  |  "
        f"시장 초과수익 방향 적중률: [bold]{m.excess_dir_accuracy * 100:.1f}%[/bold]  |  "
        f"상위20% vs 하위20% 스프레드: [bold]{m.top_bottom_spread_pct:+.2f}%[/bold]"
    )
    console.print(Panel(body, title="Cross-Sectional 통합 알파 모델 벤치마크", border_style="green", box=box.ROUNDED))

    tbl = Table(title=f"OOF Top-{m.top_k} 알파 포트폴리오 vs 시장(유니버스) 평균 (거래비용·세금 반영)", box=box.SIMPLE_HEAVY)
    tbl.add_column("성과 지표", style="bold")
    tbl.add_column(f"AI Top-{m.top_k} 알파 전략", justify="right", style="cyan")
    tbl.add_column("시장(유니버스) 평균", justify="right")
    tbl.add_row("누적 수익률", f"{m.portfolio_total_return * 100:+.2f}%", f"{m.market_total_return * 100:+.2f}%")
    tbl.add_row("시장 대비 초과수익", f"{m.excess_total_return * 100:+.2f}%", "-")
    tbl.add_row("샤프 비율 (Sharpe)", f"{m.portfolio_sharpe:.2f}", f"{m.market_sharpe:.2f}")
    tbl.add_row("최대 낙폭 (MDD)", f"{m.portfolio_mdd * 100:.2f}%", f"{m.market_mdd * 100:.2f}%")
    console.print(tbl)


def print_single_report(report: StockAnalysisReport) -> None:
    sym = report.bundle.symbol
    fc = report.forecast
    bt = report.backtest
    vd = report.verdict
    fp = report.bundle.fundamental_profile
    nr = report.bundle.news_report

    mkt_label = "국내(KRX)" if sym.market == "KR" else "해외(US/Global)"
    tags = []
    if fc.cross_sectional_used:
        tags.append("통합알파 ON")
    if fp and fp.data_completeness > 0:
        tags.append(f"재무 {fp.grade}")
    if nr and nr.article_count > 0:
        tags.append(f"이슈 {nr.article_count}건({nr.sentiment_score:+.2f})")
    if fc.memory_calibrated:
        tags.append("자가보정 ON")

    header = (
        f"[bold cyan]{sym.name} ({sym.code})[/bold cyan]  |  {mkt_label}  |  "
        f"기준일: {report.dataset.latest_date.strftime('%Y-%m-%d')}  |  {' · '.join(tags)}\n"
        f"현재가: [bold]{_fmt_price(fc.current_price, sym.currency)}[/bold]  →  "
        f"{fc.horizon}거래일 예상가: [bold]{_fmt_price(fc.target_price_median, sym.currency)}[/bold] "
        f"([bold]{fc.expected_pct_return * 100:+.2f}%[/bold])  |  "
        f"80% 구간: {_fmt_price(fc.target_price_low, sym.currency)} ~ {_fmt_price(fc.target_price_high, sym.currency)}\n"
        f"상승확률: [bold]{fc.up_probability * 100:.1f}%[/bold]  |  "
        f"모델 합의: {fc.model_agreement * 100:.0f}%  |  신뢰도: {fc.confidence_score:.1f}/100  |  "
        f"최종 의견: [{_action_style(vd.action)}]{vd.action}[/{_action_style(vd.action)}] "
        f"(종합 {vd.composite_score:+.1f}점, 권장비중 {vd.recommended_position_pct:.0f}%, 위험도: {vd.risk_level})"
    )
    console.print(Panel(header, title="AI 주식 통합 예측 리포트 (차트 + 통합알파 + 재무 + 이슈)", border_style="cyan", box=box.ROUNDED))

    # 1. 예측 기여도 분해표
    if fc.signal_breakdown:
        bd = fc.signal_breakdown
        bd_table = Table(title="예측 기대수익률 구성 요소 분해 (Attribution)", box=box.SIMPLE)
        for k in bd.keys():
            bd_table.add_column(k.replace("_", " "), justify="right")
        bd_table.add_row(*[f"{float(v):+.2f}%" for v in bd.values()])
        console.print(bd_table)

    # 2. 모델별 앙상블 성과 & 예측 표
    m_table = Table(title="참여 모델별 검증 성과 & 예측", box=box.SIMPLE_HEAVY)
    m_table.add_column("모델명", style="bold")
    m_table.add_column("비중", justify="right")
    m_table.add_column("OOF적중", justify="right")
    m_table.add_column("RankIC", justify="right")
    m_table.add_column(f"{fc.horizon}일예측", justify="right")
    m_table.add_column("상승확률", justify="right")
    m_table.add_column("목표가", justify="right")
    for m in fc.model_metrics:
        m_table.add_row(
            m.display_name,
            f"{m.weight * 100:.1f}%",
            f"{m.direction_accuracy * 100:.1f}%",
            f"{m.rank_ic:+.3f}",
            f"{m.latest_expected_return * 100:+.2f}%",
            f"{m.latest_up_prob * 100:.1f}%",
            _fmt_price(m.latest_target_price, sym.currency),
        )
    console.print(m_table)

    # 3. 기업 재정·실적 5대 기둥 요약
    if fp and fp.data_completeness > 0:
        f_table = Table(title=f"기업 재정·실적 종합 분석 ({fp.grade}, {fp.overall_score:.1f}/100점, 소스: {fp.source})", box=box.SIMPLE)
        f_table.add_column("밸류에이션", justify="right")
        f_table.add_column("수익성", justify="right")
        f_table.add_column("성장성", justify="right")
        f_table.add_column("재무건전성", justify="right")
        f_table.add_column("실적/컨센서스", justify="right")
        f_table.add_column("재무시그널", justify="right")
        f_table.add_row(
            f"{fp.valuation_score:.0f}",
            f"{fp.profitability_score:.0f}",
            f"{fp.growth_score:.0f}",
            f"{fp.health_score:.0f}",
            f"{fp.earnings_score:.0f}",
            f"{fp.fundamental_signal:+.2f}",
        )
        console.print(f_table)

    # 4. 최신 핵심 이슈 Top 4
    if nr and nr.article_count > 0:
        n_table = Table(
            title=f"실시간 이슈·뉴스 레이더 ({nr.article_count}건 · 감성 {nr.sentiment_score:+.2f} · 이벤트쇼크 {nr.event_shock_score:+.2f} · 주력: {nr.dominant_category})",
            box=box.SIMPLE,
        )
        n_table.add_column("이벤트", style="bold", no_wrap=True)
        n_table.add_column("감성", justify="right", no_wrap=True)
        n_table.add_column("헤드라인")
        n_table.add_column("출처", no_wrap=True)
        for art in nr.items[:5]:
            s_color = "green" if art.sentiment >= 0.15 else ("red" if art.sentiment <= -0.15 else "dim")
            n_table.add_row(
                art.category,
                f"[{s_color}]{art.sentiment:+.2f}[/{s_color}]",
                art.title[:68] + ("…" if len(art.title) > 68 else ""),
                art.publisher[:14] or art.source,
            )
        console.print(n_table)

    # 5. 워크포워드 백테스팅 요약
    bt_table = Table(
        title=f"단일종목 Walk-Forward 백테스트 ({bt.start_date.strftime('%Y-%m-%d')} ~ {bt.end_date.strftime('%Y-%m-%d')})",
        box=box.SIMPLE_HEAVY,
    )
    bt_table.add_column("항목", style="bold")
    bt_table.add_column("AI 전략", justify="right", style="cyan")
    bt_table.add_column("단순 보유 (Buy & Hold)", justify="right")
    bt_table.add_row("누적 수익률 / CAGR", f"{bt.strategy_total_return * 100:+.2f}% / {bt.strategy_cagr * 100:+.2f}%", f"{bt.benchmark_total_return * 100:+.2f}% / {bt.benchmark_cagr * 100:+.2f}%")
    bt_table.add_row("샤프 비율 / MDD", f"{bt.strategy_sharpe:.2f} / {bt.strategy_mdd * 100:.2f}%", f"{bt.benchmark_sharpe:.2f} / {bt.benchmark_mdd * 100:.2f}%")
    bt_table.add_row("OOF 적중률 / 활성승률", f"{bt.direction_accuracy * 100:.1f}% / {bt.win_rate * 100:.1f}%", "-")
    console.print(bt_table)

    # 6. 5인 투자 위원회 & CIO 메모
    ag_table = Table(title="5인 AI 투자 위원회 심사 결과", box=box.SIMPLE_HEAVY)
    ag_table.add_column("에이전트", style="bold")
    ag_table.add_column("스탠스", justify="center")
    ag_table.add_column("점수", justify="right")
    ag_table.add_column("핵심 요약")
    for op in vd.opinions:
        st_color = "green" if op.stance == "강세" else ("red" if op.stance == "약세" else "yellow")
        ag_table.add_row(op.role, f"[{st_color}]{op.stance}[/{st_color}]", f"{op.score:+.0f}", op.summary)
    console.print(ag_table)
    console.print(Panel(vd.cio_memo, title="CIO 최종 종합 메모" + (" (LLM 보강)" if vd.llm_enhanced else ""), border_style="magenta"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="국내(KRX) · 해외(US/Global) 주식 통합 AI 예측 · 다중종목 알파 학습 · 심층 재무/이슈 분석 CLI"
    )
    parser.add_argument("symbols", nargs="*", help="분석할 종목명 또는 티커 (예: 삼성전자 000660 NVDA AAPL)")
    parser.add_argument("--train-universe", nargs="+", choices=["KR", "US", "kr", "us"], help="다중종목 통합 알파 모델 학습 (KR / US)")
    parser.add_argument("--evaluate", action="store_true", help="만기 도래한 과거 예측 자동 채점 및 성적표 출력")
    parser.add_argument("--horizon", type=int, default=5, help="예측 기간 거래일 (기본: 5)")
    parser.add_argument("--years", type=int, default=5, help="학습 과거 데이터 연수 (기본: 5)")
    parser.add_argument("--no-dl", action="store_true", help="시계열 신경망 모델 제외")
    parser.add_argument("--no-cs", action="store_true", help="통합 Cross-Sectional 알파 모델 미사용")
    parser.add_argument("--no-fundamental", action="store_true", help="기업 재무 오버레이 미사용")
    parser.add_argument("--no-news", action="store_true", help="실시간 뉴스/이슈 오버레이 미사용")
    parser.add_argument("--use-chronos", action="store_true", help="Amazon Chronos 파운데이션 모델 결합")
    parser.add_argument("--use-llm", action="store_true", help="OpenAI LLM 뉴스/CIO 심층 분석 활성화")
    parser.add_argument("--no-cache", action="store_true", help="로컬 캐시 무시하고 최신 데이터 재수집")
    parser.add_argument("--save-json", action="store_true", help="결과 요약을 outputs/ JSON 저장")
    parser.add_argument("--save-csv", action="store_true", help="결과 요약을 outputs/ CSV 저장")
    args = parser.parse_args()

    if not args.symbols and not args.train_universe and not args.evaluate:
        parser.print_help()
        return

    config = PredictorConfig(
        horizon=args.horizon,
        lookback_years=args.years,
        use_deep_learning=not args.no_dl,
        use_cross_sectional=not args.no_cs,
        use_fundamental_overlay=not args.no_fundamental,
        use_news_overlay=not args.no_news,
        use_chronos=args.use_chronos,
        use_llm_agents=args.use_llm,
        use_cache=not args.no_cache,
    )
    pipeline = GlobalStockPipeline(config=config)

    # 1) 다중종목 통합 알파 모델 학습 모드
    if args.train_universe:
        for mkt in args.train_universe:
            mkt_u = mkt.upper()
            console.print(f"\n[bold cyan]>>> {mkt_u} 다중종목 통합 Cross-Sectional 알파 학습 시작 (horizon={args.horizon}일)...[/bold cyan]")
            art = pipeline.train_universe(market=mkt_u, progress_cb=lambda msg: console.print(f"[dim]{msg}[/dim]"))
            print_universe_artifact(art)

    # 2) 자가 채점 성적표 모드
    if args.evaluate:
        console.print("\n[bold cyan]>>> 만기 도래 과거 예측 자동 채점 중...[/bold cyan]")
        newly, sc = pipeline.evaluate_past_predictions()
        panel_txt = (
            f"신규 채점 완료: [bold]{newly}건[/bold]\n"
            f"누적 기록: {sc.total_logged}건 (채점 완료 {sc.evaluated_count}건 / 만기 대기 {sc.pending_count}건)\n"
            f"실전 방향 적중률: [bold]{sc.hit_rate * 100:.1f}%[/bold]  |  "
            f"평균 예상수익: {sc.avg_predicted_return * 100:+.2f}%  |  평균 실제수익: {sc.avg_actual_return * 100:+.2f}%"
        )
        console.print(Panel(panel_txt, title="자가 학습 예측 성적표 (SQLite Memory)", border_style="yellow"))

    # 3) 종목 심층 분석 / 스크리닝 모드
    if args.symbols:
        if len(args.symbols) == 1:
            with console.status(f"[bold cyan]{args.symbols[0]} 심층 데이터(주가·거시·재무·이슈) 수집 및 예측 중..."):
                report = pipeline.analyze(args.symbols[0])
            print_single_report(report)
            reports = [report]
        else:
            def _progress(idx: int, total: int, sym: str) -> None:
                console.print(f"[dim]({idx}/{total})[/dim] [cyan]{sym}[/cyan] 심층 수집(재무·이슈) 및 앙상블 예측 중...")

            summary_df, reports, errors = pipeline.screen(args.symbols, progress_cb=_progress)
            for rep in reports:
                print_single_report(rep)

            if summary_df is not None and not summary_df.empty:
                rank_table = Table(title="국내 · 해외 통합 AI 스크리닝 종합 랭킹", box=box.ROUNDED)
                ret_col = f"{args.horizon}일_기대수익률(%)"
                for header, just in [
                    ("시장", "left"), ("종목", "left"), ("현재가", "right"),
                    (f"{args.horizon}일예상", "right"), ("상승확률", "right"),
                    ("재무", "right"), ("이슈", "right"), ("점수", "right"), ("의견", "center"),
                ]:
                    rank_table.add_column(header, justify=just, no_wrap=True)
                for _, row in summary_df.iterrows():
                    mkt_s = "KR" if "KR" in str(row["시장"]) else "US"
                    rank_table.add_row(
                        mkt_s,
                        f"{row['종목명']}({row['티커']})",
                        _fmt_price(float(row["현재가"]), str(row["통화"])),
                        f"{float(row[ret_col]):+.2f}%",
                        f"{float(row['상승확률(%)']):.1f}%",
                        f"{float(row['재무점수']):.0f}",
                        f"{float(row['이슈감성']):+.2f}",
                        f"{float(row['위원회_점수']):+.1f}",
                        str(row["최종의견"]),
                    )
                console.print(rank_table)

            for sym, msg in errors:
                console.print(f"[bold red]오류 ({sym}):[/bold red] {msg}")

        if reports and (args.save_json or args.save_csv):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            summaries = [r.to_summary_dict() for r in reports]
            if args.save_json:
                jpath = OUTPUT_DIR / f"prediction_{ts}.json"
                jpath.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
                console.print(f"[green]JSON 저장 완료:[/green] {jpath}")
            if args.save_csv:
                import pandas as pd

                cpath = OUTPUT_DIR / f"prediction_{ts}.csv"
                pd.DataFrame(summaries).to_csv(cpath, index=False, encoding="utf-8-sig")
                console.print(f"[green]CSV 저장 완료:[/green] {cpath}")

    console.print(
        "\n[dim]※ 본 시스템의 예측 및 백테스팅 결과는 투자 참고·연구 목적이며, 실제 투자 손실에 대한 책임은 투자자 본인에게 있습니다.[/dim]"
    )


if __name__ == "__main__":
    main()
