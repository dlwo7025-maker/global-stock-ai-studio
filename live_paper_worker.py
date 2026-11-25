"""실시간 라이브 모의투자 백그라운드 감시 데몬.

지정된 주기(기본 30초)마다 보유 중인 국내/해외 종목의 실시간 체결가를 수집하고
직전 대비 상승(▲)/하락(▼), 매입가 대비 실시간 손익, 총자산 변동을 SQLite 및 CSV에 지속 기록합니다.

사용 예시:
    python live_paper_worker.py --interval 30
    python live_paper_worker.py --interval 30 --auto-guard   # 목표가 자동익절 / 손절선 자동손절 활성화
    python live_paper_worker.py --once                        # 1회만 즉시 체크
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.trading.live_monitor import poll_once_and_record
from stock_predictor.trading.paper_engine import LivePaperTrader, PaperTradingDB

load_dotenv(ROOT_DIR / ".env")


def _print_tick_report(res: dict) -> None:
    s = res["summary"]
    print(
        f"\n[{res['recorded_at']}] 총자산: {s['total_equity_krw']:,.0f}원 "
        f"({s['total_return_pct']:+.3f}% | 평가손익 {s['unrealized_pnl_krw']:+,.0f}원 | "
        f"환율 {s['fx_usdkrw']:,.1f}원)",
        flush=True,
    )
    for t in res["ticks"]:
        px = f"{t['price_local']:,.0f}원" if t["currency"] == "KRW" else f"${t['price_local']:,.2f}"
        print(
            f"  {t['direction']} {t['name']}({t['code']}): {px} "
            f"[틱 {t['tick_change_pct']:+.2f}% | 매입대비 {t['pnl_pct']:+.2f}% ({t['pnl_krw']:+,.0f}원)] "
            f"- {t['status_note']}",
            flush=True,
        )
    for ao in res["auto_orders"]:
        print(
            f"  >>> [자동 체결] {ao['side']} {ao['name']} {ao['shares']}주 "
            f"(실현손익 {ao['realized_pnl_krw']:+,.0f}원 / {ao['realized_pnl_pct']:+.2f}%)",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 국내·해외 모의투자 가격 추적 데몬")
    parser.add_argument("--interval", type=int, default=30, help="실시간 체크 주기 (초, 기본 30초)")
    parser.add_argument("--auto-guard", action="store_true", help="목표가 도달 자동 익절 / 손절선 이탈 자동 손절 활성화")
    parser.add_argument("--once", action="store_true", help="1회만 실행 후 종료")
    args = parser.parse_args()

    db = PaperTradingDB()
    trader = LivePaperTrader(db=db)

    print(
        f"=== 🔴 LIVE 실시간 모의투자 워커 시작 (주기 {args.interval}초 | 자동손절/익절: {'ON' if args.auto_guard else 'OFF'}) ===",
        flush=True,
    )

    while True:
        try:
            res = poll_once_and_record(db=db, trader=trader, auto_guard=args.auto_guard)
            _print_tick_report(res)
        except Exception as exc:
            print(f"[오류] 실시간 체크 실패: {exc}", flush=True)

        if args.once:
            break
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    main()
