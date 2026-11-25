"""컴퓨터가 며칠~10일 동안 꺼져 있었을 때 빈 기간의 시세·손절·익절·만기채점·승률훈련을 한 번에 복구하는 스크립트.

작동 순서:
1. 마지막 실시간 체크 시각 이후 공백 기간(예: 10일)의 1시간봉(1h) 실제 주가 흐름을 시간순으로 불러옵니다.
2. 컴퓨터가 꺼져 있던 동안 보유 종목이 목표 수익가(익절)나 안전 브레이크(손절)에 닿았는지 시간순으로 정확히 재현해 모의지갑에 반영합니다.
3. 꺼져 있던 10일 동안 만기일이 지난 모든 과거 AI 예측(prediction_memory.db)을 실제 주가로 자동 채점해 AI 가중치를 보정합니다.
4. 새로 쌓인 10일 치 시장 데이터까지 포함해 승률 자가진화 훈련과 '10만 원 실전 바로 시작 패키지'를 최신화합니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import KNOWN_KOSDAQ_CODES, PredictorConfig
from stock_predictor.pipeline import GlobalStockPipeline
from stock_predictor.trading.live_monitor import fetch_realtime_fx_usdkrw, poll_once_and_record
from stock_predictor.trading.paper_engine import LivePaperTrader, PaperTradingDB
from evolve_winrate_worker import run_one_cycle

load_dotenv(ROOT_DIR / ".env")


def replay_offline_gap(db: PaperTradingDB, trader: LivePaperTrader) -> int:
    """컴퓨터 종료 공백 기간의 1시간봉을 조회해 놓친 손절·익절 이벤트를 시간순 복구."""
    ticks = db.get_live_account_ticks_df(limit=5)
    pos_df = db.get_positions_df()
    if pos_df.empty or ticks.empty:
        return 0

    last_ts = pd.Timestamp(ticks["recorded_at"].iloc[-1]).tz_localize(None)
    gap_hours = (datetime.now() - last_ts.to_pydatetime()).total_seconds() / 3600.0
    if gap_hours < 4.0:
        return 0

    print(f"[공백 복구] 약 {gap_hours:.1f}시간 동안 컴퓨터가 꺼져 있었습니다. 놓친 주가 흐름을 복구합니다...", flush=True)
    fx_now = fetch_realtime_fx_usdkrw()
    executed_cnt = 0

    try:
        import yfinance as yf
    except Exception:
        return 0

    days_back = min(max(int(gap_hours // 24) + 3, 5), 30)
    for _, p in pos_df.iterrows():
        mkt = str(p["market"])
        code = str(p["code"])
        name = str(p["name"])
        stop_px = float(p["stop_loss_local"]) if p["stop_loss_local"] else None
        tgt_px = float(p["target_price_local"]) if p["target_price_local"] else None
        avg_local = float(p["avg_cost_local"])
        yf_sym = f"{code}{'.KQ' if code in KNOWN_KOSDAQ_CODES else '.KS'}" if mkt == "KR" else code

        try:
            hist = yf.Ticker(yf_sym).history(period=f"{days_back}d", interval="1h")
            if hist is None or hist.empty:
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            sub = hist[hist.index > last_ts]
            if sub.empty:
                continue

            for ts, bar in sub.iterrows():
                hi = float(bar["High"])
                lo = float(bar["Low"])
                if avg_local > 0 and ((hi / avg_local) - 1.0) >= 0.018:
                    lock_cand = avg_local * 1.004
                    stop_px = max(stop_px or 0.0, lock_cand)

                if stop_px and lo <= stop_px:
                    is_profit = avg_local > 0 and stop_px > avg_local
                    tag = "공백기간 수익보존 익절" if is_profit else "공백기간 안전브레이크 손절"
                    res = trader.sell(mkt, code, stop_px, 1.0, fx_usdkrw=fx_now, reason=f"{tag} ({ts:%m-%d %H:%M})")
                    print(f"  -> [복구 체결] {name}: {tag} ({res['realized_pnl_krw']:+,.0f}원)", flush=True)
                    executed_cnt += 1
                    break
                if tgt_px and hi >= tgt_px:
                    res = trader.sell(mkt, code, tgt_px, 1.0, fx_usdkrw=fx_now, reason=f"공백기간 목표가 자동익절 ({ts:%m-%d %H:%M})")
                    print(f"  -> [복구 체결] {name}: 목표가 자동익절 ({res['realized_pnl_krw']:+,.0f}원)", flush=True)
                    executed_cnt += 1
                    break
        except Exception:
            continue

    return executed_cnt


def main() -> None:
    db = PaperTradingDB()
    trader = LivePaperTrader(db=db)

    # 1) 꺼져 있던 기간의 손절/익절 재현 + 현재 시세 반영
    replay_offline_gap(db, trader)
    poll_once_and_record(db=db, trader=trader, auto_guard=True)

    # 2) 꺼져 있던 10일 동안 만기 도래한 과거 예측 전수 채점!
    pipe = GlobalStockPipeline(config=PredictorConfig(horizon=5, use_cache=True))
    newly, sc = pipe.evaluate_past_predictions()
    print(f"[만기 채점] 꺼져 있던 동안 결과가 나온 예측 {newly}건 채점 완료! (누적 채점 {sc.evaluated_count}건)", flush=True)

    # 3) 최신 10일 주가 반영해 승률 훈련 10세대 + 추천표 + 10만 원 패키지 일괄 갱신
    run_one_cycle(generations=10, refresh_screening=True)
    print("✅ 꺼져 있던 기간의 모든 데이터 복구 및 10만 원 실전 패키지 최신화 완료!", flush=True)


if __name__ == "__main__":
    main()
