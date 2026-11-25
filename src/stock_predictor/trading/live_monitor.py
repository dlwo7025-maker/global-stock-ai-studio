"""실시간 시세 수집 및 라이브 모의투자 자동 모니터링·체크 모듈.

- 해외(US): yfinance 1분봉/fast_info 실시간 체결가 + 당일 등락률
- 국내(KR): 네이버 금융 실시간 폴링 API -> yfinance .KS/.KQ 1분봉 폴백
- 매 체크마다:
  1) 보유 종목의 실시간 체결가·직전 대비 오름(▲)/내림(▼)/보합(─)·매입가 대비 실시간 손익 계산
  2) SQLite (live_account_ticks / live_position_ticks) 및 CSV에 실시간 시계열 누적
  3) auto_guard=True 시 목표가 도달 자동 익절 / 손절선 이탈 자동 손절 실행
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import KNOWN_KOSDAQ_CODES
from .paper_engine import PAPER_DATASET_DIR, LivePaperTrader, PaperTradingDB


def fetch_realtime_fx_usdkrw(fallback: float = 1380.0) -> float:
    """실시간 원/달러 환율 조회."""
    try:
        import yfinance as yf

        fi = yf.Ticker("KRW=X").fast_info
        val = fi.get("lastPrice") or fi.get("regularMarketPrice")
        if val and float(val) > 500:
            return float(val)
    except Exception:
        pass
    return fallback


def fetch_realtime_quote(market: str, code: str) -> dict[str, Any] | None:
    """단일 종목의 실시간 체결가, 당일 등락률(%), 시장 상태 조회."""
    if market == "KR":
        # 1) 네이버 금융 국내주식 실시간 API (가장 빠르고 정확)
        try:
            url = f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if resp.ok:
                data = resp.json()
                datas = data.get("datas") or []
                if datas:
                    item = datas[0]
                    raw_p = str(item.get("closePriceRaw") or item.get("closePrice") or "").replace(",", "")
                    raw_ratio = str(item.get("fluctuationsRatio") or "0").replace(",", "")
                    status = str(item.get("marketStatus") or "CLOSE")
                    price = float(raw_p)
                    if price > 0:
                        return {
                            "price": price,
                            "day_change_pct": float(raw_ratio),
                            "market_status": status,
                            "source": "NaverRealtime",
                        }
        except Exception:
            pass

        # 2) 폴백: yfinance .KS / .KQ
        suffix = ".KQ" if code in KNOWN_KOSDAQ_CODES else ".KS"
        yf_sym = f"{code}{suffix}"
    else:
        yf_sym = code

    # 해외 주식 (또는 KR 폴백): yfinance fast_info + 1분봉 최신 체결가
    try:
        import yfinance as yf

        tk = yf.Ticker(yf_sym)
        price = None
        prev_close = None
        source = "yfinance_live"

        try:
            fi = tk.fast_info
            lp = fi.get("lastPrice") or fi.get("regularMarketPrice")
            pc = fi.get("previousClose") or fi.get("regularMarketPreviousClose")
            if lp and float(lp) > 0:
                price = float(lp)
            if pc and float(pc) > 0:
                prev_close = float(pc)
        except Exception:
            pass

        # 장중 실시간 체결 반영을 위해 1분봉 최신 캔들로 보강
        try:
            hist = tk.history(period="1d", interval="1m", prepost=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                last_1m = float(hist["Close"].dropna().iloc[-1])
                if last_1m > 0:
                    price = last_1m
                    source = "yfinance_1m"
        except Exception:
            pass

        if price and price > 0:
            day_chg = ((price / prev_close) - 1.0) * 100.0 if (prev_close and prev_close > 0) else 0.0
            return {
                "price": price,
                "day_change_pct": round(day_chg, 2),
                "market_status": "LIVE",
                "source": source,
            }
    except Exception:
        pass

    return None


def poll_once_and_record(
    db: PaperTradingDB,
    trader: LivePaperTrader,
    auto_guard: bool = False,
) -> dict[str, Any]:
    """보유 중인 모든 종목의 실시간 체결가를 1회 수집하고 오름/내림·평가손익을 DB & CSV에 기록."""
    pos_df = db.get_positions_df()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    acc = db.get_account()
    fx_now = fetch_realtime_fx_usdkrw(fallback=float(acc.get("fx_usdkrw", 1380.0)))

    if pos_df.empty:
        summary = trader.get_portfolio_summary(fx_usdkrw=fx_now)
        return {"recorded_at": now_str, "summary": summary, "ticks": [], "auto_orders": []}

    prev_ticks = db.get_latest_position_tick_map()
    price_updates: dict[str, float] = {}
    tick_records: list[dict[str, Any]] = []
    auto_orders: list[dict[str, Any]] = []

    for _, p in pos_df.iterrows():
        market = str(p["market"])
        code = str(p["code"])
        name = str(p["name"])
        currency = str(p["currency"])
        shares = float(p["shares"])
        avg_cost_krw = float(p["avg_cost_krw"])
        prev_px = float(prev_ticks[code]["price_local"]) if code in prev_ticks else float(p["current_price_local"])

        quote = fetch_realtime_quote(market, code)
        if quote is None:
            cur_px = float(p["current_price_local"])
            day_chg = 0.0
            src = "cached"
        else:
            cur_px = float(quote["price"])
            day_chg = float(quote["day_change_pct"])
            src = str(quote["source"])

        price_updates[code] = cur_px
        tick_chg_pct = ((cur_px / prev_px) - 1.0) * 100.0 if prev_px > 0 else 0.0
        if tick_chg_pct > 0.005:
            direction = "▲ 상승"
        elif tick_chg_pct < -0.005:
            direction = "▼ 하락"
        else:
            direction = "─ 보합"

        fx_m = 1.0 if currency == "KRW" else fx_now
        val_krw = shares * cur_px * fx_m
        cost_krw = shares * avg_cost_krw
        pnl_krw = val_krw - cost_krw
        pnl_pct = (pnl_krw / (cost_krw + 1e-9)) * 100.0

        # 목표가 / 손절선 / 스마트 트레일링 스탑 체크
        avg_local = float(p["avg_cost_local"])
        local_ret_pct = ((cur_px / avg_local) - 1.0) * 100.0 if avg_local > 0 else 0.0
        status_note = f"{src} · 주가 {local_ret_pct:+.2f}% (당일 {day_chg:+.2f}%)"
        stop_px = float(p["stop_loss_local"]) if p["stop_loss_local"] else None
        target_px = float(p["target_price_local"]) if p["target_price_local"] else None

        # [스마트 트레일링 스탑] 순수 주가가 +1.8% 이상 오르면 본절 위(+0.4%) 및 고점 대비 -1.2%로 손절선 자동 상향 잠금!
        if auto_guard and avg_local > 0 and local_ret_pct >= 1.8:
            trail_cand = max(avg_local * 1.004, cur_px * 0.988)
            if market == "KR":
                trail_cand = float(round(trail_cand))
            else:
                trail_cand = round(float(trail_cand), 2)
            if stop_px is None or trail_cand > stop_px + (1.0 if market == "KR" else 0.02):
                stop_px = trail_cand
                try:
                    with db._connect() as conn:
                        conn.execute(
                            "UPDATE positions SET stop_loss_local=? WHERE market=? AND code=?",
                            (stop_px, market, code),
                        )
                    status_note = f"🔒 수익보호 트레일링스탑 상향({stop_px:,.2f}) · 주가 {local_ret_pct:+.2f}%"
                except Exception:
                    pass

        if stop_px and cur_px <= stop_px:
            is_profit_lock = avg_local > 0 and stop_px > avg_local
            label = "트레일링 수익보존 익절" if is_profit_lock else "실시간 자동 손절"
            status_note = f"{'🟢' if is_profit_lock else '⚠️'} {label}({stop_px:,.2f})"
            if auto_guard:
                try:
                    ao = trader.sell(market, code, cur_px, 1.0, fx_usdkrw=fx_now, reason=f"{label} ({cur_px:,.2f} <= {stop_px:,.2f})")
                    auto_orders.append(ao)
                    status_note = f"{'🟢' if is_profit_lock else '🔴'} {label} 체결"
                except Exception:
                    pass
        elif target_px and cur_px >= target_px:
            status_note = f"🎯 목표가({target_px:,.2f}) 도달"
            if auto_guard:
                try:
                    ao = trader.sell(market, code, cur_px, 1.0, fx_usdkrw=fx_now, reason=f"실시간 자동 익절 ({cur_px:,.2f} >= {target_px:,.2f})")
                    auto_orders.append(ao)
                    status_note = "🟢 자동 익절 체결"
                except Exception:
                    pass
        elif local_ret_pct >= 1.5:
            status_note = f"강세 수익권 (주가 {local_ret_pct:+.2f}%) · {src}"
        elif local_ret_pct <= -3.0:
            status_note = f"주의 하락권 (주가 {local_ret_pct:+.2f}%) · {src}"

        tick_records.append(
            {
                "recorded_at": now_str,
                "market": market,
                "code": code,
                "name": name,
                "currency": currency,
                "price_local": round(cur_px, 4),
                "prev_price_local": round(prev_px, 4),
                "tick_change_pct": round(tick_chg_pct, 3),
                "day_change_pct": round(day_chg, 2),
                "pnl_krw": round(pnl_krw, 0),
                "pnl_pct": round(pnl_pct, 3),
                "direction": direction,
                "status_note": status_note,
            }
        )

    # 계좌 현재가 일괄 갱신
    trader.refresh_prices(price_updates, fx_usdkrw=fx_now)
    summary = trader.get_portfolio_summary(fx_usdkrw=fx_now)

    # SQLite 실시간 틱 로그 저장
    with db._connect() as conn:
        conn.execute(
            """
            INSERT INTO live_account_ticks
            (recorded_at, total_equity_krw, cash_krw, invested_krw, unrealized_pnl_krw, return_pct, fx_usdkrw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                float(summary["total_equity_krw"]),
                float(summary["cash_krw"]),
                float(summary["invested_krw"]),
                float(summary["unrealized_pnl_krw"]),
                float(summary["total_return_pct"]),
                float(fx_now),
            ),
        )
        for tr in tick_records:
            conn.execute(
                """
                INSERT INTO live_position_ticks
                (recorded_at, market, code, name, currency, price_local, prev_price_local,
                 tick_change_pct, day_change_pct, pnl_krw, pnl_pct, direction, status_note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tr["recorded_at"], tr["market"], tr["code"], tr["name"], tr["currency"],
                    tr["price_local"], tr["prev_price_local"], tr["tick_change_pct"],
                    tr["day_change_pct"], tr["pnl_krw"], tr["pnl_pct"], tr["direction"], tr["status_note"],
                ),
            )

    # 실시간 학습/모니터링 CSV에도 계속 누적 저장
    try:
        PAPER_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = PAPER_DATASET_DIR / "live_realtime_ticks.csv"
        df_new = pd.DataFrame(tick_records)
        df_new["total_equity_krw"] = round(float(summary["total_equity_krw"]), 0)
        df_new["fx_usdkrw"] = round(float(fx_now), 2)
        header = not csv_path.exists()
        df_new.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return {
        "recorded_at": now_str,
        "summary": summary,
        "ticks": tick_records,
        "auto_orders": auto_orders,
    }
