"""누적된 AI 학습·실시간 모의투자 데이터를 집대성해 '바로 시작하는 소액(기본 10만 원) 실전 키트'를 생성하는 모듈.

언제든 사용자가 실전 투자를 시작하려 할 때 즉시 꺼내 쓸 수 있도록:
1. 지금까지 쌓인 자가진화 시뮬레이션 성적표 (`evolution_generations`)
2. 30초 단위 실시간 시세·모의지갑 체결 검증 기록 (`live_position_ticks`, `orders`)
3. 만기 자동 채점 예측 기록 (`prediction_memory.db`)
4. 폴리마켓 시장 날씨 + 챔피언 승률 필터를 통과한 최신 우량 종목 랭킹
을 하나로 묶어:
- `latest_10manwon_plan.csv` (10만 원 원단위 소수점 매수·손절·익절 장바구니 표)
- `latest_10manwon_starter_report.md` (초보자용 따라 하기 가이드북 & 누적 성적 요약)
- `ai_accumulated_data_pack.zip` (모든 원본 CSV + 실전 가이드북 일괄 압축본)
을 `data/starter_pack/` 폴더에 자동 저장·갱신합니다.
"""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from ..config import DATA_DIR, OUTPUT_DIR
from ..data.alt_intelligence import AltMarketIntelligenceEngine
from ..storage.memory_db import PredictionMemoryDB
from .paper_engine import PAPER_DATASET_DIR, PaperTradingDB
from .winrate_evolver import load_champion_artifact, load_evolution_history_df

STARTER_PACK_DIR = DATA_DIR / "starter_pack"


def _load_latest_screening_df() -> tuple[pd.DataFrame, str]:
    if not OUTPUT_DIR.exists():
        return pd.DataFrame(), "없음"
    files = sorted(OUTPUT_DIR.glob("prediction_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if f.stat().st_size < 100:
            continue
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if not df.empty:
                return df, f.name
        except Exception:
            continue
    return pd.DataFrame(), "없음"


def build_starter_allocation(
    budget_krw: float = 100_000.0,
    max_picks: int = 3,
    db: PaperTradingDB | None = None,
) -> dict[str, Any]:
    """지정된 소액 예산(기본 10만 원)을 폴리마켓 날씨 비상금 + 챔피언 검증 상위 종목으로 천 원 단위 배분."""
    paper_db = db or PaperTradingDB()
    mem_db = PredictionMemoryDB()
    champ = load_champion_artifact()
    evo_df = load_evolution_history_df(paper_db)
    pulse = AltMarketIntelligenceEngine(cache_ttl_minutes=20).get_macro_pulse(use_cache=True)
    screen_df, screen_file = _load_latest_screening_df()

    # 1) 누적 데이터 현황 집계
    acc_ticks = paper_db.get_live_account_ticks_df(limit=5000)
    orders_df = paper_db.get_orders_df(limit=500)
    with paper_db._connect() as conn:
        pos_tick_cnt = int(conn.execute("SELECT COUNT(*) FROM live_position_ticks").fetchone()[0])
    scorecard = mem_db.get_scorecard_summary()

    total_gens = len(evo_df)
    total_sim_trades = int(evo_df["closed_trades"].sum()) if not evo_df.empty else 0
    base_wr = float(evo_df.iloc[0]["win_rate_pct"]) if not evo_df.empty else 0.0
    champ_wr = champ.champion_win_rate_pct if champ else base_wr
    champ_gen = champ.generation if champ else 0

    sell_orders = orders_df[orders_df["side"] == "SELL"] if not orders_df.empty else pd.DataFrame()
    live_closed_cnt = len(sell_orders)
    live_win_cnt = int((sell_orders["realized_pnl_krw"].fillna(0) > 0).sum()) if not sell_orders.empty else 0

    # 2) 예산 배분: 폴리마켓 날씨 기반 현금 비상금 먼저 떼어두기 (천 원 단위 깔끔한 반올림)
    invest_cap = float(pulse.recommended_invest_cap)
    raw_invest = budget_krw * invest_cap
    invest_budget = float(int(raw_invest // 1000) * 1000)
    cash_reserve = max(budget_krw - invest_budget, 0.0)

    # 3) 종목 엄선: 챔피언 승률 필터 통과 + 매수 의견 + 재무 건전 종목 우선
    picks_rows: list[dict[str, Any]] = []
    if not screen_df.empty:
        df = screen_df.copy()
        if "실전승률_점수(%)" not in df.columns:
            df["실전승률_점수(%)"] = df.get("상승확률(%)", 55.0)
        if "승률필터_통과" not in df.columns:
            df["승률필터_통과"] = np.where(df["최종의견"].astype(str).str.contains("매수", na=False), "Y", "N")

        # 1순위: 매수 의견 & 승률필터 통과 & 재무점수 >= 50
        cands = df[
            df["최종의견"].astype(str).str.contains("매수", na=False)
            & (df["승률필터_통과"].astype(str) == "Y")
        ].copy()
        if cands.empty:
            cands = df[df["최종의견"].astype(str).str.contains("매수", na=False)].copy()
        if cands.empty:
            cands = df.sort_values("실전승률_점수(%)", ascending=False).head(max_picks).copy()

        # 실전승률 점수 + 위원회 점수 결합 정렬
        cands["_combo"] = cands["실전승률_점수(%)"].astype(float) * 0.65 + cands["위원회_점수"].astype(float) * 0.35
        cands = cands.sort_values("_combo", ascending=False).head(max_picks)

        n_p = len(cands)
        if n_p > 0 and invest_budget >= 3000:
            w_raw = cands["실전승률_점수(%)"].astype(float).clip(lower=45.0).to_numpy()
            w_norm = w_raw / np.sum(w_raw)
            allocs = [float(int((invest_budget * w) // 1000) * 1000) for w in w_norm]
            # 천 원 단위 절사로 남은 자투리는 1위 종목에 배정
            rem = invest_budget - sum(allocs)
            if rem >= 1000:
                allocs[0] += float(int(rem // 1000) * 1000)
            cash_reserve = budget_krw - sum(allocs)

            tgt_col = next((c for c in cands.columns if "예상가" in c), "5일_예상가")
            ret_col = next((c for c in cands.columns if "기대수익률" in c), "5일_기대수익률(%)")

            # 챔피언 정책의 안전 브레이크 / 익절 비율 반영
            pol_stop = champ.policy.stop_loss_pct if champ else -0.055
            pol_tp = champ.policy.take_profit_pct if champ else 0.08
            pol_lock_trig = champ.policy.breakeven_trigger_pct if (champ and champ.policy.breakeven_trigger_pct < 0.2) else 0.025

            for idx, ((_, r), amt) in enumerate(zip(cands.iterrows(), allocs), start=1):
                cur = str(r["통화"])
                px = float(r["현재가"])
                mkt_str = "국내주식 (소수점/일반)" if "KR" in str(r["시장"]) else "미국주식 (천원 소수점 주문)"
                model_low = float(r.get("하한_10%", px * (1.0 + pol_stop)))
                stop_px = max(model_low, px * (1.0 + max(pol_stop, -0.065)))
                ai_tgt = float(r.get(tgt_col, px * 1.04))
                tgt_px = max(ai_tgt, px * 1.036) if ai_tgt > px else px * (1.0 + max(min(pol_tp, 0.06), 0.036))
                tgt_pct_dec = (tgt_px / px) - 1.0
                eff_lock_trig = min(pol_lock_trig, max(tgt_pct_dec * 0.55, 0.018))
                lock_px = px * (1.0 + eff_lock_trig)

                fmt_p = (lambda v: f"{v:,.0f}원") if cur == "KRW" else (lambda v: f"${v:,.2f}")
                stop_pct = (stop_px / px - 1.0) * 100.0
                tgt_pct = tgt_pct_dec * 100.0

                picks_rows.append(
                    {
                        "순위": f"{idx}순위",
                        "종목명": str(r["종목명"]),
                        "티커(검색코드)": str(r["티커"]),
                        "주문방법": mkt_str,
                        "매수할_금액(원)": int(amt),
                        "비중(%)": round(amt / max(budget_krw, 1.0) * 100.0, 1),
                        "기준_현재가": fmt_p(px),
                        "안전브레이크_손절가": f"{fmt_p(stop_px)} ({stop_pct:+.1f}%)",
                        "본전잠금_기준가": f"{fmt_p(lock_px)} (+{eff_lock_trig*100:.1f}% 오르면 본전보호)",
                        "목표_익절가": f"{fmt_p(tgt_px)} ({tgt_pct:+.1f}%)",
                        "훈련된_실전승률(%)": round(float(r["실전승률_점수(%)"]), 1),
                        "5일_기대수익률(%)": round(float(r.get(ret_col, 0.0)), 2),
                        "쉬운_선정이유": f"재무 {r.get('재무등급', '-')} · {str(r.get('큰손_한줄요약', '')).split('|')[0].strip()}",
                    }
                )

    plan_df = pd.DataFrame(picks_rows)
    stats = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "budget_krw": int(budget_krw),
        "invest_krw": int(budget_krw - cash_reserve),
        "cash_reserve_krw": int(cash_reserve),
        "cash_reserve_pct": round(cash_reserve / max(budget_krw, 1.0) * 100.0, 1),
        "weather_icon": pulse.weather_icon,
        "weather_title": pulse.weather_title_ko,
        "fear_greed_score": pulse.fear_greed_score,
        "total_generations": total_gens,
        "total_simulated_trades": total_sim_trades,
        "baseline_win_rate_pct": round(base_wr, 1),
        "champion_generation": champ_gen,
        "champion_win_rate_pct": round(champ_wr, 1),
        "live_account_ticks": len(acc_ticks),
        "live_position_ticks": pos_tick_cnt,
        "live_closed_orders": live_closed_cnt,
        "live_win_orders": live_win_cnt,
        "predictions_logged": scorecard.total_logged,
        "predictions_evaluated": scorecard.evaluated_count,
        "predictions_hit_rate_pct": round(scorecard.hit_rate * 100.0, 1) if scorecard.evaluated_count > 0 else None,
        "screening_source": screen_file,
        "champion_rules_ko": champ.learned_rules_ko if champ else [],
    }
    return {"stats": stats, "plan_df": plan_df, "evo_df": evo_df, "orders_df": orders_df, "screen_df": screen_df}


def render_starter_markdown(bundle: dict[str, Any]) -> str:
    """초보자가 그대로 보고 스마트폰 증권앱에서 따라 할 수 있는 마크다운 가이드북 생성."""
    s = bundle["stats"]
    plan_df: pd.DataFrame = bundle["plan_df"]

    lines = [
        f"# 🌱 내 돈 {s['budget_krw']:,}원 실전 바로 시작 가이드북 (AI 누적 데이터 리포트)",
        f"- **자료 생성 시각**: `{s['generated_at']}`",
        f"- **누적 훈련 데이터**: 총 **{s['total_generations']}세대** 시뮬레이션 (**{s['total_simulated_trades']:,}번** 가상 매매 연습 완료)",
        f"- **실시간 장중 감시 누적**: **{s['live_position_ticks']:,}건** 시세 기록 · 실전 만기 추적 **{s['predictions_logged']}건**",
        f"- **AI 검증 승률 진화**: 초보 1세대 `{s['baseline_win_rate_pct']}%` ➡️ **챔피언(제{s['champion_generation']}세대) `{s['champion_win_rate_pct']}%`**",
        "",
        "---",
        "",
        f"## 1️⃣ 오늘의 {s['budget_krw']:,}원 황금 비율 (폴리마켓 날씨 반영)",
        f"- {s['weather_icon']} **시장 날씨**: {s['weather_title']} (심리 온도 `{s['fear_greed_score']:.0f}도`)",
        f"- 🏦 **통장에 그대로 남겨둘 안전 비상금**: **`{s['cash_reserve_krw']:,}원` ({s['cash_reserve_pct']}%)**",
        f"- 🛒 **주식에 나눠 담을 금액**: **`{s['invest_krw']:,}원`** (아래 표대로 천 원 단위 소수점 주문!)",
        "",
        "## 2️⃣ 오늘 바로 따라 사는 '소수점 장바구니' & 손절·익절 가격표",
    ]

    if plan_df.empty:
        lines.append("> 현재 챔피언 승률 기준을 만족하는 매수 종목이 없어, 전액 현금 관망을 권장합니다.")
    else:
        for _, r in plan_df.iterrows():
            lines.extend(
                [
                    f"### ✅ {r['순위']}: {r['종목명']} (`{r['티커(검색코드)']}`) — **{r['매수할_금액(원)']:,}원어치 사기**",
                    f"- **주문 방법**: 증권사 앱(토스증권·카카오페이증권·미니스탁 등) 검색창에 `{r['티커(검색코드)']}` 입력 → **'소수점/금액으로 사기'** 선택 → **`{r['매수할_금액(원)']:,}원`** 입력",
                    f"- **기준 현재가**: `{r['기준_현재가']}` (훈련된 AI 실전승률 **`{r['훈련된_실전승률(%)']}%`**)",
                    f"- 🛡️ **안전 브레이크 (손절선)**: **`{r['안전브레이크_손절가']}`** 밑으로 내려가면 미련 없이 팔아서 원금 지키기!",
                    f"- 🔒 **수익 잠금 팁**: **`{r['본전잠금_기준가']}`** → 이제부터는 최소 본전 위에서만 팔기!",
                    f"- 🎯 **목표 수익가 (익절선)**: **`{r['목표_익절가']}`** 근처에 오면 욕심 부리지 않고 팔아서 수익 확정!",
                    f"- 💡 **고른 이유**: {r['쉬운_선정이유']}",
                    "",
                ]
            )

    lines.extend(
        [
            "---",
            "## 3️⃣ AI가 수천 번 모의연습 끝에 터득한 '절대 안 물리는 약속 4가지'",
        ]
    )
    for rule in s["champion_rules_ko"]:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "- 💡 **초보 철칙**: 한 종목에 10만 원 몰빵 금지! 급등해서 뉴스 1면에 나온 날 따라 사기 금지!",
            "",
        ]
    )
    return "\n".join(lines)


def export_starter_pack_files(budget_krw: float = 100_000.0, db: PaperTradingDB | None = None) -> dict[str, Any]:
    """`data/starter_pack/` 디렉터리에 최신 가이드(.md), 장바구니(.csv), 전체 누적 데이터(.zip) 일괄 저장."""
    STARTER_PACK_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_starter_allocation(budget_krw=budget_krw, db=db)
    md_text = render_starter_markdown(bundle)

    md_path = STARTER_PACK_DIR / "latest_10manwon_starter_report.md"
    csv_path = STARTER_PACK_DIR / "latest_10manwon_plan.csv"
    zip_path = STARTER_PACK_DIR / "ai_accumulated_data_pack.zip"

    md_path.write_text(md_text, encoding="utf-8")
    bundle["plan_df"].to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 모든 누적 데이터를 하나의 ZIP 패키지로 압축
    mem_buf = io.BytesIO()
    with zipfile.ZipFile(mem_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01_10만원_실전바로시작_가이드북.md", md_text.encode("utf-8"))
        zf.writestr("02_10만원_종목배분_손절익절표.csv", bundle["plan_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))
        if not bundle["evo_df"].empty:
            zf.writestr("03_AI자가진화_세대별승률기록.csv", bundle["evo_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))
        if not bundle["screen_df"].empty:
            zf.writestr("04_최신_AI종목추천_전체랭킹.csv", bundle["screen_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))
        if not bundle["orders_df"].empty:
            zf.writestr("05_실시간모의지갑_매매체결기록.csv", bundle["orders_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))

        live_ticks_csv = PAPER_DATASET_DIR / "live_realtime_ticks.csv"
        if live_ticks_csv.exists():
            zf.write(live_ticks_csv, arcname="06_30초단위_실시간시세추적데이터.csv")
        learn_csv = PAPER_DATASET_DIR / "latest_learning_dataset.csv"
        if learn_csv.exists():
            zf.write(learn_csv, arcname="07_모의투자_진입피처_자가학습데이터셋.csv")

        mem_rec = PredictionMemoryDB().get_recent_records(limit=500)
        if not mem_rec.empty:
            zf.writestr("08_AI예측_만기채점대기및성적표.csv", mem_rec.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"))

    zip_bytes = mem_buf.getvalue()
    zip_path.write_bytes(zip_bytes)

    bundle["md_text"] = md_text
    bundle["md_path"] = str(md_path)
    bundle["csv_path"] = str(csv_path)
    bundle["zip_path"] = str(zip_path)
    bundle["zip_bytes"] = zip_bytes
    return bundle
