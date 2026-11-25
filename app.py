"""Global Stock AI Studio v4 — 주식 왕초보 맞춤 쉬운 한국어 UI & 폴리마켓·큰손 연동 플랫폼.

실행 방법:
    streamlit run app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stock_predictor.config import KR_STOCK_PRESETS, OUTPUT_DIR, PredictorConfig, US_STOCK_PRESETS
from stock_predictor.data.alt_intelligence import AltMarketIntelligenceEngine, MacroPredictionPulse
from stock_predictor.data.market_data import MarketDataLoader
from stock_predictor.models.cross_sectional import load_cross_sectional_artifact
from stock_predictor.pipeline import GlobalStockPipeline, StockAnalysisReport
from stock_predictor.trading.live_monitor import poll_once_and_record
from stock_predictor.trading.paper_engine import (
    DEFAULT_INITIAL_CASH_KRW,
    PAPER_DATASET_DIR,
    HistoricalPaperSimulator,
    LivePaperTrader,
    PaperTradingDB,
    SimulationReport,
)
from stock_predictor.trading.winrate_evolver import (
    WinRateEvolutionEngine,
    load_champion_artifact,
    load_evolution_history_df,
)
from stock_predictor.trading.starter_pack import export_starter_pack_files

load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="주린이 AI 모의투자 & 폴리마켓 상황판",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# 친근하고 가독성 높은 다크 핀테크 디자인 CSS (초보자 눈높이 카드 & 안전바 포함)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Malgun Gothic', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #10203b 0%, #091122 55%, #060a14 100%);
    color: #f1f5f9;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* 상단 친근한 배너 */
.hero-header {
    background: linear-gradient(135deg, rgba(15, 29, 58, 0.95), rgba(20, 48, 82, 0.88));
    border: 1px solid rgba(56, 189, 248, 0.30);
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
}
.hero-title {
    font-size: 1.68rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}
.hero-sub {
    color: #cbd5e1;
    font-size: 0.96rem;
    margin-top: 7px;
    line-height: 1.5;
}

/* 폴리마켓 시장 날씨 배너 */
.weather-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.92));
    border: 1px solid rgba(250, 204, 21, 0.32);
    border-radius: 18px;
    padding: 16px 22px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}
.weather-left {
    display: flex;
    align-items: center;
    gap: 15px;
    flex: 1;
    min-width: 280px;
}
.weather-emoji {
    font-size: 2.5rem;
    line-height: 1;
}
.weather-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #fef08a;
    margin-bottom: 4px;
}
.weather-desc {
    font-size: 0.88rem;
    color: #e2e8f0;
    line-height: 1.45;
}

/* 상단 핵심 지갑 카드 */
.kpi-card {
    background: linear-gradient(180deg, rgba(19, 30, 53, 0.96), rgba(13, 21, 38, 0.88));
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 18px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.24);
}
.kpi-label {
    font-size: 0.84rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 7px;
}
.kpi-value {
    font-size: 1.52rem;
    font-weight: 800;
    color: #ffffff;
    font-variant-numeric: tabular-nums;
    line-height: 1.25;
}
.kpi-sub {
    font-size: 0.84rem;
    margin-top: 7px;
    line-height: 1.4;
    font-variant-numeric: tabular-nums;
}

/* 초보자 가이드 한줄 팁 */
.guide-strip {
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 14px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 0.89rem;
    color: #e0f2fe;
    line-height: 1.55;
}

/* 종목 & 내 지갑 카드 */
.stock-card {
    background: linear-gradient(180deg, #13203a 0%, #0d1629 100%);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
}
.stock-card:hover {
    border-color: rgba(56, 189, 248, 0.55);
}
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 10px;
}
.card-title {
    font-size: 1.14rem;
    font-weight: 800;
    color: #ffffff;
}
.card-sub {
    font-size: 0.83rem;
    color: #94a3b8;
    margin-top: 3px;
}

/* 지갑 돈 비교 박스 */
.money-compare {
    background: rgba(15, 23, 42, 0.72);
    border-radius: 12px;
    padding: 11px 14px;
    margin: 10px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    border: 1px solid rgba(148, 163, 184, 0.14);
}

/* 안전 브레이크 ~ 목표 수익가 시각 게이지 */
.range-wrap {
    margin: 10px 0 8px 0;
}
.range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-bottom: 5px;
}
.range-track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(244,63,94,0.45) 0%, rgba(148,163,184,0.28) 45%, rgba(16,185,129,0.50) 100%);
}
.range-dot {
    position: absolute;
    top: -4px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #ffffff;
    border: 3px solid #38bdf8;
    box-shadow: 0 0 8px rgba(56,189,248,0.85);
    transform: translateX(-50%);
}

/* 쉬운 설명 박스 */
.easy-box {
    background: rgba(30, 41, 59, 0.70);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 0.87rem;
    line-height: 1.55;
    color: #e2e8f0;
    margin-top: 10px;
}
.easy-box-title {
    color: #7dd3fc;
    font-weight: 800;
    font-size: 0.80rem;
    margin-bottom: 3px;
}

/* 폴리마켓 베팅 카드 */
.poly-card {
    background: linear-gradient(180deg, rgba(22, 34, 62, 0.95), rgba(14, 22, 41, 0.95));
    border: 1px solid rgba(168, 85, 247, 0.30);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.prob-track {
    width: 100%;
    height: 9px;
    background: rgba(148, 163, 184, 0.20);
    border-radius: 999px;
    overflow: hidden;
    margin: 8px 0;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}

/* 신호등 뱃지 */
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
.badge-buy { background: rgba(16, 185, 129, 0.20); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.45); }
.badge-hold { background: rgba(245, 158, 11, 0.18); color: #fde68a; border: 1px solid rgba(245, 158, 11, 0.40); }
.badge-sell { background: rgba(244, 63, 94, 0.20); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.45); }
.badge-blue { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35); }
.badge-purple { background: rgba(168, 85, 247, 0.18); color: #e9d5ff; border: 1px solid rgba(168, 85, 247, 0.38); }

.pos-text { color: #34d399; font-weight: 800; }
.neg-text { color: #fb7185; font-weight: 800; }
.neu-text { color: #94a3b8; font-weight: 600; }

/* 탭 가독성 */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.78);
    padding: 7px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.18);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 11px;
    padding: 11px 18px;
    font-weight: 800;
    font-size: 0.95rem;
    color: #cbd5e1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: #ffffff !important;
}

/* LIVE 배지 */
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.16);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #6ee7b7;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.80rem;
    font-weight: 800;
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    display: inline-block;
}
</style>
"""


# ---------------------------------------------------------------------------
# 초보자 눈높이 변환 도우미 함수들
# ---------------------------------------------------------------------------
def fmt_price(price: float, currency: str) -> str:
    return f"{price:,.0f}원" if currency == "KRW" else f"${price:,.2f}"


def fmt_krw(val: float, signed: bool = False) -> str:
    return f"{val:+,.0f}원" if signed else f"{val:,.0f}원"


def fmt_manwon(val: float, signed: bool = False) -> str:
    """초보자가 감 잡기 쉬운 '약 O,OOO만 원' 표기."""
    man = val / 10_000.0
    if abs(man) >= 10_000:
        eok = int(abs(man) // 10_000)
        rem = abs(man) % 10_000
        sign = "+" if (signed and val > 0) else ("-" if val < 0 else "")
        return f"{sign}{eok}억 {rem:,.0f}만 원" if rem >= 10 else f"{sign}{eok}억 원"
    return f"{man:+,.0f}만 원" if signed else f"{man:,.0f}만 원"


def _easy_action_badge(action: str) -> str:
    a = str(action)
    if "강력 매수" in a:
        return '<span class="badge badge-buy">🟢 적극 사볼 만해요!</span>'
    if "매수" in a:
        return '<span class="badge badge-buy">🟢 조금씩 사볼 만해요</span>'
    if "과열" in a:
        return '<span class="badge badge-hold">🟡 너무 급등해서 잠시 대기</span>'
    if "승률필터" in a:
        return '<span class="badge badge-hold">🛡️ 승률 낮아 보여 AI가 참음</span>'
    if "매도" in a:
        return '<span class="badge badge-sell">🔴 지금은 피하세요</span>'
    return '<span class="badge badge-hold">🟡 지금은 지켜만 보세요</span>'


def _easy_health_label(grade: str, score: float = 50.0) -> str:
    g = str(grade).upper()
    if g.startswith("A") or score >= 75:
        return "💪 아주 튼튼함 (돈 잘 버는 우량회사)"
    if g.startswith("B") or score >= 58:
        return "🙂 양호함 (안정적인 회사)"
    if g.startswith("C") or score >= 42:
        return "😐 보통 (평범한 체력)"
    if g in ("-", "", "미제공"):
        return "📋 기본 우량주"
    return "⚠️ 체력 약함 (실적 확인 필요)"


def _easy_risk_label(risk: str) -> str:
    r = str(risk)
    if "매우 높음" in r:
        return "🎢 롤러코스터 주의 (가격이 크게 출렁여요)"
    if "높음" in r:
        return "🌊 출렁임 있는 편 (소액만 추천)"
    if "낮음" in r:
        return "🛋️ 마음 편함 (가격 움직임이 얌전해요)"
    return "⚖️ 보통 (무난한 움직임)"


def _translate_holding_reason_for_beginner(pos_row: pd.Series) -> str:
    """DB에 저장된 기술적 투자 근거를 초보자가 바로 이해하는 친절한 말로 번역."""
    name = str(pos_row.get("종목명", ""))
    grade = str(pos_row.get("재무등급", ""))
    raw = str(pos_row.get("투자근거", ""))

    parts: list[str] = []
    if grade.startswith("A") or "최우량" in raw or "우량" in raw:
        parts.append("회사 살림살이와 이익이 아주 튼튼하고")
    elif grade.startswith("B"):
        parts.append("회사가 돈을 안정적으로 벌고 있고")
    else:
        parts.append("최근 시장에서 주목받는 대표 기업이며")

    if "큰손 쌍끌이" in raw or "외국인" in raw or "임원" in raw:
        parts.append("큰손 투자자·예측시장 분위기도 긍정적이라")
    elif "호재" in raw or "이슈 +" in raw:
        parts.append("최근 반가운 뉴스가 많아")
    else:
        parts.append("AI 모델 5개가 단기 상승 가능성을 높게 봐서")

    return f"{name}은(는) {' '.join(parts)} AI가 내 지갑에 나눠 담아둔 종목이에요."


def _translate_order_story(order_row: pd.Series) -> str:
    """주문 체결 내역을 초보자용 한 줄 이야기로 변환."""
    side = str(order_row.get("side", ""))
    name = str(order_row.get("name", ""))
    net_krw = float(order_row.get("net_krw", 0.0))
    pnl = order_row.get("realized_pnl_krw")
    reason = str(order_row.get("reason", ""))
    ts = str(order_row.get("executed_at", ""))[:16].replace("T", " ")

    if side == "BUY":
        return f"🛒 [{ts}] **{name}** 주식을 **{fmt_manwon(net_krw)}**어치 새로 장바구니에 담았어요. (이유: AI 종합 추천)"
    pnl_v = float(pnl) if pd.notna(pnl) else 0.0
    if "익절" in reason or "목표가" in reason or pnl_v > 0:
        return (
            f"🎉 [{ts}] **{name}** 주식이 목표 가격에 도착해서 AI가 자동으로 팔아 "
            f"**{fmt_krw(pnl_v, signed=True)}** 수익을 지갑에 챙겼어요!"
        )
    if "비중" in reason or "축소" in reason:
        return f"⚖️ [{ts}] **{name}** 한 종목에 돈이 너무 몰리지 않도록 절반을 팔아 안전하게 현금으로 바꿨어요."
    if "손절" in reason:
        return (
            f"🛡️ [{ts}] **{name}** 가격이 안전 브레이크 선까지 내려가서, 더 큰 손해를 막기 위해 AI가 대신 정리했어요 "
            f"({fmt_krw(pnl_v, signed=True)})."
        )
    return f"📤 [{ts}] **{name}** 주식을 판매해 **{fmt_manwon(net_krw)}**을 현금으로 돌려받았어요 ({fmt_krw(pnl_v, signed=True)})."


def _load_latest_saved_screening() -> tuple[pd.DataFrame | None, str]:
    if not OUTPUT_DIR.exists():
        return None, ""
    csv_files = sorted(OUTPUT_DIR.glob("prediction_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        return None, ""
    latest = csv_files[0]
    try:
        df = pd.read_csv(latest, encoding="utf-8-sig")
        if "위원회_점수" in df.columns:
            df = df.sort_values(by="위원회_점수", ascending=False).reset_index(drop=True)
        return df, latest.name
    except Exception:
        return None, ""


def _build_easy_screening_story(row: pd.Series) -> str:
    """스크리닝 결과 1행을 초보자 눈높이 한 줄 설명으로 생성."""
    f_score = float(row.get("재무점수", 50.0))
    n_score = float(row.get("이슈감성", 0.0))
    alt_summary = str(row.get("큰손_한줄요약", ""))
    prob_v = float(row.get("상승확률(%)", 50.0))
    action = str(row.get("최종의견", "관망"))

    lines: list[str] = []
    if f_score >= 75:
        lines.append("회사 체력이 아주 튼튼해요")
    elif f_score >= 58:
        lines.append("회사 살림이 안정적이에요")
    else:
        lines.append("실적보다는 최근 이슈로 움직이는 중이에요")

    if n_score >= 0.20:
        lines.append("최근 좋은 뉴스가 나오고 있어요")
    elif n_score <= -0.15:
        lines.append("걱정스러운 뉴스가 있어 조심해야 해요")

    if alt_summary and alt_summary != "예측시장·큰손 중립" and alt_summary != "nan":
        lines.append(alt_summary.split("|")[0].strip())

    if "과열" in action:
        lines.append("최근 갑자기 너무 많이 올라서 지금은 조금 내려올 때까지 기다리는 게 안전해요")
    elif "매수" in action and prob_v >= 55:
        lines.append(f"AI가 본 5일 뒤 오를 가능성은 {prob_v:.0f}%로 긍정적이에요")

    return " · ".join(lines)


@st.cache_data(ttl=600, show_spinner=False)
def _get_cached_macro_pulse() -> MacroPredictionPulse:
    return AltMarketIntelligenceEngine(cache_ttl_minutes=15).get_macro_pulse(use_cache=True)


# ---------------------------------------------------------------------------
# 전문가용 접기 영역 차트 빌더
# ---------------------------------------------------------------------------
CHART_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15, 23, 42, 0.55)",
    font=dict(family="Pretendard, Malgun Gothic, sans-serif", color="#cbd5e1"),
)


def plot_live_intraday_equity(ticks_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
        subplot_titles=("내 모의지갑 실시간 총액 변화 (원)", "산 가격 대비 실시간 수익·손실 (원)"),
    )
    fig.add_trace(
        go.Scatter(
            x=ticks_df["recorded_at"], y=ticks_df["total_equity_krw"], mode="lines+markers",
            name="지갑 총액(원)", line=dict(color="#38bdf8", width=2.6), marker=dict(size=5),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.12)",
        ),
        row=1, col=1,
    )
    pnl_colors = ["#10b981" if v >= 0 else "#f43f5e" for v in ticks_df["unrealized_pnl_krw"]]
    fig.add_trace(
        go.Bar(x=ticks_df["recorded_at"], y=ticks_df["unrealized_pnl_krw"], name="수익·손실(원)", marker_color=pnl_colors),
        row=2, col=1,
    )
    fig.update_layout(**CHART_LAYOUT_BASE, height=360, showlegend=False, margin=dict(l=16, r=16, t=42, b=16))
    return fig


def plot_price_and_forecast(report: StockAnalysisReport) -> go.Figure:
    ohlcv = report.bundle.ohlcv.tail(90).copy()
    fc = report.forecast
    path = fc.forecast_path
    sym = report.bundle.symbol

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index, y=ohlcv["Close"], mode="lines",
            name="지금까지 실제 주가", line=dict(width=2.4, color="#cbd5e1"),
        )
    )
    fig.add_trace(go.Scatter(x=path["date"], y=path["high_price"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=path["date"], y=path["low_price"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(56, 189, 248, 0.22)", name="예상 움직임 범위 (안전선~기대선)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path["date"], y=path["median_price"], mode="lines+markers",
            name=f"AI {fc.horizon}일 뒤 예상 경로", line=dict(width=3.0, color="#38bdf8", dash="dash"), marker=dict(size=7),
        )
    )
    fig.update_layout(
        **CHART_LAYOUT_BASE,
        title=f"{sym.name} 최근 3개월 주가 흐름 & 앞으로 {fc.horizon}일 AI 예상 길잡이",
        height=390,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=16, r=16, t=52, b=16),
    )
    return fig


def plot_signal_attribution(report: StockAnalysisReport) -> go.Figure:
    bd = report.forecast.signal_breakdown
    labels = ["차트·패턴 AI", "비슷한 주식 비교", "회사 체력(재무)", "최근 뉴스", "🔮 폴리마켓·큰손", "최종 예상수익"]
    vals = [
        bd.get("단일종목_ML_DL_앙상블(%)", 0.0),
        bd.get("다중종목_통합알파_보정(%)", 0.0),
        bd.get("기업_재정실적_보정(%)", 0.0),
        bd.get("실시간_이슈뉴스_보정(%)", 0.0),
        bd.get("예측시장_큰손민심_보정(%)", 0.0),
        bd.get("최종_기대수익률(%)", 0.0),
    ]
    colors = [
        "#38bdf8" if i < 5 and v >= 0 else ("#fb7185" if i < 5 else ("#10b981" if v >= 0 else "#f43f5e"))
        for i, v in enumerate(vals)
    ]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, text=[f"{v:+.2f}%" for v in vals], textposition="outside"))
    fig.add_hline(y=0, line_color="#475569", line_dash="dot")
    fig.update_layout(**CHART_LAYOUT_BASE, title="어디서 점수를 얻었나요? (5가지 정보 반영 비율)", yaxis_title="%", height=300, margin=dict(l=12, r=12, t=44, b=16))
    return fig


# ---------------------------------------------------------------------------
# 탭 1: 👛 내 지갑 & 실시간 오르내림 (15초 자동 새로고침 초보자 뷰)
# ---------------------------------------------------------------------------
@st.fragment(run_every="15s")
def render_beginner_wallet_live() -> None:
    db = PaperTradingDB()
    trader = LivePaperTrader(db=db)
    live_summary = trader.get_portfolio_summary()
    acc_ticks = db.get_live_account_ticks_df(limit=200)
    latest_tick_map = db.get_latest_position_tick_map()
    orders_df = db.get_orders_df(limit=25)

    last_ts = acc_ticks["recorded_at"].iloc[-1].strftime("%H시 %M분 %S초") if not acc_ticks.empty else "방금 전"

    st.markdown(
        f"""
        <div class="guide-strip">
            <span class="live-pill"><span class="live-dot"></span> 15초마다 자동 확인 중 (최근 확인: {last_ts})</span>
            &nbsp;&nbsp;💡 <b>초보자 3초 설명서:</b> 내가 직접 지켜보지 않아도 AI가 30초마다 가격을 감시해요!
            가격이 <b>🎯 목표 수익가</b>에 닿으면 자동으로 팔아서 수익을 챙기고, 혹시 <b>🛡️ 안전 브레이크</b>까지 떨어지면 큰 손해가 나기 전에 알아서 멈춰줍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    pos_df = live_summary["positions_df"]
    if pos_df.empty:
        st.info("지금 내 지갑에 주식이 하나도 없고 전액 현금으로 보관 중이에요! 아래 `🤖 폴리마켓 날씨 맞춰 AI 추천주 자동 담기` 버튼을 눌러 시작해 보세요.")
        return

    st.markdown("### 🛍️ 지금 내 지갑에 들어있는 주식들 (실시간 오르내림)")
    cols = st.columns(2)
    for idx, (_, p) in enumerate(pos_df.iterrows()):
        code = str(p["티커"])
        cur = str(p["통화"])
        tk = latest_tick_map.get(code, {})
        direction = str(tk.get("direction", "─ 보합"))
        day_chg = float(tk.get("day_change_pct", 0.0)) if tk.get("day_change_pct") is not None else 0.0

        val_krw = float(p["평가금액(원)"])
        pnl_krw = float(p["평가손익(원)"])
        cost_krw = val_krw - pnl_krw
        local_ret = float(p.get("주가수익률(%)", p["수익률(%)"]))

        is_up = pnl_krw >= 0
        pnl_cls = "pos-text" if is_up else "neg-text"
        mood_pill = (
            f'<span class="badge badge-buy">😊 수익 중 ({local_ret:+.2f}%)</span>'
            if local_ret >= 0
            else f'<span class="badge badge-sell">🩹 조금 내림 ({local_ret:+.2f}%)</span>'
        )
        country_flag = "🇰🇷 한국주식" if "KR" in str(p["시장"]) else "🇺🇸 미국주식"

        # 안전 브레이크(손절가) ~ 현재가 ~ 목표 수익가(익절가) 위치 계산 (0~100%)
        cur_p = float(p["현재가"])
        stop_p = float(p["손절가"]) if pd.notna(p["손절가"]) and float(p["손절가"]) > 0 else cur_p * 0.93
        tgt_p = float(p["목표가"]) if pd.notna(p["목표가"]) and float(p["목표가"]) > 0 else cur_p * 1.06
        span = max(tgt_p - stop_p, 1e-6)
        dot_pct = min(max(((cur_p - stop_p) / span) * 100.0, 4.0), 96.0)

        easy_why = html.escape(_translate_holding_reason_for_beginner(p))
        tick_word = "방금 전보다 오르는 중 🔼" if "상승" in direction else ("방금 전보다 살짝 내림 🔽" if "하락" in direction else "가격 숨고르기 중 ─")

        card_html = f"""
        <div class="stock-card">
            <div class="card-top">
                <div>
                    <div class="card-title">{html.escape(str(p['종목명']))}</div>
                    <div class="card-sub">{country_flag} · 내 지갑의 <b>{p['비중(%)']:.1f}%</b> 차지 · {tick_word} (오늘 {day_chg:+.2f}%)</div>
                </div>
                <div>{mood_pill}</div>
            </div>
            <div class="money-compare">
                <div>
                    <div style="font-size:0.77rem;color:#94a3b8;">내가 산 돈 → 지금 가치</div>
                    <div style="font-size:1.05rem;font-weight:800;color:#f8fafc;">{fmt_manwon(cost_krw)} → {fmt_manwon(val_krw)}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.77rem;color:#94a3b8;">지금 팔면 손익</div>
                    <div style="font-size:1.12rem;" class="{pnl_cls}">{fmt_krw(pnl_krw, signed=True)}</div>
                </div>
            </div>
            <div class="range-wrap">
                <div class="range-labels">
                    <span>🛡️ 안전 브레이크: <b style="color:#fda4af;">{fmt_price(stop_p, cur)}</b></span>
                    <span>현재 <b>{fmt_price(cur_p, cur)}</b></span>
                    <span>🎯 목표 수익가: <b style="color:#6ee7b7;">{fmt_price(tgt_p, cur)}</b></span>
                </div>
                <div class="range-track">
                    <div class="range-dot" style="left:{dot_pct:.1f}%;"></div>
                </div>
            </div>
            <div class="easy-box">
                <div class="easy-box-title">💬 왜 AI가 이 주식을 샀나요?</div>
                {easy_why}
            </div>
        </div>
        """
        with cols[idx % 2]:
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button(f"💸 지금 바로 팔아서 현금 챙기기 ({p['종목명']})", key=f"easy_sell_{code}", use_container_width=True):
                mkt_code = "KR" if "KR" in str(p["시장"]) else "US"
                res = trader.sell(mkt_code, code, cur_p, 1.0, reason="초보자 직접 현금화 판매")
                st.success(f"{res['name']} 판매 완료! 현금 {fmt_manwon(res['net_krw'])}이 지갑으로 들어왔어요 ({fmt_krw(res['realized_pnl_krw'], signed=True)})")
                st.rerun()

    # 최근 AI가 자동으로 사고판 이야기 피드
    if not orders_df.empty:
        st.markdown("#### 🤖 최근 AI가 혼자 알아서 일한 기록 (쉬운 요약)")
        for _, od in orders_df.head(6).iterrows():
            st.markdown(f"- {_translate_order_story(od)}")


def render_wallet_tab(
    db: PaperTradingDB,
    trader: LivePaperTrader,
    live_summary: dict[str, Any],
    sim_report: SimulationReport | None,
    screen_df: pd.DataFrame | None,
    pulse: MacroPredictionPulse,
) -> None:
    render_beginner_wallet_live()

    st.write("")
    b1, b2, b3 = st.columns([1.2, 1.5, 1.0])
    with b1:
        if st.button("⚡ 지금 당장 최신 가격 확인하기", type="primary", use_container_width=True):
            with st.spinner("한국·미국 실시간 주가와 환율을 확인하고 있어요..."):
                poll_once_and_record(db=db, trader=trader, auto_guard=True)
            st.success("최신 가격 업데이트 완료!")
            st.rerun()
    with b2:
        cap_pct = int(pulse.recommended_invest_cap * 100)
        if st.button(f"🤖 폴리마켓 날씨 맞춰 AI 추천주 자동 담기 (최대 {cap_pct}%까지만 투자)", use_container_width=True):
            if screen_df is not None and not screen_df.empty:
                res = trader.execute_ai_portfolio_from_screening(
                    screen_df,
                    fx_usdkrw=live_summary["fx_usdkrw"],
                    max_total_invest_ratio=pulse.recommended_invest_cap,
                )
                if res:
                    st.success(f"폴리마켓 날씨 기준에 맞춰 AI 추천 주식 {len(res)}개를 새로 담았어요!")
                    st.rerun()
                else:
                    st.info("이미 좋은 추천 주식을 다 가지고 있거나, 안전 비상금 현금 비율을 지키고 있어요!")
            else:
                st.warning("'🛒 AI가 골라준 주식' 탭에서 먼저 종목 추천을 실행해 주세요.")
    with b3:
        if st.button("♻️ 지갑 다시 1억 원으로 리셋", use_container_width=True):
            db.reset_account(DEFAULT_INITIAL_CASH_KRW, fx_usdkrw=live_summary["fx_usdkrw"])
            st.warning("모의지갑이 깨끗하게 처음 1억 원 현금 상태로 돌아갔어요.")
            st.rerun()

    # 초보자가 궁금해할 만한 과거 1년 AI 성적 요약 (쉽게 한 줄 카드로)
    if sim_report is not None:
        st.markdown(
            f"""
            <div class="easy-box" style="margin-top:14px;border-left-color:#10b981;">
                <div class="easy-box-title" style="color:#6ee7b7;">🏆 참고: 이 AI한테 지난 1년 동안 1억 원을 맡겼더니 어떻게 됐나요?</div>
                실제로 지난 1년({sim_report.start_date} ~ {sim_report.end_date}) 동안 수수료와 세금까지 다 빼고 모의투자를 해본 결과,
                <b>1억 원이 {fmt_manwon(sim_report.final_equity_krw)} ({sim_report.total_return_pct:+.2f}%)</b>으로 불어났어요!
                (그냥 주식을 사놓고 가만히 있었을 때보다 <b>{sim_report.excess_return_pct:+.2f}%p</b> 더 잘 벌었어요.)
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 전문가/상세 기록은 깔끔하게 접어두기
    with st.expander("🤓 전문가용 상세 그래프 · 30초 실시간 기록표 · 직접 금액 정해서 사기 (궁금할 때만 클릭)", expanded=False):
        acc_ticks = db.get_live_account_ticks_df(limit=240)
        if not acc_ticks.empty:
            st.plotly_chart(plot_live_intraday_equity(acc_ticks), use_container_width=True)

        st.markdown("##### 🛒 내가 원하는 주식 직접 골라서 지갑에 담기")
        oc1, oc2, oc3 = st.columns([1.5, 1.2, 1.0])
        with oc1:
            man_sym = st.text_input("사고 싶은 회사 이름이나 코드", placeholder="예: 삼성전자, NVDA, AAPL", key="man_sym_easy")
        with oc2:
            man_budget = st.number_input("얼마어치 살까요? (원)", min_value=100_000, max_value=50_000_000, value=3_000_000, step=500_000)
        with oc3:
            st.write("")
            st.write("")
            if st.button("내 지갑에 담기", type="primary", use_container_width=True):
                if man_sym.strip():
                    try:
                        b = MarketDataLoader(use_cache=True).load_bundle(man_sym.strip(), lookback_years=1, include_deep_info=False)
                        px = float(b.ohlcv["Close"].iloc[-1])
                        r = trader.buy(
                            market=b.symbol.market, code=b.symbol.code, name=b.symbol.name,
                            currency=b.symbol.currency, price_local=px,
                            budget_krw=float(man_budget),
                            stop_loss_local=px * 0.93, target_price_local=px * 1.07,
                            reason="내가 직접 골라서 산 주식",
                        )
                        st.success(f"{r['name']} {r['shares']}주 담기 성공! ({fmt_manwon(r['net_krw'])})")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"주식을 담지 못했어요: {exc}")

        pos_ticks = db.get_live_position_ticks_df(limit=50)
        if not pos_ticks.empty:
            st.markdown("##### 🕒 30초 단위 실시간 가격 체크 원본 기록")
            st.dataframe(pos_ticks.head(25), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 탭 2: 🔮 폴리마켓 & 큰손·개미 정보실 (예측시장 + 임원/외인기관 + 레딧)
# ---------------------------------------------------------------------------
def render_polymarket_tab(pulse: MacroPredictionPulse, live_summary: dict[str, Any]) -> None:
    st.markdown("### 🔮 사람들이 실제 돈을 걸고 맞히는 '폴리마켓' & 큰손·개미 실시간 정보실")
    st.markdown(
        """
        <div class="guide-strip">
            💡 <b>왜 고수들은 뉴스만 안 보고 '폴리마켓(Polymarket)'을 챙겨 볼까요?</b><br>
            뉴스 댓글이나 말로 하는 전망은 틀려도 손해가 없지만, <b>폴리마켓·칼시(Kalshi)</b>는 전 세계 참가자들이 <b>자기 돈을 수백억 원씩 실제로 걸고</b> 미국 금리·경기침체·관세·AI 이슈를 맞히는 곳이라 훨씬 솔직하고 빨라요!<br>
            우리 AI는 여기에 <b>① 폴리마켓 돈 걸린 확률</b> + <b>② 회사 임원·외국인·기관 큰손의 실제 매수</b> + <b>③ 미국 개미 커뮤니티(레딧) 입소문</b>까지 합쳐서 <b>비상금 현금 비율과 주식 점수에 직접 반영</b>합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_c1, top_c2 = st.columns([3.2, 1.0])
    with top_c1:
        st.caption(f"최근 수집 시각: {pulse.fetched_at} · 시장 심리 온도계: {pulse.fear_greed_score:.0f}도 ({pulse.fear_greed_label_ko})")
    with top_c2:
        if st.button("🔄 폴리마켓·레딧 최신 새로고침", use_container_width=True):
            _get_cached_macro_pulse.clear()
            AltMarketIntelligenceEngine(cache_ttl_minutes=0).get_macro_pulse(use_cache=False)
            st.rerun()

    # 지금 이 정보가 내 투자에 어떻게 영향을 주는지 3칸 직관 카드
    cash_reserve_pct = int((1.0 - pulse.recommended_invest_cap) * 100)
    invest_cap_pct = int(pulse.recommended_invest_cap * 100)
    i1, i2, i3 = st.columns(3)
    i1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">1️⃣ 폴리마켓이 정해준 오늘 비상금 비율</div>
            <div class="kpi-value">현금 최소 {cash_reserve_pct}% 보관</div>
            <div class="kpi-sub neu-text">주식은 최대 {invest_cap_pct}%까지만 사고 나머지는 비상금 보호</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    i2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">2️⃣ 오늘 시장 심리 온도계 (0~100도)</div>
            <div class="kpi-value">{pulse.fear_greed_score:.0f}도</div>
            <div class="kpi-sub" style="color:#fde68a;">{html.escape(pulse.fear_greed_label_ko)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    bull_cnt = sum(1 for b in pulse.polymarket_bets if b.market_impact == "호재")
    bear_cnt = sum(1 for b in pulse.polymarket_bets if b.market_impact == "악재")
    i3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">3️⃣ 큰돈 걸린 경제 이슈 분위기</div>
            <div class="kpi-value"><span class="pos-text">호재 {bull_cnt}</span> vs <span class="neg-text">주의 {bear_cnt}</span></div>
            <div class="kpi-sub neu-text">총 {len(pulse.polymarket_bets)}개 글로벌 핵심 베팅 실시간 반영 중</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 섹션 A: 폴리마켓 & 칼시 실제 돈 걸린 베팅 카드
    st.markdown("#### 💰 전 세계 사람들이 실제 돈을 걸고 베팅 중인 경제·빅테크 예측 (Polymarket · Kalshi)")
    if pulse.polymarket_bets:
        pcols = st.columns(2)
        for idx, bet in enumerate(pulse.polymarket_bets[:10]):
            if bet.market_impact == "호재":
                imp_badge = '<span class="badge badge-buy">🟢 주식에 좋은 신호</span>'
                bar_color = "#10b981"
            elif bet.market_impact == "악재":
                imp_badge = '<span class="badge badge-sell">🔴 조심해야 할 신호</span>'
                bar_color = "#f43f5e"
            else:
                imp_badge = '<span class="badge badge-hold">⚪ 지켜볼 이슈</span>'
                bar_color = "#38bdf8"

            vol_man_usd = bet.volume_usd / 10_000.0
            vol_str = f"약 ${bet.volume_usd:,.0f} (한화 약 {bet.volume_usd * 1350 / 1e8:,.1f}억 원 판돈)" if bet.volume_usd >= 50_000 else f"거래량 ${bet.volume_usd:,.0f}"

            p_html = f"""
            <div class="poly-card">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
                    <div><span class="badge badge-purple">{html.escape(bet.source)}</span> <span class="badge badge-blue">{html.escape(bet.topic_ko)}</span></div>
                    <div>{imp_badge}</div>
                </div>
                <div style="font-size:1.02rem;font-weight:800;color:#ffffff;margin-top:6px;">
                    {html.escape(bet.easy_explain_ko)}
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{min(max(bet.yes_prob_pct, 4), 100):.0f}%;background:{bar_color};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.80rem;color:#cbd5e1;">
                    <span>🎯 현재 '예(Yes)' 베팅 확률: <b style="color:#fef08a;font-size:0.92rem;">{bet.yes_prob_pct:.1f}%</b></span>
                    <span>💵 걸린 돈: <b>{vol_str}</b></span>
                </div>
                <div style="font-size:0.76rem;color:#94a3b8;margin-top:5px;">원문 질문: {html.escape(bet.question)}</div>
            </div>
            """
            with pcols[idx % 2]:
                st.markdown(p_html, unsafe_allow_html=True)
    else:
        st.info("현재 예측시장 서버 응답을 기다리고 있어요. 위 새로고침 버튼을 눌러보세요.")

    st.divider()

    # 섹션 B: 내 지갑 속 종목들의 큰손(임원/외국인·기관) + 레딧 + 폴리마켓 진단 & 미국 개미 인기 순위
    col_big, col_soc = st.columns([1.15, 1.0])

    with col_big:
        st.markdown("#### 🏦 지금 내 지갑 속 주식들의 '큰손(임원·외국인·기관) & 민심' 체크")
        pos_df = live_summary["positions_df"]
        alt_engine = AltMarketIntelligenceEngine(cache_ttl_minutes=20)
        if not pos_df.empty:
            for _, p in pos_df.iterrows():
                mkt = "KR" if "KR" in str(p["시장"]) else "US"
                code = str(p["티커"])
                name = str(p["종목명"])
                yf_tk = f"{code}.KS" if mkt == "KR" else code
                sig = alt_engine.get_stock_alt_signal(mkt, code, name, yf_tk, macro_pulse=pulse, use_cache=True)
                s_badge = (
                    '<span class="badge badge-buy">큰손·민심 긍정</span>'
                    if sig.combined_signal >= 0.08
                    else ('<span class="badge badge-sell">큰손 매도·주의</span>' if sig.combined_signal <= -0.08 else '<span class="badge badge-hold">평온·중립</span>')
                )
                bullets_html = "<br>".join(f"• {html.escape(b)}" for b in sig.easy_bullets_ko)
                st.markdown(
                    f"""
                    <div class="stock-card" style="padding:14px 16px;margin-bottom:10px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <div style="font-weight:800;font-size:1.0rem;color:#ffffff;">{html.escape(name)} ({html.escape(code)})</div>
                            <div>{s_badge}</div>
                        </div>
                        <div style="font-size:0.85rem;color:#e2e8f0;line-height:1.55;">{bullets_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("지갑에 보유 중인 주식이 있으면 그 회사의 임원 매수·외국인 수급을 여기서 바로 체크해 드려요!")

    with col_soc:
        st.markdown("#### 🗣️ 지금 미국 개미들이 제일 많이 떠드는 주식 TOP 10 (레딧 실시간)")
        st.caption("※ 적당히 관심이 늘면 호재지만, 하루 만에 +250% 넘게 폭증한 주식은 유행 고점일 수 있어 AI가 자동으로 주의 표시를 줍니다!")
        if pulse.social_trending:
            for s in pulse.social_trending[:10]:
                medal = "🥇" if s.rank == 1 else ("🥈" if s.rank == 2 else ("🥉" if s.rank == 3 else f"#{s.rank}"))
                chg_cls = "pos-text" if s.mention_change_pct >= 0 else "neg-text"
                st.markdown(
                    f"""
                    <div style="background:rgba(15,23,42,0.82);border:1px solid rgba(148,163,184,0.16);border-radius:12px;padding:10px 14px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <b style="color:#fef08a;">{medal} {html.escape(s.ticker)}</b>
                            <span style="color:#cbd5e1;font-size:0.84rem;"> ({html.escape(s.name[:24])})</span>
                            <div style="font-size:0.78rem;color:#93c5fd;margin-top:2px;">{html.escape(s.mood_ko)}</div>
                        </div>
                        <div style="text-align:right;font-size:0.82rem;">
                            <div>언급 <b>{s.mentions:,}회</b></div>
                            <div class="{chg_cls}">어제보다 {s.mention_change_pct:+.0f}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("소셜 트렌드 데이터를 불러오는 중입니다.")


# ---------------------------------------------------------------------------
# 탭 3: 🛒 AI가 골라준 살 만한 주식 (신호등 + 쉬운 별점표)
# ---------------------------------------------------------------------------
def render_easy_store_tab(
    screen_df: pd.DataFrame | None,
    screen_source: str,
    config: PredictorConfig,
    trader: LivePaperTrader,
) -> None:
    st.markdown("### 🛒 AI가 회사 체력 · 뉴스 · 폴리마켓 · 큰손을 다 합쳐 골라준 주식 장터")
    st.markdown(
        """
        <div class="guide-strip">
            💡 <b>초보자 쇼핑 팁:</b> 초록색 <b>🟢 사볼 만해요</b> 표시가 붙은 주식만 천천히 살펴보세요!
            아무리 유명한 주식이라도 최근 며칠 새 너무 급등했으면 AI가 고점에 물리지 않도록 <b>🟡 잠시 대기</b> 신호등을 켜줍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔄 최신 정보(폴리마켓·큰손 포함)로 전체 추천 목록 다시 뽑기", expanded=False):
        default_basket = (
            "삼성전자, SK하이닉스, KB금융, 삼성바이오로직스, 한화에어로스페이스, HD현대중공업, 두산에너빌리티, "
            "NVDA, AAPL, MSFT, META, COST, V, LLY, OKLO"
        )
        basket_input = st.text_area("살펴볼 회사 목록 (쉼표로 구분)", value=default_basket, height=68)
        if st.button("🚀 지금 새로 AI 추천 뽑기", type="primary"):
            raw_items = [s.strip() for s in basket_input.replace("\n", ",").split(",") if s.strip()]
            if raw_items:
                prog = st.progress(0.0, text="회사 체력 + 뉴스 + 폴리마켓·큰손 데이터를 모으는 중...")
                df_new, _, errors = GlobalStockPipeline(config=config).screen(
                    raw_items,
                    progress_cb=lambda i, t, q: prog.progress((i - 1) / t, text=f"({i}/{t}) {q} 쉽게 분석하는 중..."),
                )
                prog.progress(1.0, text="분석 완료!")
                st.session_state["screen_df"] = df_new
                st.session_state["screen_source"] = "방금 폴리마켓·큰손 반영 실시간 분석"
                for sym, err in errors:
                    st.warning(f"{sym} 확인 실패: {err}")
                st.rerun()

    if screen_df is None or screen_df.empty:
        st.info("위 `🔄 최신 정보로 전체 추천 목록 다시 뽑기`를 열어서 실행해 주세요!")
        return

    f1, f2 = st.columns([1.3, 1.8])
    with f1:
        mkt_filter = st.radio("어느 나라 주식을 볼까요?", ["전체", "🇰🇷 한국 주식", "🇺🇸 미국 주식"], horizontal=True)
    with f2:
        act_filter = st.radio("어떤 신호등만 볼까요?", ["전체 보기", "🟢 지금 사볼 만한 주식만", "🟡 잠시 구경만 할 주식"], horizontal=True)

    view_df = screen_df.copy()
    if "한국" in mkt_filter:
        view_df = view_df[view_df["시장"].astype(str).str.contains("KR")]
    elif "미국" in mkt_filter:
        view_df = view_df[view_df["시장"].astype(str).str.contains("US")]

    if "사볼 만한" in act_filter:
        view_df = view_df[view_df["최종의견"].astype(str).str.contains("매수", na=False)]
    elif "구경만" in act_filter:
        view_df = view_df[~view_df["최종의견"].astype(str).str.contains("매수", na=False)]

    ret_col = next((c for c in view_df.columns if "기대수익률" in c), "5일_기대수익률(%)")
    tgt_col = next((c for c in view_df.columns if "예상가" in c), "5일_예상가")

    grid_cols = st.columns(2)
    for idx, (_, row) in enumerate(view_df.head(10).iterrows()):
        cur = str(row["통화"])
        code = str(row["티커"])
        name = str(row["종목명"])
        ret_v = float(row[ret_col])
        prob_v = float(row["상승확률(%)"])
        f_grade = str(row.get("재무등급", "-"))
        f_score = float(row.get("재무점수", 50.0))
        risk = str(row.get("리스크", "보통"))
        alt_txt = str(row.get("큰손_한줄요약", "예측시장·큰손 중립"))
        if alt_txt == "nan":
            alt_txt = "예측시장·큰손 중립"
        win_conf = float(row.get("실전승률_점수(%)", prob_v))
        win_note = str(row.get("승률필터_설명", f"예상 실전승률 {win_conf:.0f}%"))
        if win_note == "nan" or not win_note:
            win_note = f"예상 실전승률 {win_conf:.0f}%"

        ret_cls = "pos-text" if ret_v >= 0 else "neg-text"
        bar_col = "#10b981" if prob_v >= 55 else ("#38bdf8" if prob_v >= 50 else "#f43f5e")
        flag = "🇰🇷 한국" if "KR" in str(row["시장"]) else "🇺🇸 미국"
        story = html.escape(_build_easy_screening_story(row))

        card_html = f"""
        <div class="stock-card">
            <div class="card-top">
                <div>
                    <div class="card-title">{flag} · {html.escape(name)} <span style="font-size:0.84rem;color:#94a3b8;">({html.escape(code)})</span></div>
                    <div class="card-sub">{html.escape(_easy_health_label(f_grade, f_score))}</div>
                </div>
                <div>{_easy_action_badge(str(row['최종의견']))}</div>
            </div>
            <div class="money-compare">
                <div>
                    <div style="font-size:0.76rem;color:#94a3b8;">지금 1주 가격</div>
                    <div style="font-size:1.15rem;font-weight:800;">{fmt_price(float(row['현재가']), cur)}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.76rem;color:#94a3b8;">5일 뒤 AI 예상 가격</div>
                    <div style="font-size:1.10rem;" class="{ret_cls}">→ {fmt_price(float(row[tgt_col]), cur)} ({ret_v:+.2f}%)</div>
                </div>
            </div>
            <div style="font-size:0.83rem;color:#e2e8f0;line-height:1.65;">
                • 🤖 <b>5일 뒤 오를 가능성:</b> <b style="color:#fef08a;">{prob_v:.0f}%</b> · 🎯 <b>훈련된 실전승률: <span style="color:#6ee7b7;">{win_conf:.0f}%</span></b><br>
                • 🧬 <b>자가진화 판정:</b> {html.escape(win_note[:65])}<br>
                • {html.escape(_easy_risk_label(risk))}<br>
                • 🔮 <b>큰손·예측시장:</b> {html.escape(alt_txt[:75])}
            </div>
            <div class="prob-track"><div class="prob-fill" style="width:{min(max(prob_v, 5), 100):.0f}%;background:{bar_col};"></div></div>
            <div class="easy-box">
                <div class="easy-box-title">💡 초보자용 한 줄 이유</div>
                {story}
            </div>
        </div>
        """
        with grid_cols[idx % 2]:
            st.markdown(card_html, unsafe_allow_html=True)
            if "매수" in str(row["최종의견"]):
                if st.button(f"🛒 내 모의지갑에 500만 원어치 담아보기 ({name})", key=f"store_buy_{code}", use_container_width=True):
                    try:
                        mkt_c = "KR" if "KR" in str(row["시장"]) else "US"
                        r = trader.buy(
                            market=mkt_c, code=code, name=name, currency=cur,
                            price_local=float(row["현재가"]), budget_krw=5_000_000.0,
                            stop_loss_local=float(row.get("하한_10%", float(row["현재가"]) * 0.93)),
                            target_price_local=float(row[tgt_col]),
                            reason=f"AI 추천 장터 선택 ({_build_easy_screening_story(row)})",
                            ai_action=str(row["최종의견"]),
                            ai_score=float(row.get("위원회_점수", 20.0)),
                            up_prob=prob_v / 100.0,
                        )
                        st.success(f"🎉 {r['name']} {r['shares']}주를 모의지갑에 담았어요! ('내 지갑' 탭에서 실시간으로 확인하세요)")
                    except Exception as exc:
                        st.error(f"담기 실패: {exc}")

    with st.expander(f"🤓 전문가용 전체 종목 비교표 펼치기 ({screen_source})", expanded=False):
        st.dataframe(view_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 탭 4: 🔍 궁금한 주식 쉽게 물어보기 (3줄 성적표 + 폴리마켓·큰손 진단)
# ---------------------------------------------------------------------------
def render_easy_ask_tab(config: PredictorConfig, trader: LivePaperTrader) -> None:
    st.markdown("### 🔍 이 주식 지금 사도 될까요? — AI에게 초보 눈높이로 물어보기")
    c1, c2, c3, c4 = st.columns([1.0, 1.4, 1.5, 1.1])
    with c1:
        market_pick = st.selectbox("나라 선택", ["🇰🇷 한국 주식", "🇺🇸 미국 주식"])
    with c2:
        if "한국" in market_pick:
            preset_label = st.selectbox("유명한 회사 고르기", list(KR_STOCK_PRESETS.keys()))
            preset_val = KR_STOCK_PRESETS[preset_label]
        else:
            preset_label = st.selectbox("유명한 회사 고르기", list(US_STOCK_PRESETS.keys()))
            preset_val = US_STOCK_PRESETS[preset_label]
    with c3:
        custom_sym = st.text_input("또는 직접 이름/코드 입력", placeholder="예: 삼성전자, 카카오, NVDA, TSLA")
    with c4:
        st.write("")
        st.write("")
        run_btn = st.button("🔍 쉽게 알려줘!", type="primary", use_container_width=True)

    target_q = custom_sym.strip() if custom_sym.strip() else preset_val
    if run_btn:
        with st.spinner(f"'{target_q}'의 회사 체력, 최근 뉴스, 폴리마켓 확률, 큰손 매매를 한꺼번에 조사하고 있어요..."):
            try:
                st.session_state["last_report"] = GlobalStockPipeline(config=config).analyze(target_q)
            except Exception as exc:
                st.error(f"종목 확인 중 문제가 생겼어요: {exc}")

    if "last_report" not in st.session_state:
        st.info("위에서 궁금한 회사를 고르고 `🔍 쉽게 알려줘!` 버튼을 눌러보세요!")
        return

    report: StockAnalysisReport = st.session_state["last_report"]
    sym = report.bundle.symbol
    fc = report.forecast
    vd = report.verdict
    fp = report.bundle.fundamental_profile
    nr = report.bundle.news_report
    alt = report.bundle.alt_signal

    # 쉬운 3줄 진단 문장 조립
    health_line = _easy_health_label(fp.grade if fp else "-", fp.overall_score if fp else 50.0)
    if fp and fp.strengths:
        health_line += f" — {fp.strengths[0]}"

    alt_bullets = " / ".join(alt.easy_bullets_ko) if (alt and alt.easy_bullets_ko) else "큰손·예측시장 특이사항 없음"
    news_line = (
        f"최근 뉴스 {nr.article_count}건 중 좋은 소식 {nr.positive_count}건 · 걱정 소식 {nr.negative_count}건"
        if (nr and nr.article_count > 0)
        else "최근 특별한 뉴스 없이 조용해요"
    )

    st.markdown(
        f"""
        <div class="stock-card" style="border-color:rgba(56,189,248,0.55);padding:22px 24px;">
            <div class="card-top">
                <div>
                    <div class="card-title" style="font-size:1.42rem;">{'🇰🇷' if sym.market == 'KR' else '🇺🇸'} {html.escape(sym.name)} ({html.escape(sym.code)}) 초보자 3줄 성적표</div>
                    <div class="card-sub">지금 1주 가격: <b style="color:#ffffff;">{fmt_price(fc.current_price, sym.currency)}</b> · {_easy_risk_label(vd.risk_level)}</div>
                </div>
                <div>{_easy_action_badge(vd.action)}</div>
            </div>
            <div class="easy-box" style="font-size:0.94rem;line-height:1.75;">
                1️⃣ <b>회사 체력(돈 잘 버나요?):</b> {html.escape(health_line)}<br>
                2️⃣ <b>뉴스 · 폴리마켓 · 큰손 반응:</b> {html.escape(news_line)} · <b>{html.escape(alt_bullets)}</b><br>
                3️⃣ <b>AI의 {fc.horizon}일 뒤 예상:</b> 오를 가능성 <b style="color:#fef08a;">{fc.up_probability*100:.0f}%</b> ·
                훈련된 실전승률 점수 <b style="color:#6ee7b7;">{vd.win_confidence_pct:.0f}%</b> ({html.escape(vd.champion_filter_note)})!<br>
                👉 예상 가격은 <b style="color:#6ee7b7;">{fmt_price(fc.target_price_median, sym.currency)} ({fc.expected_pct_return*100:+.2f}%)</b>이고,
                혹시 <b style="color:#fda4af;">{fmt_price(fc.target_price_low, sym.currency)}</b> 밑으로 내려가면 안전 브레이크(손절)를 걸어두면 안심이에요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 바로 내 모의지갑에 담기
    bc1, bc2, _ = st.columns([1.3, 1.1, 2.2])
    with bc1:
        buy_amt = st.number_input("모의지갑에 담아볼 금액 (원)", min_value=500_000, max_value=30_000_000, value=5_000_000, step=500_000)
    with bc2:
        st.write("")
        st.write("")
        if st.button("🛒 내 모의지갑에 담기", type="primary", use_container_width=True):
            try:
                res = trader.buy(
                    market=sym.market, code=sym.code, name=sym.name, currency=sym.currency,
                    price_local=fc.current_price, budget_krw=float(buy_amt),
                    stop_loss_local=fc.target_price_low, target_price_local=fc.target_price_median,
                    reason=f"궁금한 주식 조회 후 매수 ({vd.action} · 상승확률 {fc.up_probability*100:.0f}%)",
                    ai_action=vd.action, ai_score=vd.composite_score, up_prob=fc.up_probability,
                )
                st.success(f"🎉 {res['name']} {res['shares']}주를 모의지갑에 담았어요! ({fmt_manwon(res['net_krw'])})")
            except Exception as exc:
                st.error(f"담기 실패: {exc}")

    # 쉬운 예상 길잡이 차트 + 5가지 점수 반영 그래프
    ch1, ch2 = st.columns([1.35, 1.0])
    with ch1:
        st.plotly_chart(plot_price_and_forecast(report), use_container_width=True)
    with ch2:
        st.plotly_chart(plot_signal_attribution(report), use_container_width=True)

    # 전문가 상세 내용은 접기 영역에 보존
    with st.expander("🤓 전문가용 AI 위원회 찬반 토론 · 최신 뉴스 원문 · 세부 모델 성적 펼쳐보기", expanded=False):
        bl, br = st.columns(2)
        with bl:
            st.success("**👍 좋은 점 (오를 만한 이유)**\n" + "\n".join(f"- {x}" for x in vd.bull_case))
        with br:
            st.error("**⚠️ 조심할 점 (내릴 수도 있는 이유)**\n" + "\n".join(f"- {x}" for x in vd.bear_case))
        if nr and nr.items:
            st.markdown("##### 📰 수집된 최신 뉴스 헤드라인")
            for it in nr.items[:8]:
                st.markdown(f"- **[{it.category}]** {it.title} *(감성 {it.sentiment:+.2f})*")


# ---------------------------------------------------------------------------
# 탭 5: 🧬 AI 혼자 연습하며 승률 올리기 (자가진화 훈련소)
# ---------------------------------------------------------------------------
@st.fragment(run_every="20s")
def render_evolution_live_status() -> None:
    db = PaperTradingDB()
    hist_df = load_evolution_history_df(db)
    champ = load_champion_artifact()

    if hist_df.empty or champ is None:
        st.info("아직 첫 진화 훈련 기록을 준비 중이에요! 아래 `🧬 지금 바로 AI 특훈 시키기` 버튼을 누르면 바로 시작됩니다.")
        return

    base_wr = float(hist_df.iloc[0]["win_rate_pct"])
    champ_wr = champ.champion_win_rate_pct
    gain_wr = champ_wr - base_wr
    total_gens = len(hist_df)

    st.markdown(
        f"""
        <div class="guide-strip">
            <span class="live-pill"><span class="live-dot"></span> 백그라운드 연속 연습 가동 중 (현재 총 {total_gens}세대 훈련 완료)</span>
            &nbsp;&nbsp;💡 <b>어떻게 혼자서 승률을 올리나요?</b><br>
            운동선수가 연습경기를 수백 번 뛰면서 약점을 고치듯, 우리 AI도 국내·미국 대표 주식 차트에서 <b>수백 번 모의매매를 반복</b>해요.
            그러다 <b>"어떨 때 사면 돈 잃고(고점 물림), 어떨 때 사면 수수료 떼고도 확실히 이기는지"</b>를 2차 승률 판별기(Meta-AI)가 배워서,
            이전 챔피언보다 <b>실전 검증 승률이 더 높은 전략만 새 챔피언으로 승격</b>시킵니다!
        </div>
        """,
        unsafe_allow_html=True,
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🐣 처음 초보 시절(1세대) 승률</div>
            <div class="kpi-value">{base_wr:.1f}%</div>
            <div class="kpi-sub neu-text">요령 없이 단순 AI 예측만 믿고 샀을 때</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    e2.markdown(
        f"""
        <div class="kpi-card" style="border-color:rgba(16,185,129,0.50);">
            <div class="kpi-label">🏆 지금 챔피언(제{champ.generation}세대) 승률</div>
            <div class="kpi-value"><span class="pos-text">{champ_wr:.1f}%</span></div>
            <div class="kpi-sub pos-text">초보 시절보다 승률 {gain_wr:+.1f}%p 상승!</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    e3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 챔피언 전략 검증 수익률</div>
            <div class="kpi-value">{champ.champion_return_pct:+.2f}%</div>
            <div class="kpi-sub neu-text">벌 때/잃을 때 비율(손익비) {champ.champion_profit_factor:.2f}배</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    e4.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">💪 누적 모의연습 매매 횟수</div>
            <div class="kpi-value">{champ.total_simulated_trades:,}번</div>
            <div class="kpi-sub neu-text">총 {total_gens}세대 반복 훈련 완료 ({champ.updated_at[11:]})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 세대별 승률 성장 그래프
    fig = go.Figure()
    bar_colors = ["#10b981" if int(c) == 1 else "#38bdf8" for c in hist_df["is_champion"]]
    labels = [f"{int(g)}세대" + (" 🏆" if int(c) == 1 else "") for g, c in zip(hist_df["generation"], hist_df["is_champion"])]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=hist_df["win_rate_pct"],
            marker_color=bar_colors,
            text=[f"{w:.1f}%" for w in hist_df["win_rate_pct"]],
            textposition="outside",
            name="검증 승률(%)",
        )
    )
    fig.add_hline(y=base_wr, line_dash="dot", line_color="#fb7185", annotation_text=f"1세대 초보 승률 ({base_wr:.1f}%)")
    fig.update_layout(
        **CHART_LAYOUT_BASE,
        title="📈 연습 세대가 쌓이면서 AI의 실제 검증 승률이 어떻게 올라갔나요? (🏆 = 현재 적용 중인 챔피언)",
        yaxis_title="수수료·세금 제외 실제 승률 (%)",
        yaxis_range=[0, min(100.0, float(hist_df["win_rate_pct"].max()) + 16.0)],
        height=350,
        margin=dict(l=16, r=16, t=48, b=16),
    )
    st.plotly_chart(fig, use_container_width=True)

    c_left, c_right = st.columns([1.25, 1.0])
    with c_left:
        rules_html = "<br>".join(f"{html.escape(r)}" for r in champ.learned_rules_ko)
        st.markdown(
            f"""
            <div class="stock-card" style="border-color:rgba(16,185,129,0.45);">
                <div class="card-title">📖 AI가 수백 번 모의매매 끝에 스스로 터득한 '안 물리는 필승 비법 노트'</div>
                <div class="card-sub">현재 선택된 챔피언 전략: <b>{html.escape(champ.policy.name)}</b> (추천 탭과 실시간 지갑에 자동 적용 중!)</div>
                <div class="easy-box" style="margin-top:12px;border-left-color:#10b981;line-height:1.75;">
                    {rules_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown("##### 🔍 이긴 매매 vs 돈 잃은 매매, 뭐가 달랐나요?")
        if champ.feature_win_insights:
            st.dataframe(pd.DataFrame(champ.feature_win_insights), use_container_width=True, hide_index=True)
        st.caption("👉 이 차이를 2차 승률 판별기(Meta-AI)가 외워서, 앞으로 돈 잃을 확률이 높은 주식은 애초에 '잠시 대기'로 걸러냅니다!")

    with st.expander("📋 전체 세대별 모의연습 성적표 펼쳐보기", expanded=False):
        show_h = hist_df[
            ["generation", "is_champion", "policy_name", "win_rate_pct", "total_return_pct", "profit_factor", "closed_trades", "created_at"]
        ].rename(
            columns={
                "generation": "세대", "is_champion": "챔피언", "policy_name": "연습한 전략 이름",
                "win_rate_pct": "실제 승률(%)", "total_return_pct": "수익률(%)", "profit_factor": "손익비",
                "closed_trades": "매매횟수", "created_at": "훈련시각",
            }
        )
        st.dataframe(show_h.sort_values("세대", ascending=False), use_container_width=True, hide_index=True)


def render_evolution_lab_tab(config: PredictorConfig, db: PaperTradingDB) -> None:
    st.markdown("### 🧬 AI가 혼자 시뮬레이션을 계속 돌리며 승률 올리는 훈련소")
    render_evolution_live_status()

    if st.button("🚀 지금 바로 AI에게 추가 특훈 3세대 더 시키기! (약 15초 소요)", type="primary"):
        status = st.empty()
        loader = MarketDataLoader(use_cache=True)
        syms = [
            "삼성전자", "SK하이닉스", "KB금융", "삼성바이오로직스", "현대차", "한화에어로스페이스", "HD현대중공업", "두산에너빌리티",
            "NVDA", "AAPL", "MSFT", "META", "COST", "V", "LLY", "OKLO",
        ]
        bundles = []
        for s in syms:
            try:
                bundles.append(loader.load_bundle(s, lookback_years=5, include_deep_info=False))
            except Exception:
                pass
        evolver = WinRateEvolutionEngine(config=config, db=db)
        champ = evolver.run_evolution_rounds(bundles=bundles, n_generations=3, progress_cb=lambda m: status.info(m))
        status.success(f"🎉 추가 특훈 완료! 현재 챔피언(제{champ.generation}세대) 승률: {champ.champion_win_rate_pct:.1f}%!")
        st.rerun()


# ---------------------------------------------------------------------------
# 탭 6: 🎁 10만 원 실전 바로 시작 자료함 (누적 데이터 통합 다운로드 & 행동지침)
# ---------------------------------------------------------------------------
def render_starter_pack_tab(db: PaperTradingDB) -> None:
    st.markdown("### 🎁 나중에 실전 시작할 때 바로 꺼내 쓰는 '10만 원 시작 자료함 & 누적 데이터'")
    st.markdown(
        """
        <div class="guide-strip">
            💡 <b>언제든 마음먹었을 때 바로 시작하세요!</b><br>
            AI가 뒤에서 24시간 시뮬레이션 세대와 실시간 주가 데이터를 차곡차곡 모으면서, <b>오늘 당장 내 돈 10만 원(또는 원하는 금액)으로 시작한다면 어떤 주식을 몇천 원어치 사고 얼마에 팔아야 하는지</b> 자동으로 정리해 두는 자료함입니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_bud, c_btn = st.columns([1.4, 2.2])
    with c_bud:
        my_budget = st.number_input(
            "실전 시작할 내 예산 입력 (원)",
            min_value=30_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
        )
    pack = export_starter_pack_files(budget_krw=float(my_budget), db=db)
    s = pack["stats"]
    plan_df: pd.DataFrame = pack["plan_df"]

    with c_btn:
        st.write("")
        st.caption(
            f"📌 최신 자동 정리 시각: **{s['generated_at']}** · "
            f"폴더 위치: `data/starter_pack/ai_accumulated_data_pack.zip`"
        )

    # 누적 데이터 & 오늘 10만 원 배분 요약 4칸
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🧬 지금까지 쌓인 AI 연습량</div>
            <div class="kpi-value">{s['total_generations']}세대</div>
            <div class="kpi-sub pos-text">누적 {s['total_simulated_trades']:,}번 매매 · 챔피언 승률 {s['champion_win_rate_pct']}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    p2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">⏱️ 쌓인 실시간 시세 · 예측 기록</div>
            <div class="kpi-value">{s['live_position_ticks']:,}건</div>
            <div class="kpi-sub neu-text">5일 만기 채점 추적 {s['predictions_logged']}건 누적 중</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    p3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🏦 통장에 그대로 남겨둘 비상금</div>
            <div class="kpi-value">{s['cash_reserve_krw']:,}원</div>
            <div class="kpi-sub neu-text">{s['weather_icon']} 폴리마켓 날씨 맞춰 {s['cash_reserve_pct']:.0f}% 현금 보호</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    p4.markdown(
        f"""
        <div class="kpi-card" style="border-color:rgba(16,185,129,0.5);">
            <div class="kpi-label">🛒 소수점으로 나눠 살 주식 합계</div>
            <div class="kpi-value"><span class="pos-text">{s['invest_krw']:,}원</span></div>
            <div class="kpi-sub pos-text">챔피언 승률 통과 우량주 {len(plan_df)}종목 분산</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 원클릭 다운로드 버튼 3종
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "📦 전체 누적 데이터 + 시작 가이드 ZIP 다운로드",
            data=pack["zip_bytes"],
            file_name=f"AI_누적데이터_{int(my_budget)//10000}만원_실전패키지.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "📄 초보자 따라 하기 가이드북 (.md) 다운로드",
            data=pack["md_text"].encode("utf-8"),
            file_name=f"{int(my_budget)//10000}만원_실전시작_가이드북.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "📊 종목별 금액·손절·익절 표 (.csv) 다운로드",
            data=plan_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"{int(my_budget)//10000}만원_실전장바구니.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(f"#### 📱 스마트폰 증권앱(토스·카카오페이·미니스탁) 보고 그대로 따라 하기 ({int(my_budget):,}원 기준)")
    if plan_df.empty:
        st.info("지금은 챔피언 안전 기준을 통과한 매수 종목이 없어 전액 통장 현금 보관을 추천해요!")
    else:
        cols = st.columns(min(len(plan_df), 3))
        for idx, (_, r) in enumerate(plan_df.iterrows()):
            card_h = f"""
            <div class="stock-card" style="border-color:rgba(56,189,248,0.45);">
                <div class="card-top">
                    <div>
                        <div class="card-title">{html.escape(str(r['순위']))}: {html.escape(str(r['종목명']))} ({html.escape(str(r['티커(검색코드)']))})</div>
                        <div class="card-sub">{html.escape(str(r['주문방법']))} · 실전승률 <b style="color:#6ee7b7;">{r['훈련된_실전승률(%)']}%</b></div>
                    </div>
                    <div><span class="badge badge-buy">{int(r['매수할_금액(원)']):,}원 사기</span></div>
                </div>
                <div class="easy-box" style="margin-top:8px;line-height:1.7;">
                    • 💵 <b>주문할 금액:</b> <b style="color:#fef08a;font-size:1.02rem;">{int(r['매수할_금액(원)']):,}원</b> (기준가 {html.escape(str(r['기준_현재가']))})<br>
                    • 🛡️ <b>안전 브레이크(손절):</b> <b style="color:#fda4af;">{html.escape(str(r['안전브레이크_손절가']))}</b><br>
                    • 🔒 <b>본전 보호선:</b> {html.escape(str(r['본전잠금_기준가']))}<br>
                    • 🎯 <b>목표 수익가(익절):</b> <b style="color:#6ee7b7;">{html.escape(str(r['목표_익절가']))}</b>
                </div>
            </div>
            """
            with cols[idx % len(cols)]:
                st.markdown(card_h, unsafe_allow_html=True)

        st.dataframe(plan_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 메인 앱 엔트리
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # 사이드바: 초보자는 건드릴 필요 없도록 안내 후 고급 설정 보관
    with st.sidebar:
        st.header("⚙️ AI 세부 설정 (건드리지 않아도 돼요!)")
        st.caption("기본값으로 두시면 AI가 알아서 폴리마켓·큰손·뉴스·회사 체력을 모두 켜고 작동합니다.")
        horizon = st.selectbox("며칠 뒤를 예측할까요?", [1, 5, 10, 20], index=1, format_func=lambda d: f"{d}거래일 뒤")
        use_alt = st.checkbox("🔮 폴리마켓·레딧·큰손 수급 반영", value=True)
        use_fund = st.checkbox("💪 회사 체력(재무·실적) 반영", value=True)
        use_news = st.checkbox("📰 실시간 뉴스 분위기 반영", value=True)
        use_cs = st.checkbox("🧠 20개 대표주식 비교 알파 결합", value=True)
        use_dl = st.checkbox("🤖 딥러닝 신경망 포함", value=True)
        use_cache = st.checkbox("⚡ 빠른 캐시 사용", value=True)

    config = PredictorConfig(
        horizon=horizon,
        use_deep_learning=use_dl,
        use_cross_sectional=use_cs,
        use_fundamental_overlay=use_fund,
        use_news_overlay=use_news,
        use_alt_overlay=use_alt,
        use_cache=use_cache,
    )

    paper_db = PaperTradingDB()
    trader = LivePaperTrader(db=paper_db)
    live_summary = trader.get_portfolio_summary()
    sim_report = paper_db.load_latest_simulation()
    pulse = _get_cached_macro_pulse()

    if "screen_df" not in st.session_state:
        saved_df, saved_name = _load_latest_saved_screening()
        if saved_df is not None and not saved_df.empty:
            st.session_state["screen_df"] = saved_df
            st.session_state["screen_source"] = f"최근 저장된 추천 목록 ({saved_name})"
    screen_df = st.session_state.get("screen_df")
    screen_source = st.session_state.get("screen_source", "")

    # 1) 상단 친근한 히어로 배너
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-title">🌱 주식 왕초보를 위한 AI 자동 모의투자 & 폴리마켓 상황판</div>
            <div class="hero-sub">
                어려운 주식 용어는 몰라도 괜찮아요! 전 세계 사람들이 실제 돈을 걸고 맞히는 <b>폴리마켓(예측시장)</b>, <b>임원·외국인 큰손 매수</b>, <b>미국 레딧 입소문</b>까지 AI가 읽고 내 <b>1억 원 모의지갑</b>을 실시간으로 대신 관리해 줍니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2) 오늘의 폴리마켓 시장 날씨 배너 (투자 비중에 직접 영향!)
    st.markdown(
        f"""
        <div class="weather-banner">
            <div class="weather-left">
                <div class="weather-emoji">{pulse.weather_icon}</div>
                <div>
                    <div class="weather-title">오늘의 예측시장(폴리마켓) 투자 날씨: {html.escape(pulse.weather_title_ko)}</div>
                    <div class="weather-desc">{html.escape(pulse.weather_summary_ko)}</div>
                </div>
            </div>
            <div>
                <span class="badge badge-purple">🌡️ 시장 심리 {pulse.fear_greed_score:.0f}도</span>
                <span class="badge badge-blue">🏦 비상금 최소 {int((1.0 - pulse.recommended_invest_cap)*100)}% 보호</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3) 최상단 왕초보 4칸 지갑 요약 카드
    total_eq = live_summary["total_equity_krw"]
    diff_from_init = total_eq - DEFAULT_INITIAL_CASH_KRW
    ret_cls = "pos-text" if diff_from_init >= 0 else "neg-text"
    cash_krw = live_summary["cash_krw"]
    cash_ratio = (cash_krw / max(total_eq, 1.0)) * 100.0

    orders_all = paper_db.get_orders_df(limit=200)
    win_sells = 0
    if not orders_all.empty and "realized_pnl_krw" in orders_all.columns:
        win_sells = int((orders_all["realized_pnl_krw"].fillna(0) > 0).sum())

    bull_bets = sum(1 for b in pulse.polymarket_bets if b.market_impact == "호재")
    bear_bets = sum(1 for b in pulse.polymarket_bets if b.market_impact == "악재")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">👛 지금 내 모의지갑 총액 (다 팔면 받는 돈)</div>
            <div class="kpi-value">{fmt_manwon(total_eq)}</div>
            <div class="kpi-sub {ret_cls}">처음 1억 원 시작 → {fmt_krw(diff_from_init, signed=True)} ({live_summary['total_return_pct']:+.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🏦 아껴둔 안전한 현금 비상금</div>
            <div class="kpi-value">{fmt_manwon(cash_krw)}</div>
            <div class="kpi-sub neu-text">지갑의 <b>{cash_ratio:.0f}%</b>는 현금 보관 · 주식 {live_summary['n_positions']}종목 보유 중</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🔮 돈 걸린 예측시장(폴리마켓) 신호</div>
            <div class="kpi-value"><span class="pos-text">호재 {bull_bets}건</span> · <span class="neg-text">주의 {bear_bets}건</span></div>
            <div class="kpi-sub neu-text">금리·관세·AI 실제 돈 걸린 확률이 AI 투자에 반영됨</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    champ_now = load_champion_artifact()
    champ_wr_txt = f"승률 {champ_now.champion_win_rate_pct:.1f}%" if champ_now else f"{win_sells}번 성공!"
    champ_sub_txt = (
        f"초보 {champ_now.baseline_win_rate_pct:.0f}%→{champ_now.champion_win_rate_pct:.0f}% 진화 · 자동익절 {win_sells}회"
        if champ_now
        else "내가 안 봐도 오르면 알아서 팔아 현금으로 챙김"
    )
    k4.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">🧬 자가진화 AI 승률 & 자동 수익 확정</div>
            <div class="kpi-value"><span class="pos-text">{champ_wr_txt}</span></div>
            <div class="kpi-sub pos-text">{champ_sub_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # 4) 초보자 눈높이 6대 메인 탭
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎁 10만 원 실전 시작 자료함",
        "👛 내 지갑 & 실시간 오르내림",
        "🧬 AI 혼자 연습하며 승률 올리기",
        "🔮 폴리마켓 & 큰손·개미 정보실",
        "🛒 AI가 골라준 살 만한 주식",
        "🔍 궁금한 주식 쉽게 물어보기",
    ])

    with tab1:
        render_starter_pack_tab(paper_db)
    with tab2:
        render_wallet_tab(paper_db, trader, live_summary, sim_report, screen_df, pulse)
    with tab3:
        render_evolution_lab_tab(config, paper_db)
    with tab4:
        render_polymarket_tab(pulse, live_summary)
    with tab5:
        render_easy_store_tab(screen_df, screen_source, config, trader)
    with tab6:
        render_easy_ask_tab(config, trader)

    st.caption("🌱 안내: 이 화면은 실제 내 돈이 들지 않는 안전한 1억 원 가상 모의투자 연습장입니다. AI와 폴리마켓 신호는 투자 공부를 돕는 참고 길잡이예요!")


if __name__ == "__main__":
    main()
