"""예측시장(Polymarket · Kalshi) + 소셜 민심(Reddit ApeWisdom) + 내부자/외인·기관 큰손 수급 통합 모듈.

주식 초보자도 이해할 수 있는 쉬운 한국어 해설과 함께, 실제 투자 예측 확률·목표가·현금 비중에
반영되는 정량 시그널(-1.0 ~ +1.0)을 산출합니다.

수집 소스 (모두 무료 공개 엔드포인트, API 키 불필요):
1. Polymarket Gamma API: 전 세계 참가자들이 실제 돈을 걸고 베팅하는 금리·침체·관세·AI·빅테크 확률
2. Kalshi Public Trade API: 미국 CFTC 공인 이벤트 거래소의 경제·증시 계약 확률
3. ApeWisdom API: Reddit(r/wallstreetbets, r/stocks 등) 실시간 종목 언급 순위 및 24시간 급증률
4. 내부자 & 큰손 수급:
   - 해외(US): yfinance SEC Form 4 임원/내부자 자사주 순매수 동향
   - 국내(KR): 네이버 금융 최근 5거래일 외국인·기관 순매수 동향
5. 시장 심리 온도계 (0~100): CNN Fear & Greed API 또는 VIX·지수 이격도 기반 실시간 심리 지수
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import html
import json
import pickle
import re
import time
from typing import Any

import numpy as np
import requests

from ..config import CACHE_DIR

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
KALSHI_MARKETS_URLS = [
    "https://api.elections.kalshi.com/trade-api/v2/markets",
    "https://external-api.kalshi.com/trade-api/v2/markets",
]
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# 폴리마켓 경제·증시 관련 필터 키워드
FINANCE_EVENT_KEYWORDS = [
    "fed", "rate", "interest", "fomc", "powell", "inflation", "cpi", "recession",
    "gdp", "tariff", "trade", "s&p", "nasdaq", "dow", "stock", "market",
    "ai", "openai", "nvidia", "apple", "tesla", "microsoft", "google", "meta",
    "amazon", "semiconductor", "chip", "nuclear", "china", "korea", "treasury",
    "oil", "gold", "bitcoin", "debt", "tax",
]

# 미국/한국 주요 종목별 폴리마켓 매칭 키워드
SYMBOL_POLY_KEYWORDS: dict[str, list[str]] = {
    "NVDA": ["nvidia", "ai", "chip", "semiconductor", "openai"],
    "AAPL": ["apple", "iphone", "tariff", "china"],
    "MSFT": ["microsoft", "openai", "ai"],
    "GOOGL": ["google", "alphabet", "antitrust", "ai"],
    "META": ["meta", "tiktok", "ai", "zuckerberg"],
    "AMZN": ["amazon", "tariff", "retail", "recession"],
    "TSLA": ["tesla", "elon", "musk", "ev", "robotaxi", "tariff"],
    "MU": ["semiconductor", "chip", "ai", "tariff"],
    "AVGO": ["semiconductor", "chip", "ai"],
    "TSM": ["taiwan", "tsmc", "chip", "semiconductor", "tariff"],
    "OKLO": ["nuclear", "energy", "ai", "power"],
    "COST": ["recession", "inflation", "consumer", "tariff"],
    "V": ["recession", "consumer", "fed", "rate"],
    "LLY": ["fda", "drug", "health", "pharma"],
    "005930": ["semiconductor", "chip", "ai", "tariff", "korea"],
    "000660": ["semiconductor", "chip", "ai", "tariff", "korea"],
    "105560": ["fed", "rate", "korea", "recession"],
    "207940": ["fda", "pharma", "tariff", "rate"],
    "329180": ["trade", "tariff", "navy", "korea", "oil"],
}


@dataclass
class PredictionMarketBet:
    """폴리마켓/칼시에서 가져온 개별 돈 걸린 예측 항목."""

    source: str                # "Polymarket" | "Kalshi"
    title: str                 # 원문 제목
    question: str              # 세부 질문
    yes_prob_pct: float        # Yes 확률 (0~100%)
    volume_usd: float          # 거래대금 ($)
    topic_ko: str              # 초보자용 한글 주제 태그 (예: "미국 금리", "경기침체 걱정")
    easy_explain_ko: str       # 초보자용 쉬운 1줄 해석
    market_impact: str         # "호재" | "악재" | "중립"
    impact_score: float        # -1.0 ~ +1.0 (주식시장 전반에 미치는 방향성)


@dataclass
class SocialBuzzItem:
    """레딧(WallStreetBets/Stocks) 실시간 인기 급상승 종목."""

    rank: int
    ticker: str
    name: str
    mentions: int
    mentions_24h_ago: int
    mention_change_pct: float
    upvotes: int
    mood_ko: str               # 초보자용 화제성 설명


@dataclass
class MacroPredictionPulse:
    """시장 전체 예측시장(폴리마켓·칼시) + 심리 온도계 + 소셜 트렌드 종합 결과."""

    macro_signal: float = 0.0                  # -1.0(강한 위험회피) ~ +1.0(강한 위험선호)
    fear_greed_score: float = 50.0             # 0(극도의 공포) ~ 100(극도의 탐욕)
    fear_greed_label_ko: str = "보통 (평온함) 🙂"
    weather_icon: str = "⛅"
    weather_title_ko: str = "구름 조금 — 절반은 투자하고 비상금도 챙기는 날씨"
    weather_summary_ko: str = ""
    recommended_invest_cap: float = 0.72       # 포트폴리오 최대 주식 비중 권고치 (0.45 ~ 0.85)
    polymarket_bets: list[PredictionMarketBet] = field(default_factory=list)
    social_trending: list[SocialBuzzItem] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class StockAltSignal:
    """개별 종목의 예측시장·소셜 민심·내부자/외인기관 큰손 수급 종합 시그널."""

    combined_signal: float = 0.0       # -1.0 ~ +1.0 (예측 모델 및 위원회 반영치)
    insider_or_bigmoney_score: float = 0.0
    insider_summary_ko: str = "큰손 매매 특이사항 없음"
    social_score: float = 0.0
    social_summary_ko: str = "커뮤니티 언급 평소 수준"
    poly_score: float = 0.0
    poly_summary_ko: str = "예측시장 중립"
    related_bets: list[PredictionMarketBet] = field(default_factory=list)
    easy_bullets_ko: list[str] = field(default_factory=list)


def _interpret_bet_for_beginners(title: str, question: str, yes_pct: float) -> tuple[str, str, str, float]:
    """폴리마켓/칼시 영어 베팅 질문과 확률을 초보자용 한국어 설명 및 주식 영향 점수로 변환."""
    text = f"{title} {question}".lower()

    # 1) 경기침체 (Recession)
    if "recession" in text:
        topic = "미국 경기침체 가능성"
        if yes_pct >= 45:
            return (
                topic,
                f"사람들이 돈 걸고 본 경기침체 확률이 {yes_pct:.0f}%로 꽤 높아요. 안전하게 현금 비상금을 아껴야 해요.",
                "악재",
                -0.65 * (yes_pct / 100.0),
            )
        return (
            topic,
            f"경기침체가 올 확률을 {yes_pct:.0f}%로 낮게 보고 있어요! 큰 경제 위기 걱정은 덜한 편이에요.",
            "호재",
            0.35 * ((100.0 - yes_pct) / 100.0),
        )

    # 2) 연준 금리 인하 / 인상 (Fed Rate Cut / Hike)
    if any(k in text for k in ["fed", "fomc", "interest rate", "rate cut", "bps"]):
        topic = "미국 기준금리 전망"
        if any(k in text for k in ["cut", "decrease", "lower"]):
            if yes_pct >= 50:
                return (
                    topic,
                    f"미국이 이자(금리)를 내릴 확률에 {yes_pct:.0f}%가 베팅 중이에요. 금리가 내려가면 주식엔 반가운 소식이에요!",
                    "호재",
                    0.55 * (yes_pct / 100.0),
                )
            return (
                topic,
                f"당장 금리를 내릴 확률이 {yes_pct:.0f}%로 낮아졌어요. 이자가 높게 유지되면 주식엔 약간 부담이에요.",
                "악재",
                -0.25,
            )
        if any(k in text for k in ["hike", "increase", "raise"]):
            if yes_pct >= 35:
                return (
                    topic,
                    f"금리를 오히려 올릴 수도 있다는 베팅이 {yes_pct:.0f}%예요. 기술주에는 조심스러운 신호예요.",
                    "악재",
                    -0.60 * (yes_pct / 100.0),
                )
            return (topic, f"금리 인상 가능성은 {yes_pct:.0f}%로 매우 낮게 보고 있어요.", "호재", 0.20)
        if "no change" in text or "unchanged" in text or "hold" in text:
            return (
                topic,
                f"이번엔 금리를 그대로 둘 확률을 {yes_pct:.0f}%로 보고 있어요.",
                "중립",
                0.0,
            )

    # 3) 관세 / 무역 갈등 (Tariff / Trade War)
    if "tariff" in text or "trade war" in text or "export ban" in text:
        topic = "관세 · 무역 규제"
        if yes_pct >= 50:
            return (
                topic,
                f"관세나 수출 규제가 생길 확률을 {yes_pct:.0f}%로 높게 봐요. 반도체·수출 기업들은 가격 흔들림을 조심해야 해요.",
                "악재",
                -0.45 * (yes_pct / 100.0),
            )
        return (topic, f"심한 관세 갈등 확률은 {yes_pct:.0f}%로 낮게 베팅되고 있어요.", "호재", 0.20)

    # 4) 인플레이션 / 물가 (Inflation / CPI)
    if "inflation" in text or "cpi" in text:
        topic = "물가(인플레이션) 전망"
        if any(w in text for w in ["above", "higher", "exceed", "rise"]) and yes_pct >= 55:
            return (topic, f"물가가 예상보다 높을 확률이 {yes_pct:.0f}%예요. 물가가 오르면 금리가 안 내려가서 주식이 쉬어갈 수 있어요.", "악재", -0.35)
        return (topic, f"물가 예측 베팅 현재 확률은 {yes_pct:.0f}% 수준이에요.", "중립", 0.05)

    # 5) 빅테크 / AI / 시가총액 1위 경쟁
    if any(k in text for k in ["nvidia", "apple", "microsoft", "tesla", "google", "meta", "openai", "ai"]):
        topic = "AI · 빅테크 기대감"
        if any(
            bad in text
            for bad in [
                "ban", "lawsuit", "breakup", "crash", "drop", "miss",
                "resignation", "resign", "removal", "step down", "fired", "investigat",
            ]
        ):
            imp = "악재" if yes_pct >= 45 else "호재"
            sc = -0.35 if yes_pct >= 45 else 0.12
            msg = (
                f"빅테크 경영진 교체·규제·하락 리스크 확률이 {yes_pct:.0f}%로 높은 편이라 주의해요."
                if yes_pct >= 45
                else f"빅테크 나쁜 이슈가 터질 확률은 {yes_pct:.0f}%로 낮게 보고 있어 안심이에요."
            )
            return (topic, msg, imp, sc)
        imp = "호재" if yes_pct >= 50 else "중립"
        sc = 0.30 if yes_pct >= 50 else -0.05
        return (topic, f"AI·빅테크 성장 목표 달성 가능성에 실제 돈 {yes_pct:.0f}% 확률이 걸려 있어요!", imp, sc)

    # 6) 비트코인 / 위험자산 투자심리
    if "bitcoin" in text or "btc" in text:
        topic = "위험자산(비트코인) 심리"
        if any(bad in text for bad in ["dip", "drop", "below", "crash"]):
            imp = "악재" if yes_pct >= 50 else "호재"
            sc = -0.20 if yes_pct >= 50 else 0.15
            return (
                topic,
                f"위험자산 조정(하락) 베팅 확률이 {yes_pct:.0f}%예요. {'시장 심리가 위축된 편이에요.' if yes_pct >= 50 else '큰 폭락 가능성은 낮게 보고 있어요.'}",
                imp,
                sc,
            )
        imp = "호재" if yes_pct >= 50 else "중립"
        sc = 0.20 if yes_pct >= 50 else 0.0
        return (topic, f"위험자산 상승 목표 달성 베팅 확률이 {yes_pct:.0f}%를 기록 중이에요.", imp, sc)

    # 7) 원전 / 에너지
    if "nuclear" in text or "energy" in text:
        return ("원전 · 에너지 정책", f"에너지·원전 관련 이벤트 실현 확률이 {yes_pct:.0f}%로 베팅 중이에요.", "호재" if yes_pct >= 50 else "중립", 0.20 if yes_pct >= 50 else 0.0)

    return ("글로벌 경제 이슈", f"예측시장 참가자들은 이 이벤트가 일어날 확률을 {yes_pct:.0f}%로 보고 있어요.", "중립", 0.0)


class AltMarketIntelligenceEngine:
    """폴리마켓 · 칼시 · 레딧(ApeWisdom) · 내부자/외인기관 큰손 수급 통합 수집기."""

    def __init__(self, cache_ttl_minutes: int = 20) -> None:
        self.cache_ttl_minutes = cache_ttl_minutes
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_get(self, key: str) -> Any | None:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        p = CACHE_DIR / f"alt_{digest}.pkl"
        if not p.exists():
            return None
        if (time.time() - p.stat().st_mtime) / 60.0 > self.cache_ttl_minutes:
            return None
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _cache_set(self, key: str, val: Any) -> None:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        p = CACHE_DIR / f"alt_{digest}.pkl"
        try:
            with p.open("wb") as f:
                pickle.dump(val, f)
        except Exception:
            pass

    def get_macro_pulse(self, vix_hint: float = 20.0, use_cache: bool = True) -> MacroPredictionPulse:
        """폴리마켓 + 칼시 + 공포탐욕 온도계 + 레딧 인기주를 통합해 현재 시장 날씨 산출."""
        cache_key = "macro_pulse_v1"
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        bets: list[PredictionMarketBet] = []
        bets.extend(self._fetch_polymarket_finance_bets(limit=14))
        if len(bets) < 8:
            bets.extend(self._fetch_kalshi_finance_bets(limit=6))

        social_list = self._fetch_apewisdom_top_stocks(limit=15)
        fng_score, fng_label = self._fetch_fear_and_greed(vix_hint=vix_hint)

        # 예측시장 가중 신호 합산 (거래액 로그 가중)
        if bets:
            weights = np.array([np.log1p(max(b.volume_usd, 1000.0)) for b in bets], dtype=float)
            scores = np.array([b.impact_score for b in bets], dtype=float)
            poly_macro = float(np.sum(weights * scores) / (np.sum(weights) + 1e-9))
        else:
            poly_macro = 0.0

        # 공포·탐욕 온도계 결합 (극단적 공포/과열 시 신중 모드)
        fng_centered = (fng_score - 50.0) / 50.0  # -1 ~ +1
        macro_sig = float(np.clip(0.65 * poly_macro + 0.35 * fng_centered * 0.5, -1.0, 1.0))

        if macro_sig >= 0.15 and 38 <= fng_score <= 78:
            icon = "☀️"
            title = "맑음 — 예측시장 호재가 우세해 우량주 투자하기 좋은 날씨예요!"
            cap = 0.80
        elif macro_sig <= -0.18 or fng_score < 28:
            icon = "🌧️"
            title = "비바람 주의 — 큰돈 걸린 예측시장에서 걱정거리가 많아 현금 비상금을 넉넉히 남겨요!"
            cap = 0.55
        else:
            icon = "⛅"
            title = "구름 조금 — 좋은 소식과 걱정거리가 섞여 있어 튼튼한 회사만 골라 담는 날씨예요."
            cap = 0.70

        bull_bets = sum(1 for b in bets if b.market_impact == "호재")
        bear_bets = sum(1 for b in bets if b.market_impact == "악재")
        summary_ko = (
            f"전 세계 예측시장(폴리마켓 등) 주요 경제 베팅 {len(bets)}건을 분석한 결과 "
            f"주식 호재 {bull_bets}건 · 주의(악재) {bear_bets}건이며, 시장 심리 온도는 {fng_score:.0f}도({fng_label})입니다. "
            f"이에 맞춰 내 지갑의 최소 {int((1.0 - cap) * 100)}%는 안전한 비상금(현금)으로 자동 보호합니다."
        )

        pulse = MacroPredictionPulse(
            macro_signal=macro_sig,
            fear_greed_score=fng_score,
            fear_greed_label_ko=fng_label,
            weather_icon=icon,
            weather_title_ko=title,
            weather_summary_ko=summary_ko,
            recommended_invest_cap=cap,
            polymarket_bets=bets,
            social_trending=social_list,
        )
        self._cache_set(cache_key, pulse)
        return pulse

    def get_stock_alt_signal(
        self,
        market: str,
        code: str,
        name: str,
        yf_ticker: str,
        macro_pulse: MacroPredictionPulse | None = None,
        use_cache: bool = True,
    ) -> StockAltSignal:
        """개별 종목에 대한 [폴리마켓 연관 베팅 + 레딧 민심 + 임원/외인기관 큰손 수급] 통합 시그널."""
        cache_key = f"stock_alt_v1_{market}_{code}"
        if use_cache:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        pulse = macro_pulse or self.get_macro_pulse(use_cache=use_cache)
        bullets: list[str] = []

        # 1) 큰손 수급: 미국=회사 임원(내부자) 매매, 한국=외국인·기관 최근 5일 순매수
        if market == "KR":
            big_score, big_txt = self._fetch_kr_foreign_inst_trend(code)
        else:
            big_score, big_txt = self._fetch_us_insider_trend(yf_ticker)
        bullets.append(big_txt)

        # 2) 소셜 민심 (ApeWisdom Reddit 언급량 — 미국 종목 중심)
        soc_score = 0.0
        soc_txt = "커뮤니티 언급량 평소 수준 (조용히 움직이는 중)"
        if market == "US" and pulse.social_trending:
            hit = next((s for s in pulse.social_trending if s.ticker.upper() == code.upper()), None)
            if hit:
                if hit.mention_change_pct > 250.0:
                    soc_score = -0.15
                    soc_txt = f"🔥 미국 개미 커뮤니티 인기 {hit.rank}위 (언급 {hit.mention_change_pct:+.0f}% 폭증 — 너무 과열돼 고점 물림 조심!)"
                elif hit.mention_change_pct >= 15.0:
                    soc_score = 0.30
                    soc_txt = f"🗣️ 미국 레딧 인기 {hit.rank}위 ({hit.mentions}회 언급, 어제보다 {hit.mention_change_pct:+.0f}% 관심 증가!)"
                else:
                    soc_score = 0.12
                    soc_txt = f"🗣️ 미국 레딧 관심 상위 {hit.rank}위권 종목 ({hit.mentions}회 언급)"
                bullets.append(soc_txt)

        # 3) 폴리마켓 연관 이벤트 매칭 + 매크로 신호 결합
        kws = SYMBOL_POLY_KEYWORDS.get(code.upper(), [name.lower(), code.lower()])
        related: list[PredictionMarketBet] = []
        for bet in pulse.polymarket_bets:
            hay = f"{bet.title} {bet.question}".lower()
            if any(k in hay for k in kws):
                related.append(bet)
        related = related[:3]

        if related:
            rel_mean = float(np.mean([b.impact_score for b in related]))
            poly_score = float(np.clip(0.65 * rel_mean + 0.35 * pulse.macro_signal, -1.0, 1.0))
            top_b = related[0]
            poly_txt = f"🔮 폴리마켓 연관 베팅({top_b.topic_ko}): 확률 {top_b.yes_prob_pct:.0f}% ({top_b.market_impact})"
            bullets.append(poly_txt)
        else:
            poly_score = float(np.clip(pulse.macro_signal * 0.6, -0.6, 0.6))
            poly_txt = f"🔮 폴리마켓 거시경제 분위기 반영 ({pulse.weather_icon} {pulse.macro_signal:+.2f})"

        combined = float(np.clip(0.45 * big_score + 0.35 * poly_score + 0.20 * soc_score, -1.0, 1.0))
        res = StockAltSignal(
            combined_signal=combined,
            insider_or_bigmoney_score=big_score,
            insider_summary_ko=big_txt,
            social_score=soc_score,
            social_summary_ko=soc_txt,
            poly_score=poly_score,
            poly_summary_ko=poly_txt,
            related_bets=related,
            easy_bullets_ko=bullets,
        )
        self._cache_set(cache_key, res)
        return res

    def _fetch_polymarket_finance_bets(self, limit: int = 14) -> list[PredictionMarketBet]:
        bets: list[PredictionMarketBet] = []
        try:
            resp = requests.get(
                GAMMA_EVENTS_URL,
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "order": "volume24hr",
                    "ascending": "false",
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            if not resp.ok:
                return bets
            events = resp.json()
            if not isinstance(events, list):
                return bets

            seen_titles: set[str] = set()
            noise_words = [
                "nba", "nfl", "mlb", "ufc", "soccer", "champions league", "oscar", "grammy",
                "affair", "divorce", "dating", "tweet", "tweets", "prison", "jail", "gta", "pope",
            ]
            for ev in events:
                title = html.unescape(str(ev.get("title") or "").strip())
                if not title or title in seen_titles:
                    continue
                low_t = title.lower()
                # 정치/스포츠/가십 제외, 경제·금리·관세·기업·기술 이벤트만 엄선
                if not any(kw in low_t for kw in FINANCE_EVENT_KEYWORDS):
                    continue
                if any(sp in low_t for sp in noise_words):
                    continue

                markets = ev.get("markets") or []
                best_m = None
                best_vol = -1.0
                best_prob = 50.0
                for m in markets:
                    if m.get("closed") is True or m.get("active") is False:
                        continue
                    q_low = str(m.get("question") or "").lower()
                    if any(sp in q_low for sp in noise_words):
                        continue
                    outcomes_raw = m.get("outcomes", "[]")
                    prices_raw = m.get("outcomePrices", "[]")
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                    if not outcomes or not prices:
                        continue
                    p_pct = float(prices[0]) * 100.0
                    # 이미 사실상 확정/종료된(0~1.5% 또는 98.5~100%) 하위 마켓은 제외하고 현재 승부 중인 계약 선택
                    if p_pct < 1.5 or p_pct > 98.5:
                        continue
                    v = float(m.get("volume") or m.get("liquidity") or 0.0)
                    if v > best_vol:
                        best_vol = v
                        best_m = m
                        best_prob = p_pct
                if best_m is None:
                    continue

                yes_prob = best_prob
                question = html.unescape(str(best_m.get("question") or title).strip())
                vol_usd = float(ev.get("volume") or best_vol or 0.0)
                topic_ko, explain_ko, impact, imp_score = _interpret_bet_for_beginners(title, question, yes_prob)

                seen_titles.add(title)
                bets.append(
                    PredictionMarketBet(
                        source="Polymarket",
                        title=title,
                        question=question,
                        yes_prob_pct=round(yes_prob, 1),
                        volume_usd=round(vol_usd, 0),
                        topic_ko=topic_ko,
                        easy_explain_ko=explain_ko,
                        market_impact=impact,
                        impact_score=imp_score,
                    )
                )
                if len(bets) >= limit:
                    break
        except Exception:
            pass
        return bets

    def _fetch_kalshi_finance_bets(self, limit: int = 6) -> list[PredictionMarketBet]:
        bets: list[PredictionMarketBet] = []
        for base_url in KALSHI_MARKETS_URLS:
            try:
                resp = requests.get(
                    base_url,
                    params={"status": "open", "limit": 60},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=5,
                )
                if not resp.ok:
                    continue
                markets = resp.json().get("markets") or []
                for m in markets:
                    title = str(m.get("title") or m.get("subtitle") or "").strip()
                    if not title:
                        continue
                    if not any(kw in title.lower() for kw in FINANCE_EVENT_KEYWORDS):
                        continue
                    raw_p = m.get("last_price") or m.get("yes_bid") or 50
                    yes_prob = float(raw_p) if float(raw_p) > 1.0 else float(raw_p) * 100.0
                    vol = float(m.get("volume") or 0.0)
                    topic_ko, explain_ko, impact, imp_score = _interpret_bet_for_beginners(title, title, yes_prob)
                    bets.append(
                        PredictionMarketBet(
                            source="Kalshi",
                            title=title,
                            question=title,
                            yes_prob_pct=round(yes_prob, 1),
                            volume_usd=vol,
                            topic_ko=topic_ko,
                            easy_explain_ko=explain_ko,
                            market_impact=impact,
                            impact_score=imp_score,
                        )
                    )
                    if len(bets) >= limit:
                        return bets
                if bets:
                    break
            except Exception:
                continue
        return bets

    def _fetch_apewisdom_top_stocks(self, limit: int = 15) -> list[SocialBuzzItem]:
        items: list[SocialBuzzItem] = []
        try:
            resp = requests.get(APEWISDOM_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            if not resp.ok:
                return items
            results = resp.json().get("results") or []
            for r in results[:limit]:
                mentions = int(r.get("mentions") or 0)
                m_prev = int(r.get("mentions_24h_ago") or mentions or 1)
                chg = ((mentions - m_prev) / max(m_prev, 1)) * 100.0
                if chg >= 150:
                    mood = "🔥 오늘 개미 관심 대폭발 (과열 주의)"
                elif chg >= 20:
                    mood = "📈 어제보다 입소문 빠르게 늘어나는 중"
                elif chg <= -25:
                    mood = "📉 어제보다 언급 줄어드는 중"
                else:
                    mood = "💬 꾸준히 많이 이야기되는 인기 종목"
                items.append(
                    SocialBuzzItem(
                        rank=int(r.get("rank") or len(items) + 1),
                        ticker=html.unescape(str(r.get("ticker") or "")),
                        name=html.unescape(str(r.get("name") or "")),
                        mentions=mentions,
                        mentions_24h_ago=m_prev,
                        mention_change_pct=round(chg, 1),
                        upvotes=int(r.get("upvotes") or 0),
                        mood_ko=mood,
                    )
                )
        except Exception:
            pass
        return items

    def _fetch_fear_and_greed(self, vix_hint: float = 20.0) -> tuple[float, str]:
        score = None
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.cnn.com/markets/fear-and-greed",
            }
            resp = requests.get(CNN_FNG_URL, headers=headers, timeout=4)
            if resp.ok:
                fg = resp.json().get("fear_and_greed") or {}
                if fg.get("score") is not None:
                    score = float(fg["score"])
        except Exception:
            pass

        if score is None:
            # 실시간 VIX 기반 정밀 변환 (VIX 12 -> 80도 탐욕, VIX 20 -> 50도 보통, VIX 32 -> 20도 공포)
            score = float(np.clip(95.0 - (vix_hint - 10.0) * 3.2, 8.0, 92.0))

        if score <= 25:
            label = "극도의 공포 (다들 겁먹은 상태) 🥶"
        elif score <= 44:
            label = "공포 · 조심스러운 분위기 😟"
        elif score <= 56:
            label = "보통 · 차분한 눈치보기 🙂"
        elif score <= 74:
            label = "탐욕 · 자신감 붙는 상승 분위기 😊"
        else:
            label = "극도의 탐욕 (너무 들뜬 과열 주의) 🥵"
        return round(score, 1), label

    def _fetch_us_insider_trend(self, yf_ticker: str) -> tuple[float, str]:
        """미국 주식 SEC Form 4 임원·내부자 최근 매수/매도 동향 분석."""
        try:
            import yfinance as yf

            tk = yf.Ticker(yf_ticker)
            df = tk.insider_transactions
            if df is None or df.empty:
                return 0.0, "👔 회사 임원 매매: 최근 특별한 대량 매매 내역 없음"

            recent = df.head(12)
            text_col = "Text" if "Text" in recent.columns else ("Transaction" if "Transaction" in recent.columns else None)
            buys, sells = 0, 0
            if text_col:
                for val in recent[text_col].astype(str):
                    low = val.lower()
                    if "purchase" in low or "buy" in low:
                        buys += 1
                    elif "sale" in low or "sell" in low:
                        sells += 1

            if buys >= 2 and buys >= sells:
                return 0.45, f"👔 회사 임원 찬스: 최근 임원들이 직접 자기 돈으로 {buys}차례 자사주를 사들였어요! (강력 호재)"
            if buys == 1 and sells == 0:
                return 0.25, "👔 회사 임원 매수: 최근 내부 임원의 자사주 매입이 포착됐어요."
            if sells >= 5 and buys == 0:
                return -0.15, f"👔 회사 임원 동향: 최근 임원 차익실현 매도({sells}건)가 있어 무리한 추격은 조심해요."
            return 0.0, f"👔 회사 임원 동향: 평소 수준의 일상적 지분 변동 (매수 {buys} · 매도 {sells})"
        except Exception:
            return 0.0, "👔 회사 임원 매매: 특이사항 없음"

    def _fetch_kr_foreign_inst_trend(self, code: str) -> tuple[float, str]:
        """국내 주식 네이버 증권 모바일 API + 웹 테이블 폴백으로 외국인·기관 최근 5거래일 순매수 파싱."""
        inst_vals: list[int] = []
        frgn_vals: list[int] = []
        try:
            # 1차: 네이버 모바일 증권 공식 JSON 엔드포인트 (인코딩 문제 없음, 초고속)
            m_url = f"https://m.stock.naver.com/api/stock/{code}/trend"
            m_resp = requests.get(m_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
            if m_resp.ok:
                rows = m_resp.json()
                if isinstance(rows, list):
                    for item in rows[:5]:
                        f_raw = str(item.get("foreignerPureBuyQuant") or "0")
                        i_raw = str(item.get("organPureBuyQuant") or "0")
                        frgn_vals.append(int(re.sub(r"[^0-9\-]", "", f_raw) or "0"))
                        inst_vals.append(int(re.sub(r"[^0-9\-]", "", i_raw) or "0"))
        except Exception:
            pass

        if not inst_vals:
            try:
                url = f"https://finance.naver.com/item/frgn.naver?code={code}"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                if resp.ok:
                    from bs4 import BeautifulSoup

                    html_txt = resp.content.decode("euc-kr", errors="replace")
                    soup = BeautifulSoup(html_txt, "lxml")
                    for tr in soup.select("table.type2 tr"):
                        tds = tr.select("td")
                        if len(tds) < 7:
                            continue
                        date_txt = tds[0].get_text(strip=True)
                        if not re.match(r"\d{4}\.\d{2}\.\d{2}", date_txt):
                            continue
                        try:
                            i_v = int(re.sub(r"[^0-9\-]", "", tds[5].get_text(strip=True)) or "0")
                            f_v = int(re.sub(r"[^0-9\-]", "", tds[6].get_text(strip=True)) or "0")
                            inst_vals.append(i_v)
                            frgn_vals.append(f_v)
                        except ValueError:
                            continue
                        if len(inst_vals) >= 5:
                            break
            except Exception:
                pass

        if not inst_vals:
            return 0.0, "🏦 큰손(외국인·기관) 수급: 중립"

        i_sum = sum(inst_vals)
        f_sum = sum(frgn_vals)
        i_pos_days = sum(1 for x in inst_vals if x > 0)
        f_pos_days = sum(1 for x in frgn_vals if x > 0)

        if f_sum > 0 and i_sum > 0:
            return (
                0.48,
                f"🏦 큰손 쌍끌이 매수! 최근 5일 동안 외국인({f_pos_days}일 매수)과 기관({i_pos_days}일 매수)이 함께 사모으고 있어요!",
            )
        if f_sum > 0:
            return 0.25, f"🏦 외국인 큰손 매수 우위 (최근 5일 중 {f_pos_days}일 순매수 유입)"
        if i_sum > 0:
            return 0.22, f"🏦 국내 기관 큰손 매수 우위 (최근 5일 중 {i_pos_days}일 순매수 유입)"
        if f_sum < 0 and i_sum < 0:
            return -0.38, "🏦 큰손 동반 매도 주의: 최근 5일간 외국인과 기관이 둘 다 주식을 팔고 있어요."
        return 0.0, "🏦 큰손 수급: 외국인과 기관이 서로 사고팔며 팽팽히 맞서는 중이에요."
