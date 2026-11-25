"""다중 소스 실시간 이슈 · 뉴스 · 공시 수집 및 이벤트 기반 금융 감성 엔진.

특징:
1. 다중 소스 수집:
   - 국내(KR): 네이버 금융 종목뉴스 + Google News (KR) + yfinance 뉴스
   - 해외(US): Yahoo Finance 뉴스 + Google News (US)
2. 6대 주가 영향 이벤트 자동 분류 & 임팩트 가중:
   - 실적/어닝(×1.6), 대형 수주·신제품·자사주(×1.4), 증권사 투자의견(×1.2),
     유상증자·규제·소송·악재(×1.7), 업종/매크로(×1.0), 일반 뉴스(×0.8)
3. 시간 감쇠(Time Decay): 최근 24~48시간 이슈에 높은 가중치, 오래된 기사는 지수 감쇠
4. (선택) OPENAI_API_KEY + use_llm=True 시 LLM이 수집된 뉴스 묶음을 읽고 구조화된 감성/촉매 리포트 보강
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import re
from typing import Any

import numpy as np
import requests


# (카테고리, 이벤트 라벨, 임팩트 가중치, 긍정 키워드, 부정 키워드)
EVENT_RULES: list[tuple[str, str, float, list[str], list[str]]] = [
    (
        "earnings",
        "실적/어닝",
        1.6,
        [
            "어닝 서프라이즈", "호실적", "최대 실적", "사상 최대", "흑자전환", "흑자 전환",
            "영업이익 급증", "매출 급증", "실적 개선", "컨센서스 상회", "가이던스 상향",
            "beats estimates", "earnings beat", "record revenue", "record profit",
            "raises guidance", "raises outlook", "strong earnings", "profit surges", "beat expectations",
        ],
        [
            "어닝 쇼크", "실적 부진", "영업손실", "적자전환", "적자 전환", "적자 확대",
            "컨센서스 하회", "가이던스 하향", "매출 감소", "영업이익 급감",
            "misses estimates", "earnings miss", "cuts guidance", "lowers outlook",
            "weak earnings", "profit drops", "revenue falls", "disappointing results",
        ],
    ),
    (
        "catalyst",
        "수주/신제품/주주환원",
        1.4,
        [
            "대규모 수주", "공급계약", "수주", "자사주 매입", "자사주 소각", "배당 확대",
            "신제품", "신약 승인", "fda 승인", "임상 성공", "파트너십", "협업", "양산", "hbm",
            "contract win", "major deal", "buyback", "stock split", "dividend hike",
            "fda approval", "partnership", "launch", "breakthrough", "ai chip", "expansion",
        ],
        [
            "계약 해지", "수주 취소", "임상 실패", "임상 중단", "출시 지연", "양산 지연",
            "배당 축소", "무배당", "supply chain disruption", "delay", "halted", "canceled",
            "clinical trial fails", "dividend cut", "product recall",
        ],
    ),
    (
        "risk_event",
        "유상증자/규제/소송",
        1.7,
        [
            "규제 완화", "소송 승소", "합의 타결", "제재 해제", "무혐의",
            "antitrust cleared", "lawsuit dismissed", "regulatory approval", "settlement",
        ],
        [
            "유상증자", "전환사채", "cb 발행", "오버행", "블록딜", "횡령", "배임",
            "압수수색", "과징금", "반독점", "관세", "수출 규제", "제재", "소송", "리콜",
            "구조조정", "감원", "파업", "화재", "감사의견 거절",
            "share offering", "dilution", "secondary offering", "lawsuit", "antitrust",
            "investigation", "probe", "subpoena", "tariff", "sanction", "export ban",
            "fine", "recall", "layoffs", "sec probe", "fraud", "downturn",
        ],
    ),
    (
        "analyst",
        "증권사/목표가",
        1.25,
        [
            "목표가 상향", "목표주가 상향", "투자의견 상향", "매수 추천", "최선호주", "top pick",
            "upgrades", "price target raised", "raises price target", "initiates at buy",
            "outperform", "overweight", "strong buy",
        ],
        [
            "목표가 하향", "목표주가 하향", "투자의견 하향", "매도 리포트", "비중 축소",
            "downgrades", "price target cut", "lowers price target", "underperform",
            "underweight", "sell rating",
        ],
    ),
    (
        "general",
        "시장/일반 이슈",
        0.9,
        [
            "상승", "급등", "강세", "신고가", "돌파", "기대감", "반등", "호재", "성장", "확대", "매수세",
            "surge", "rally", "soar", "gain", "bullish", "jump", "rebound", "growth", "strong", "boost",
        ],
        [
            "하락", "급락", "약세", "신저가", "우려", "부진", "둔화", "악재", "폭락", "경고", "매도세",
            "drop", "plunge", "slump", "bearish", "fall", "decline", "weak", "warning", "crash", "selloff",
        ],
    ),
]


@dataclass
class NewsItemAnalysis:
    """개별 뉴스/이슈 기사의 분석 결과."""

    title: str
    publisher: str
    published: str
    source: str
    category: str          # 실적/어닝, 수주/신제품/주주환원, 유상증자/규제/소송 등
    sentiment: float       # -1.0 ~ +1.0
    impact_weight: float   # 이벤트 중요도 × 시간감쇠
    recency_days: float
    url: str = ""


@dataclass
class NewsIntelligenceReport:
    """종목의 종합 이슈 · 뉴스 감성 리포트."""

    sentiment_score: float            # -1.0 ~ +1.0 (가중 평균 감성)
    event_shock_score: float          # -1.0 ~ +1.0 (고임팩트 이벤트 편향)
    article_count: int
    positive_count: int
    negative_count: int
    neutral_count: int
    dominant_category: str
    key_catalysts: list[str] = field(default_factory=list)
    key_risks: list[str] = field(default_factory=list)
    items: list[NewsItemAnalysis] = field(default_factory=list)
    llm_summary: str = ""
    llm_used: bool = False


def _parse_age_days(pub_str: str) -> float:
    """다양한 날짜 포맷에서 현재 기준 경과 일수 추출 (파싱 실패 시 3.0일 기본값)."""
    if not pub_str:
        return 3.0
    now = datetime.now(timezone.utc)
    s = str(pub_str).strip()
    try:
        # Unix timestamp
        if re.fullmatch(r"\d{9,11}", s):
            dt = datetime.fromtimestamp(int(s), tz=timezone.utc)
            return max((now - dt).total_seconds() / 86400.0, 0.0)
        # ISO format
        if "T" in s or re.match(r"\d{4}-\d{2}-\d{2}", s):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max((now - dt).total_seconds() / 86400.0, 0.0)
        # 네이버 금융 날짜 포맷 (예: 2026.11.23 15:20)
        m_naver = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", s)
        if m_naver:
            dt = datetime(int(m_naver.group(1)), int(m_naver.group(2)), int(m_naver.group(3)), tzinfo=timezone.utc)
            return max((now - dt).total_seconds() / 86400.0, 0.0)
        # RFC 2822 (Google News RSS)
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((now - dt).total_seconds() / 86400.0, 0.0)
    except Exception:
        return 3.0


def _classify_and_score_headline(title: str) -> tuple[str, float, float]:
    """헤드라인에서 (이벤트 카테고리, 감성 점수 -1~+1, 이벤트 가중치) 추출."""
    low = title.lower()
    best_cat = "시장/일반 이슈"
    best_weight = 0.85
    weighted_pos = 0.0
    weighted_neg = 0.0

    for _, label, weight, pos_kws, neg_kws in EVENT_RULES:
        p_hit = sum(1 for kw in pos_kws if kw in low)
        n_hit = sum(1 for kw in neg_kws if kw in low)
        if p_hit + n_hit > 0:
            weighted_pos += p_hit * weight
            weighted_neg += n_hit * weight
            if weight >= best_weight:
                best_cat = label
                best_weight = weight

    total = weighted_pos + weighted_neg
    if total <= 0:
        return best_cat, 0.0, 0.6
    sent = float(np.clip((weighted_pos - weighted_neg) / (total + 0.35), -1.0, 1.0))
    return best_cat, sent, best_weight


class NewsIntelligenceEngine:
    """국내·해외 다중 소스 뉴스·이슈 수집 및 감성·이벤트 분석 엔진."""

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm

    def collect_and_analyze(
        self,
        market: str,
        code: str,
        name: str,
        yf_ticker: str,
        max_items: int = 16,
    ) -> NewsIntelligenceReport:
        raw_articles: list[dict[str, str]] = []

        if market == "KR":
            raw_articles.extend(self._fetch_naver_stock_news(code, limit=10))
            raw_articles.extend(self._fetch_google_news_rss(f"{name} 주가 OR 실적 OR 수주", "KR", limit=10))
            raw_articles.extend(self._fetch_yfinance_news(yf_ticker, limit=6))
        else:
            raw_articles.extend(self._fetch_yfinance_news(yf_ticker, limit=10))
            raw_articles.extend(self._fetch_google_news_rss(f"{code} stock earnings OR guidance OR analyst", "US", limit=10))

        # 중복 제목 제거
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for art in raw_articles:
            title = art.get("title", "").strip()
            norm = re.sub(r"\s+", " ", title.lower())[:55]
            if title and norm not in seen:
                seen.add(norm)
                deduped.append(art)
            if len(deduped) >= max_items:
                break

        return self.analyze_articles(deduped, symbol_label=f"{name}({code})")

    def analyze_articles(
        self, articles: list[dict[str, str]], symbol_label: str = ""
    ) -> NewsIntelligenceReport:
        """수집된 기사 목록을 이벤트 분류·시간감쇠 감성 점수로 종합 (오프라인 테스트도 지원)."""
        analyzed: list[NewsItemAnalysis] = []
        for art in articles:
            title = str(art.get("title", "")).strip()
            if not title:
                continue
            pub_str = str(art.get("published", ""))
            age_days = _parse_age_days(pub_str)
            # 시간 감쇠: 반감기 약 5일
            time_decay = float(np.exp(-0.14 * min(age_days, 30.0)))
            cat, sent, ev_weight = _classify_and_score_headline(title)
            impact = ev_weight * time_decay
            analyzed.append(
                NewsItemAnalysis(
                    title=title,
                    publisher=str(art.get("publisher", "")).strip(),
                    published=pub_str,
                    source=str(art.get("source", "news")),
                    category=cat,
                    sentiment=sent,
                    impact_weight=impact,
                    recency_days=age_days,
                    url=str(art.get("url", "")),
                )
            )

        if not analyzed:
            return NewsIntelligenceReport(
                sentiment_score=0.0,
                event_shock_score=0.0,
                article_count=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                dominant_category="이슈 없음",
            )

        # 시간·이벤트 가중 감성 평균
        w_arr = np.array([a.impact_weight for a in analyzed], dtype=float)
        s_arr = np.array([a.sentiment for a in analyzed], dtype=float)
        sentiment_score = float(np.clip(np.sum(w_arr * s_arr) / (np.sum(w_arr) + 1e-9), -1.0, 1.0))

        # 고임팩트 이벤트(실적/수주/규제 등) 쇼크 점수
        high_imp = [a for a in analyzed if a.category != "시장/일반 이슈" and abs(a.sentiment) > 0.1]
        if high_imp:
            hw = np.array([a.impact_weight for a in high_imp], dtype=float)
            hs = np.array([a.sentiment for a in high_imp], dtype=float)
            event_shock = float(np.clip(np.sum(hw * hs) / (np.sum(hw) + 1e-9), -1.0, 1.0))
        else:
            event_shock = sentiment_score * 0.5

        pos_cnt = sum(1 for a in analyzed if a.sentiment >= 0.15)
        neg_cnt = sum(1 for a in analyzed if a.sentiment <= -0.15)
        neu_cnt = len(analyzed) - pos_cnt - neg_cnt

        # 주요 카테고리 집계
        cat_weights: dict[str, float] = {}
        for a in analyzed:
            cat_weights[a.category] = cat_weights.get(a.category, 0.0) + a.impact_weight
        dominant_cat = max(cat_weights.items(), key=lambda kv: kv[1])[0]

        key_catalysts = [
            f"[{a.category}] {a.title}"
            for a in sorted(analyzed, key=lambda x: x.sentiment * x.impact_weight, reverse=True)
            if a.sentiment >= 0.2
        ][:4]
        key_risks = [
            f"[{a.category}] {a.title}"
            for a in sorted(analyzed, key=lambda x: x.sentiment * x.impact_weight)
            if a.sentiment <= -0.2
        ][:4]

        # 최신·고임팩트 순으로 정렬
        analyzed.sort(key=lambda x: abs(x.sentiment) * x.impact_weight + 0.2 / (x.recency_days + 1.0), reverse=True)

        report = NewsIntelligenceReport(
            sentiment_score=sentiment_score,
            event_shock_score=event_shock,
            article_count=len(analyzed),
            positive_count=pos_cnt,
            negative_count=neg_cnt,
            neutral_count=neu_cnt,
            dominant_category=dominant_cat,
            key_catalysts=key_catalysts,
            key_risks=key_risks,
            items=analyzed,
        )

        # 선택적 LLM 정밀 이슈 분석 (OPENAI_API_KEY + use_llm=True)
        if self.use_llm and os.getenv("OPENAI_API_KEY"):
            self._enhance_with_llm(report, symbol_label)

        return report

    def _enhance_with_llm(self, report: NewsIntelligenceReport, symbol_label: str) -> None:
        try:
            from openai import OpenAI

            client = OpenAI()
            headlines = "\n".join(
                f"- ({a.category}, {a.recency_days:.0f}일전) {a.title}" for a in report.items[:12]
            )
            prompt = (
                f"당신은 전문 주식 이벤트 애널리스트입니다. '{symbol_label}'의 최신 뉴스 헤드라인을 분석하고 "
                f"반드시 JSON으로만 응답하세요. 키: sentiment(-1.0~1.0 숫자), event_shock(-1.0~1.0 숫자), "
                f"summary(한국어 2문장 핵심 이슈 요약), catalysts(핵심 호재 리스트 최대 3개), risks(핵심 악재 리스트 최대 3개).\n\n"
                f"{headlines}"
            )
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=350,
            )
            parsed = json.loads(resp.choices[0].message.content)
            llm_sent = float(np.clip(float(parsed.get("sentiment", report.sentiment_score)), -1.0, 1.0))
            llm_shock = float(np.clip(float(parsed.get("event_shock", report.event_shock_score)), -1.0, 1.0))
            # 규칙 점수와 LLM 점수를 3:7로 융합
            report.sentiment_score = float(0.3 * report.sentiment_score + 0.7 * llm_sent)
            report.event_shock_score = float(0.3 * report.event_shock_score + 0.7 * llm_shock)
            if parsed.get("summary"):
                report.llm_summary = str(parsed["summary"])
            if isinstance(parsed.get("catalysts"), list) and parsed["catalysts"]:
                report.key_catalysts = [str(c) for c in parsed["catalysts"][:4]]
            if isinstance(parsed.get("risks"), list) and parsed["risks"]:
                report.key_risks = [str(r) for r in parsed["risks"][:4]]
            report.llm_used = True
        except Exception:
            pass

    def _fetch_naver_stock_news(self, code: str, limit: int = 10) -> list[dict[str, str]]:
        """네이버 금융 종목 뉴스 탭 스크래핑."""
        items: list[dict[str, str]] = []
        try:
            url = f"https://finance.naver.com/item/news_news.naver?code={code}&page=1"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": f"https://finance.naver.com/item/news.naver?code={code}",
            }
            resp = requests.get(url, headers=headers, timeout=5)
            if not resp.ok:
                return items
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "lxml")
            for tr in soup.select("table.type5 tbody tr"):
                title_a = tr.select_one("td.title a")
                if not title_a:
                    continue
                title = title_a.get_text(strip=True)
                href = title_a.get("href", "")
                info_td = tr.select_one("td.info")
                date_td = tr.select_one("td.date")
                items.append(
                    {
                        "title": title,
                        "publisher": info_td.get_text(strip=True) if info_td else "네이버금융",
                        "published": date_td.get_text(strip=True) if date_td else "",
                        "source": "NaverFinance",
                        "url": f"https://finance.naver.com{href}" if href.startswith("/") else href,
                    }
                )
                if len(items) >= limit:
                    break
        except Exception:
            pass
        return items

    def _fetch_yfinance_news(self, yf_ticker: str, limit: int = 10) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        try:
            import yfinance as yf

            raw_news = yf.Ticker(yf_ticker).news or []
            for entry in raw_news[:limit]:
                content = entry.get("content", entry) if isinstance(entry, dict) else {}
                title = content.get("title") or entry.get("title")
                pub = content.get("pubDate") or entry.get("providerPublishTime") or ""
                provider = ""
                if isinstance(content.get("provider"), dict):
                    provider = content["provider"].get("displayName", "")
                elif isinstance(entry.get("publisher"), str):
                    provider = entry["publisher"]
                url = ""
                if isinstance(content.get("canonicalUrl"), dict):
                    url = content["canonicalUrl"].get("url", "")
                elif isinstance(entry.get("link"), str):
                    url = entry["link"]
                if title:
                    items.append(
                        {
                            "title": str(title).strip(),
                            "publisher": str(provider).strip(),
                            "published": str(pub),
                            "source": "YahooFinance",
                            "url": str(url),
                        }
                    )
        except Exception:
            pass
        return items

    def _fetch_google_news_rss(self, query: str, market: str, limit: int = 10) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        try:
            hl = "ko" if market == "KR" else "en-US"
            gl = "KR" if market == "KR" else "US"
            ceid = "KR:ko" if market == "KR" else "US:en"
            url = (
                f"https://news.google.com/rss/search?"
                f"q={requests.utils.quote(query)}&hl={hl}&gl={gl}&ceid={ceid}"
            )
            resp = requests.get(url, timeout=5)
            if not resp.ok:
                return items
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "xml")
            for item in soup.find_all("item")[:limit]:
                title_tag = item.find("title")
                pub_tag = item.find("pubDate")
                src_tag = item.find("source")
                link_tag = item.find("link")
                if title_tag and title_tag.text:
                    items.append(
                        {
                            "title": title_tag.text.strip(),
                            "publisher": src_tag.text.strip() if src_tag else "Google News",
                            "published": pub_tag.text.strip() if pub_tag else "",
                            "source": "GoogleNews",
                            "url": link_tag.text.strip() if link_tag else "",
                        }
                    )
        except Exception:
            pass
        return items
