"""국내(KRX) 및 해외(US/Global) 주가 · 거시지표 · 펀더멘털 · 수급 · 뉴스 통합 수집 모듈.

핵심 설계:
1. 국내 주식: FinanceDataReader -> pykrx -> yfinance 순으로 자동 폴백하여 수집 안정성 극대화
2. 해외 주식: yfinance 기반 수정주가(OHLCV) + 펀더멘털 + 최신 뉴스 수집
3. 글로벌 매크로 컨텍스트: S&P500, 나스닥, 코스피, VIX, 미국 10년물 금리, 원/달러 환율, DXY 결합
4. Look-ahead Bias(미래 정보 유출) 방지: 한국 시장(KST) 종목에는 미국 거시지표를 반드시 1거래일 지연(Lag) 적용
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ..config import (
    CACHE_DIR,
    KNOWN_KOSDAQ_CODES,
    KR_STOCK_PRESETS,
    MACRO_TICKERS,
    US_STOCK_PRESETS,
)
from .alt_intelligence import (
    AltMarketIntelligenceEngine,
    MacroPredictionPulse,
    StockAltSignal,
)
from .fundamentals import FundamentalAnalyzer, FundamentalProfile
from .news_intelligence import NewsIntelligenceEngine, NewsIntelligenceReport

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ResolvedSymbol:
    """정규화된 종목 식별 정보."""

    raw_input: str
    market: str          # "KR" 또는 "US"
    code: str            # KR: "005930", US: "NVDA"
    yf_ticker: str       # KR: "005930.KS" or "005930.KQ", US: "NVDA"
    name: str            # 종목명 (표시용)
    currency: str        # "KRW" 또는 "USD"
    benchmark_key: str   # "kospi" 또는 "sp500"


@dataclass
class MarketDataBundle:
    """종목 분석·예측에 필요한 모든 원시 및 심층 정보 묶음."""

    symbol: ResolvedSymbol
    ohlcv: pd.DataFrame                        # Index: DatetimeIndex, Cols: Open, High, Low, Close, Volume
    macro: pd.DataFrame                        # 거시지표 일별 종가 DataFrame
    investor_flow: pd.DataFrame | None = None  # 국내 주식 외국인/기관 순매수 (선택)
    fundamentals: dict[str, Any] = field(default_factory=dict)
    fundamental_profile: FundamentalProfile | None = None
    news: list[dict[str, str]] = field(default_factory=list)
    news_report: NewsIntelligenceReport | None = None
    macro_pulse: MacroPredictionPulse | None = None
    alt_signal: StockAltSignal | None = None
    data_source: str = "unknown"
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _clean_yf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """최신 yfinance의 MultiIndex 컬럼 및 타임존을 표준 OHLCV 형식으로 정규화."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    # 중복 컬럼 제거
    out = out.loc[:, ~out.columns.duplicated()]
    rename_map = {c: c.capitalize() for c in out.columns if isinstance(c, str)}
    out = out.rename(columns=rename_map)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in out.columns for col in required):
        return pd.DataFrame()
    out = out[required].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # 종가 0 또는 결측 행 제거
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    # OHLC 결측은 종가로 보정, 거래량 결측은 0
    for col in ["Open", "High", "Low"]:
        out[col] = out[col].fillna(out["Close"])
        out.loc[out[col] <= 0, col] = out.loc[out[col] <= 0, "Close"]
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0.0)
    return out


def resolve_symbol(query: str) -> ResolvedSymbol:
    """사용자 입력(종목명, 6자리 코드, .KS/.KQ, 미국 티커)을 표준 ResolvedSymbol로 변환."""
    raw = query.strip()
    if not raw:
        raise ValueError("종목 코드 또는 종목명을 입력해 주세요.")

    # 1. 국내 프리셋 한글명 매칭
    if raw in KR_STOCK_PRESETS:
        code = KR_STOCK_PRESETS[raw]
        suffix = ".KQ" if code in KNOWN_KOSDAQ_CODES else ".KS"
        return ResolvedSymbol(
            raw_input=raw,
            market="KR",
            code=code,
            yf_ticker=f"{code}{suffix}",
            name=raw,
            currency="KRW",
            benchmark_key="kospi",
        )

    upper = raw.upper()

    # 2. 국내 티커 (.KS / .KQ 명시)
    kr_dot = re.fullmatch(r"(\d{6})\.(KS|KQ)", upper)
    if kr_dot:
        code, market_suffix = kr_dot.group(1), kr_dot.group(2)
        preset_name = next((k for k, v in KR_STOCK_PRESETS.items() if v == code), code)
        return ResolvedSymbol(
            raw_input=raw,
            market="KR",
            code=code,
            yf_ticker=f"{code}.{market_suffix}",
            name=preset_name,
            currency="KRW",
            benchmark_key="kospi",
        )

    # 3. 순수 6자리 숫자 -> 국내 종목
    if re.fullmatch(r"\d{6}", upper):
        code = upper
        suffix = ".KQ" if code in KNOWN_KOSDAQ_CODES else ".KS"
        preset_name = next((k for k, v in KR_STOCK_PRESETS.items() if v == code), code)
        return ResolvedSymbol(
            raw_input=raw,
            market="KR",
            code=code,
            yf_ticker=f"{code}{suffix}",
            name=preset_name,
            currency="KRW",
            benchmark_key="kospi",
        )

    # 4. 한글이 포함되어 있으면 KRX 종목 리스트에서 탐색 시도
    if re.search(r"[가-힣]", raw):
        found = _lookup_krx_name(raw)
        if found is not None:
            code, name, market_type = found
            suffix = ".KQ" if "KOSDAQ" in market_type.upper() else ".KS"
            return ResolvedSymbol(
                raw_input=raw,
                market="KR",
                code=code,
                yf_ticker=f"{code}{suffix}",
                name=name,
                currency="KRW",
                benchmark_key="kospi",
            )
        raise ValueError(f"국내 종목 '{raw}'을(를) 찾을 수 없습니다. 6자리 종목코드(예: 005930)로 입력해 보세요.")

    # 5. 해외(미국/글로벌) 티커
    us_name = next((k for k, v in US_STOCK_PRESETS.items() if v == upper), upper)
    return ResolvedSymbol(
        raw_input=raw,
        market="US",
        code=upper,
        yf_ticker=upper,
        name=us_name,
        currency="USD",
        benchmark_key="sp500",
    )


def _lookup_krx_name(name_query: str) -> tuple[str, str, str] | None:
    """FinanceDataReader를 통해 한글 종목명으로 KRX 종목코드 조회."""
    try:
        import FinanceDataReader as fdr

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        listing_cache = CACHE_DIR / "krx_listing.pkl"
        df = None
        if listing_cache.exists():
            age_hours = (time.time() - listing_cache.stat().st_mtime) / 3600.0
            if age_hours < 24:
                df = pd.read_pickle(listing_cache)
        if df is None:
            df = fdr.StockListing("KRX")
            if df is not None and not df.empty:
                df.to_pickle(listing_cache)
        if df is None or df.empty:
            return None
        name_col = "Name" if "Name" in df.columns else df.columns[2]
        code_col = "Code" if "Code" in df.columns else df.columns[0]
        mkt_col = "Market" if "Market" in df.columns else None

        exact = df[df[name_col].astype(str).str.strip().str.lower() == name_query.strip().lower()]
        hit = exact if not exact.empty else df[df[name_col].astype(str).str.contains(name_query.strip(), case=False, na=False)]
        if hit.empty:
            return None
        row = hit.iloc[0]
        code = str(row[code_col]).zfill(6)
        name = str(row[name_col])
        mkt = str(row[mkt_col]) if mkt_col else "KOSPI"
        return code, name, mkt
    except Exception:
        return None


class MarketDataLoader:
    """국내·해외 주가, 거시지표, 수급, 펀더멘털, 뉴스 통합 로더."""

    def __init__(
        self,
        use_cache: bool = True,
        cache_ttl_hours: int = 4,
        use_llm_news: bool = False,
    ) -> None:
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours
        self.use_llm_news = use_llm_news
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.news_engine = NewsIntelligenceEngine(use_llm=use_llm_news)
        self.alt_engine = AltMarketIntelligenceEngine()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        return CACHE_DIR / f"{digest}.pkl"

    def _load_cache(self, key: str) -> Any | None:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours > self.cache_ttl_hours:
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, key: str, value: Any) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(key)
        try:
            with path.open("wb") as f:
                pickle.dump(value, f)
        except Exception:
            pass

    def load_bundle(
        self,
        query: str,
        lookback_years: int = 5,
        include_deep_info: bool = True,
    ) -> MarketDataBundle:
        """종목의 전체 원시 데이터(OHLCV, 거시지표, 수급, 심층 재무, 다중 소스 뉴스)를 수집."""
        symbol = resolve_symbol(query)
        today_str = datetime.now().strftime("%Y-%m-%d")
        info_tag = "deep" if include_deep_info else "priceonly"
        llm_tag = "llm" if (include_deep_info and self.use_llm_news) else "rule"
        cache_key = f"bundle_v4_{symbol.market}_{symbol.code}_{lookback_years}_{info_tag}_{llm_tag}_{today_str}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        end_dt = datetime.now()
        # 200일 이동평균 등 초기 롤링 피처 계산을 위해 여유 기간(+180일) 추가 확보
        start_dt = end_dt - timedelta(days=int(lookback_years * 365 + 180))
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        if symbol.market == "KR":
            ohlcv, source, resolved_yf = self._fetch_kr_ohlcv(symbol, start_str, end_str)
            symbol.yf_ticker = resolved_yf
            investor_flow = self._fetch_kr_investor_flow(symbol.code, start_dt, end_dt) if include_deep_info else None
            fundamentals = self._fetch_kr_fundamentals(symbol) if include_deep_info else {}
        else:
            ohlcv, source = self._fetch_us_ohlcv(symbol.yf_ticker, start_str, end_str)
            investor_flow = None
            fundamentals = self._fetch_us_fundamentals(symbol.yf_ticker) if include_deep_info else {}
            if fundamentals.get("shortName") and symbol.name == symbol.code:
                symbol.name = str(fundamentals["shortName"])

        if ohlcv.empty or len(ohlcv) < 120:
            raise RuntimeError(
                f"'{symbol.name} ({symbol.code})'의 충분한 주가 데이터를 가져오지 못했습니다 "
                f"(수집 행 수: {len(ohlcv)}). 종목코드나 네트워크 연결을 확인해 주세요."
            )

        macro = self.load_macro_data(start_str, end_str)

        latest_close = float(ohlcv["Close"].iloc[-1])
        fundamental_profile = None
        news_report = None
        macro_pulse = None
        alt_signal = None
        news_list: list[dict[str, str]] = []

        if include_deep_info:
            fundamental_profile = self.fundamental_analyzer.collect_and_score(
                market=symbol.market,
                code=symbol.code,
                yf_ticker=symbol.yf_ticker,
                current_price=latest_close,
                base_info=fundamentals,
            )
            news_report = self.news_engine.collect_and_analyze(
                market=symbol.market,
                code=symbol.code,
                name=symbol.name,
                yf_ticker=symbol.yf_ticker,
                max_items=16,
            )
            vix_hint = (
                float(macro["vix"].dropna().iloc[-1])
                if ("vix" in macro.columns and not macro["vix"].dropna().empty)
                else 20.0
            )
            macro_pulse = self.alt_engine.get_macro_pulse(vix_hint=vix_hint, use_cache=self.use_cache)
            alt_signal = self.alt_engine.get_stock_alt_signal(
                market=symbol.market,
                code=symbol.code,
                name=symbol.name,
                yf_ticker=symbol.yf_ticker,
                macro_pulse=macro_pulse,
                use_cache=self.use_cache,
            )
            news_list = [
                {
                    "title": it.title,
                    "publisher": it.publisher,
                    "published": it.published,
                    "category": it.category,
                    "sentiment": str(round(it.sentiment, 2)),
                    "source": it.source,
                    "url": it.url,
                }
                for it in news_report.items
            ]

        bundle = MarketDataBundle(
            symbol=symbol,
            ohlcv=ohlcv,
            macro=macro,
            investor_flow=investor_flow,
            fundamentals=fundamentals,
            fundamental_profile=fundamental_profile,
            news=news_list,
            news_report=news_report,
            macro_pulse=macro_pulse,
            alt_signal=alt_signal,
            data_source=source,
        )
        self._save_cache(cache_key, bundle)
        return bundle

    def _fetch_kr_ohlcv(
        self, symbol: ResolvedSymbol, start_str: str, end_str: str
    ) -> tuple[pd.DataFrame, str, str]:
        """국내 주식 OHLCV: FinanceDataReader -> pykrx -> yfinance 순 폴백."""
        # 1) FinanceDataReader
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader(symbol.code, start_str, end_str)
            cleaned = _clean_yf_dataframe(df)
            if not cleaned.empty and len(cleaned) >= 120:
                return cleaned, "FinanceDataReader", symbol.yf_ticker
        except Exception:
            pass

        # 2) pykrx
        try:
            from pykrx import stock as krx_stock

            s = start_str.replace("-", "")
            e = end_str.replace("-", "")
            df = krx_stock.get_market_ohlcv_by_date(s, e, symbol.code)
            if df is not None and not df.empty:
                df = df.rename(
                    columns={
                        "시가": "Open",
                        "고가": "High",
                        "저가": "Low",
                        "종가": "Close",
                        "거래량": "Volume",
                    }
                )
                cleaned = _clean_yf_dataframe(df)
                if not cleaned.empty and len(cleaned) >= 120:
                    return cleaned, "pykrx", symbol.yf_ticker
        except Exception:
            pass

        # 3) yfinance (.KS 및 .KQ 자동 시도)
        candidates = [symbol.yf_ticker]
        alt = f"{symbol.code}.KQ" if symbol.yf_ticker.endswith(".KS") else f"{symbol.code}.KS"
        candidates.append(alt)
        for cand in candidates:
            df, src = self._fetch_us_ohlcv(cand, start_str, end_str)
            if not df.empty and len(df) >= 120:
                return df, f"{src} ({cand})", cand

        return pd.DataFrame(), "none", symbol.yf_ticker

    def _fetch_us_ohlcv(self, yf_ticker: str, start_str: str, end_str: str) -> tuple[pd.DataFrame, str]:
        """해외 주식 / ETF OHLCV 수집 (yfinance)."""
        try:
            import yfinance as yf

            df = yf.download(
                yf_ticker,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            cleaned = _clean_yf_dataframe(df)
            if not cleaned.empty:
                return cleaned, "yfinance"
        except Exception:
            pass

        # 폴백: FinanceDataReader
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader(yf_ticker, start_str, end_str)
            cleaned = _clean_yf_dataframe(df)
            if not cleaned.empty:
                return cleaned, "FinanceDataReader"
        except Exception:
            pass

        return pd.DataFrame(), "none"

    def _fetch_kr_investor_flow(
        self, code: str, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame | None:
        """pykrx로 국내 종목 외국인/기관 일별 순매수 금액 수집 (KRX 인증정보 없으면 스킵)."""
        if not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
            return None
        try:
            from pykrx import stock as krx_stock

            # 최근 2년~전체 구간 수급 (호출 안정성 위해 3년으로 제한)
            flow_start = max(start_dt, end_dt - timedelta(days=365 * 3))
            df = krx_stock.get_market_trading_value_by_date(
                flow_start.strftime("%Y%m%d"),
                end_dt.strftime("%Y%m%d"),
                code,
            )
            if df is None or df.empty:
                return None
            out = pd.DataFrame(index=pd.to_datetime(df.index).normalize())
            if "외국인합계" in df.columns:
                out["foreign_net"] = pd.to_numeric(df["외국인합계"], errors="coerce")
            elif "외국인" in df.columns:
                out["foreign_net"] = pd.to_numeric(df["외국인"], errors="coerce")
            if "기관합계" in df.columns:
                out["inst_net"] = pd.to_numeric(df["기관합계"], errors="coerce")
            elif "기관" in df.columns:
                out["inst_net"] = pd.to_numeric(df["기관"], errors="coerce")
            if "개인" in df.columns:
                out["retail_net"] = pd.to_numeric(df["개인"], errors="coerce")
            return out.dropna(how="all").sort_index() if not out.empty else None
        except Exception:
            return None

    def _fetch_kr_fundamentals(self, symbol: ResolvedSymbol) -> dict[str, Any]:
        """국내 주식 펀더멘털 (pykrx 인증 시 KRX 우선 + yfinance 보강)."""
        info: dict[str, Any] = {}
        if os.getenv("KRX_ID") and os.getenv("KRX_PW"):
            try:
                from pykrx import stock as krx_stock

                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=20)
                fdf = krx_stock.get_market_fundamental_by_date(
                    start_dt.strftime("%Y%m%d"),
                    end_dt.strftime("%Y%m%d"),
                    symbol.code,
                )
                if fdf is not None and not fdf.empty:
                    last = fdf.iloc[-1]
                    if "PER" in last and float(last["PER"]) > 0:
                        info["trailingPE"] = float(last["PER"])
                    if "PBR" in last and float(last["PBR"]) > 0:
                        info["priceToBook"] = float(last["PBR"])
                    if "DIV" in last and float(last["DIV"]) >= 0:
                        info["dividendYield"] = float(last["DIV"]) / 100.0
                    if "EPS" in last:
                        info["trailingEps"] = float(last["EPS"])
                    if "BPS" in last:
                        info["bookValue"] = float(last["BPS"])
            except Exception:
                pass

        # yfinance 정보로 시가총액, ROE, 성장률 등 보강
        yf_info = self._fetch_us_fundamentals(symbol.yf_ticker)
        for k, v in yf_info.items():
            if k not in info or info[k] is None:
                info[k] = v
        return info

    def _fetch_us_fundamentals(self, yf_ticker: str) -> dict[str, Any]:
        """yfinance를 통한 기업 펀더멘털 요약 수집."""
        keys = [
            "shortName",
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "returnOnEquity",
            "profitMargins",
            "revenueGrowth",
            "earningsGrowth",
            "debtToEquity",
            "dividendYield",
            "beta",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "targetMeanPrice",
            "recommendationKey",
        ]
        result: dict[str, Any] = {}
        try:
            import yfinance as yf

            ticker = yf.Ticker(yf_ticker)
            raw_info = ticker.info or {}
            for k in keys:
                val = raw_info.get(k)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    result[k] = val
        except Exception:
            pass
        return result

    def _fetch_news(self, symbol: ResolvedSymbol, max_items: int = 8) -> list[dict[str, str]]:
        """최신 종목 뉴스 헤드라인 수집 (yfinance + Google News RSS 폴백)."""
        items: list[dict[str, str]] = []
        try:
            import yfinance as yf

            raw_news = yf.Ticker(symbol.yf_ticker).news or []
            for entry in raw_news[:max_items]:
                content = entry.get("content", entry) if isinstance(entry, dict) else {}
                title = content.get("title") or entry.get("title")
                pub = content.get("pubDate") or entry.get("providerPublishTime") or ""
                provider = ""
                if isinstance(content.get("provider"), dict):
                    provider = content["provider"].get("displayName", "")
                elif isinstance(entry.get("publisher"), str):
                    provider = entry["publisher"]
                if title:
                    items.append(
                        {
                            "title": str(title).strip(),
                            "publisher": str(provider).strip(),
                            "published": str(pub),
                        }
                    )
        except Exception:
            pass

        # 국내 주식 또는 뉴스 부족 시 Google News RSS 경량 보강
        if len(items) < 3:
            try:
                q = f"{symbol.name} 주가" if symbol.market == "KR" else f"{symbol.code} stock"
                hl = "ko" if symbol.market == "KR" else "en-US"
                gl = "KR" if symbol.market == "KR" else "US"
                ceid = "KR:ko" if symbol.market == "KR" else "US:en"
                url = (
                    f"https://news.google.com/rss/search?"
                    f"q={requests.utils.quote(q)}&hl={hl}&gl={gl}&ceid={ceid}"
                )
                resp = requests.get(url, timeout=5)
                if resp.ok:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(resp.text, "xml")
                    for item in soup.find_all("item")[: max_items - len(items)]:
                        title_tag = item.find("title")
                        pub_tag = item.find("pubDate")
                        src_tag = item.find("source")
                        if title_tag and title_tag.text:
                            items.append(
                                {
                                    "title": title_tag.text.strip(),
                                    "publisher": src_tag.text.strip() if src_tag else "Google News",
                                    "published": pub_tag.text.strip() if pub_tag else "",
                                }
                            )
            except Exception:
                pass
        return items[:max_items]

    def load_macro_data(self, start_str: str, end_str: str) -> pd.DataFrame:
        """글로벌 거시지표(S&P500, 나스닥, 코스피, VIX, 미10년금리, 원/달러, DXY) 일괄 수집."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"macro_v2_{start_str}_{today_str}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        series_map: dict[str, pd.Series] = {}
        try:
            import yfinance as yf

            tickers_list = list(MACRO_TICKERS.values())
            raw = yf.download(
                tickers_list,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw is not None and not raw.empty:
                for key, ticker in MACRO_TICKERS.items():
                    try:
                        if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                            sub = raw[ticker]
                            close_col = "Close" if "Close" in sub.columns else sub.columns[0]
                            s = pd.to_numeric(sub[close_col], errors="coerce").dropna()
                        elif "Close" in raw.columns:
                            s = pd.to_numeric(raw["Close"], errors="coerce").dropna()
                        else:
                            continue
                        s.index = pd.to_datetime(s.index)
                        if s.index.tz is not None:
                            s.index = s.index.tz_localize(None)
                        s.index = s.index.normalize()
                        s = s[~s.index.duplicated(keep="last")].sort_index()
                        if len(s) > 30:
                            series_map[key] = s
                    except Exception:
                        continue
        except Exception:
            pass

        if not series_map:
            return pd.DataFrame()

        macro_df = pd.DataFrame(series_map).sort_index().ffill()
        self._save_cache(cache_key, macro_df)
        return macro_df
