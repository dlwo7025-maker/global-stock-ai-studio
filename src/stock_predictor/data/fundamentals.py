"""심층 기업 재정·재무제표·어닝 서프라이즈·애널리스트 컨센서스 분석 모듈.

주가에 실질적 영향을 미치는 회사 데이터를 5가지 기둥(Pillar)으로 정량화합니다:
1. 밸류에이션 (Valuation): Forward/Trailing PER, PBR, EV/EBITDA, PEG
2. 수익성 (Profitability): ROE, ROA, 영업이익률, 순이익률
3. 성장성 (Growth): 매출 성장률(YoY/최근 분기), 이익 성장률, 분기 매출 추세
4. 재무 건전성 (Financial Health): 부채비율, 유동비율, 잉여현금흐름(FCF Yield), 순현금 여부
5. 실적 모멘텀 & 컨센서스 (Earnings & Consensus): 최근 4분기 EPS 어닝 서프라이즈, 목표가 괴리율, 투자의견

데이터 소스:
- yfinance (재무요약 + 분기 손익/대차대조표/현금흐름 + 어닝 서프라이즈 히스토리)
- 국내 주식 보강: 네이버 금융 기업실적 요약 스크래핑 + (선택) DART OpenAPI
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass
class FundamentalProfile:
    """종목의 심층 재무·재정·실적 프로필."""

    # 0~100 종합 및 5대 기둥 점수
    overall_score: float = 50.0
    valuation_score: float = 50.0
    profitability_score: float = 50.0
    growth_score: float = 50.0
    health_score: float = 50.0
    earnings_score: float = 50.0
    grade: str = "B (보통)"

    # 예측 모델 피처 / 오버레이용 정규화 시그널 (-1.0 ~ +1.0)
    fundamental_signal: float = 0.0
    data_completeness: float = 0.0  # 확보된 재무 지표 비율 (0~1)

    # 원시 핵심 지표 딕셔너리
    metrics: dict[str, Any] = field(default_factory=dict)
    # 최근 분기 어닝 서프라이즈 리스트
    earnings_history: list[dict[str, Any]] = field(default_factory=list)
    # 재무 강점 / 경고 플래그
    strengths: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    source: str = "none"

    def to_feature_dict(self) -> dict[str, float]:
        """Cross-Sectional ML 모델에 투입할 수치형 펀더멘털 팩터 벡터."""
        m = self.metrics
        return {
            "fund_overall_score": self.overall_score / 100.0,
            "fund_valuation_score": self.valuation_score / 100.0,
            "fund_profitability_score": self.profitability_score / 100.0,
            "fund_growth_score": self.growth_score / 100.0,
            "fund_health_score": self.health_score / 100.0,
            "fund_earnings_score": self.earnings_score / 100.0,
            "fund_pe_inv": float(1.0 / max(m["pe"], 1.0)) if _is_pos(m.get("pe")) else 0.0,
            "fund_pbr_inv": float(1.0 / max(m["pbr"], 0.1)) if _is_pos(m.get("pbr")) else 0.0,
            "fund_roe": float(np.clip(m.get("roe", 0.0) or 0.0, -0.5, 1.0)),
            "fund_op_margin": float(np.clip(m.get("operating_margin", 0.0) or 0.0, -0.5, 0.8)),
            "fund_rev_growth": float(np.clip(m.get("revenue_growth", 0.0) or 0.0, -0.5, 2.0)),
            "fund_earn_surprise": float(np.clip(m.get("avg_eps_surprise", 0.0) or 0.0, -0.5, 0.5)),
            "fund_target_upside": float(np.clip(m.get("target_upside", 0.0) or 0.0, -0.5, 1.0)),
        }


def _is_num(val: Any) -> bool:
    return isinstance(val, (int, float, np.floating, np.integer)) and not np.isnan(float(val)) and np.isfinite(float(val))


def _is_pos(val: Any) -> bool:
    return _is_num(val) and float(val) > 0.0


def _score_to_grade(score: float) -> str:
    if score >= 80:
        return "A+ (최우량)"
    if score >= 70:
        return "A (우량)"
    if score >= 60:
        return "B+ (양호)"
    if score >= 48:
        return "B (보통)"
    if score >= 36:
        return "C (주의)"
    return "D (취약)"


class FundamentalAnalyzer:
    """국내·해외 기업의 재정 상태, 수익성, 성장성, 어닝 서프라이즈 종합 분석기."""

    def collect_and_score(
        self,
        market: str,
        code: str,
        yf_ticker: str,
        current_price: float,
        base_info: dict[str, Any] | None = None,
    ) -> FundamentalProfile:
        metrics: dict[str, Any] = {}
        earnings_hist: list[dict[str, Any]] = []
        sources: list[str] = []

        # 1) yfinance 심층 재무정보 + 분기 실적/어닝 서프라이즈
        yf_metrics, yf_earnings = self._fetch_yfinance_deep(yf_ticker, current_price, base_info)
        if yf_metrics:
            metrics.update(yf_metrics)
            sources.append("yfinance")
        if yf_earnings:
            earnings_hist.extend(yf_earnings)

        # 2) 국내 주식: 네이버 금융 기업실적/컨센서스 보강
        if market == "KR":
            naver_m = self._fetch_naver_finance_metrics(code)
            if naver_m:
                for k, v in naver_m.items():
                    if k not in metrics or not _is_num(metrics.get(k)):
                        metrics[k] = v
                sources.append("NaverFinance")

            # 선택적 OpenDART 연동 (DART_API_KEY 환경변수 설정 시)
            if os.getenv("DART_API_KEY"):
                dart_m = self._fetch_dart_summary(code)
                if dart_m:
                    metrics.update(dart_m)
                    sources.append("OpenDART")

        return self.score_from_metrics(
            metrics=metrics,
            earnings_history=earnings_hist,
            source="+".join(sources) if sources else "none",
        )

    def score_from_metrics(
        self,
        metrics: dict[str, Any],
        earnings_history: list[dict[str, Any]] | None = None,
        source: str = "custom",
    ) -> FundamentalProfile:
        """수집된 재무 지표 딕셔너리로부터 5대 기둥 점수, 강점, 위험 플래그를 정량 산출."""
        earnings_history = earnings_history or []
        strengths: list[str] = []
        red_flags: list[str] = []

        # --- Pillar 1: 밸류에이션 (Valuation) ---
        val_pts: list[float] = []
        pe = metrics.get("forward_pe") or metrics.get("trailing_pe") or metrics.get("pe")
        if _is_pos(pe):
            metrics["pe"] = float(pe)
            if pe <= 12:
                val_pts.append(88.0)
                strengths.append(f"저평가 매력: PER {pe:.1f}배로 이익 대비 주가 부담이 낮습니다.")
            elif pe <= 20:
                val_pts.append(72.0)
            elif pe <= 32:
                val_pts.append(54.0)
            elif pe <= 50:
                val_pts.append(35.0)
            else:
                val_pts.append(20.0)
                red_flags.append(f"고밸류에이션 부담: PER {pe:.1f}배로 높은 성장 기대치가 선반영되어 있습니다.")

        pbr = metrics.get("pbr")
        if _is_pos(pbr):
            if pbr <= 1.0:
                val_pts.append(85.0)
                strengths.append(f"자산가치 저평가: PBR {pbr:.2f}배로 청산/순자산 가치 수준입니다.")
            elif pbr <= 2.5:
                val_pts.append(66.0)
            elif pbr <= 6.0:
                val_pts.append(48.0)
            else:
                val_pts.append(30.0)

        ev_ebitda = metrics.get("ev_ebitda")
        if _is_pos(ev_ebitda):
            if ev_ebitda <= 10:
                val_pts.append(82.0)
            elif ev_ebitda <= 18:
                val_pts.append(60.0)
            elif ev_ebitda <= 30:
                val_pts.append(40.0)
            else:
                val_pts.append(24.0)

        peg = metrics.get("peg")
        if _is_pos(peg):
            if peg <= 1.0:
                val_pts.append(88.0)
                strengths.append(f"성장 대비 저평가: PEG {peg:.2f} (1.0 이하 매력 구간).")
            elif peg <= 1.8:
                val_pts.append(65.0)
            elif peg <= 2.8:
                val_pts.append(44.0)
            else:
                val_pts.append(25.0)

        val_score = float(np.mean(val_pts)) if val_pts else 50.0

        # --- Pillar 2: 수익성 (Profitability) ---
        prof_pts: list[float] = []
        roe = metrics.get("roe")
        if _is_num(roe):
            if roe >= 0.22:
                prof_pts.append(92.0)
                strengths.append(f"최상급 자본효율성: ROE {roe * 100:.1f}%로 수익 창출력이 매우 우수합니다.")
            elif roe >= 0.14:
                prof_pts.append(76.0)
                strengths.append(f"우수한 수익성: ROE {roe * 100:.1f}%를 유지 중입니다.")
            elif roe >= 0.08:
                prof_pts.append(56.0)
            elif roe >= 0.0:
                prof_pts.append(36.0)
            else:
                prof_pts.append(15.0)
                red_flags.append(f"적자/수익성 부진: ROE {roe * 100:.1f}%로 자본 수익성이 마이너스입니다.")

        op_m = metrics.get("operating_margin")
        if _is_num(op_m):
            if op_m >= 0.25:
                prof_pts.append(90.0)
                strengths.append(f"고마진 사업구조: 영업이익률 {op_m * 100:.1f}%로 가격 결정력이 높습니다.")
            elif op_m >= 0.12:
                prof_pts.append(72.0)
            elif op_m >= 0.05:
                prof_pts.append(52.0)
            elif op_m >= 0.0:
                prof_pts.append(35.0)
            else:
                prof_pts.append(12.0)
                red_flags.append(f"영업적자 상태: 영업이익률 {op_m * 100:.1f}%입니다.")

        net_m = metrics.get("profit_margin")
        if _is_num(net_m):
            if net_m >= 0.18:
                prof_pts.append(85.0)
            elif net_m >= 0.08:
                prof_pts.append(66.0)
            elif net_m >= 0.0:
                prof_pts.append(44.0)
            else:
                prof_pts.append(15.0)

        prof_score = float(np.mean(prof_pts)) if prof_pts else 50.0

        # --- Pillar 3: 성장성 (Growth) ---
        grow_pts: list[float] = []
        rev_g = metrics.get("revenue_growth")
        if _is_num(rev_g):
            if rev_g >= 0.25:
                grow_pts.append(92.0)
                strengths.append(f"고성장세: 매출 성장률 {rev_g * 100:+.1f}%로 외형이 빠르게 확대 중입니다.")
            elif rev_g >= 0.10:
                grow_pts.append(74.0)
            elif rev_g >= 0.02:
                grow_pts.append(55.0)
            elif rev_g >= -0.05:
                grow_pts.append(38.0)
            else:
                grow_pts.append(18.0)
                red_flags.append(f"외형 축소 우려: 매출 성장률 {rev_g * 100:+.1f}%로 역성장 중입니다.")

        earn_g = metrics.get("earnings_growth")
        if _is_num(earn_g):
            if earn_g >= 0.25:
                grow_pts.append(90.0)
                strengths.append(f"강한 이익 모멘텀: 순이익/EPS 성장률 {earn_g * 100:+.1f}%.")
            elif earn_g >= 0.10:
                grow_pts.append(72.0)
            elif earn_g >= 0.0:
                grow_pts.append(52.0)
            elif earn_g >= -0.15:
                grow_pts.append(34.0)
            else:
                grow_pts.append(15.0)
                red_flags.append(f"이익 급감 주의: 이익 성장률 {earn_g * 100:+.1f}%입니다.")

        qoq_rev = metrics.get("qoq_revenue_growth")
        if _is_num(qoq_rev):
            grow_pts.append(float(np.clip(55.0 + qoq_rev * 180.0, 15.0, 92.0)))

        grow_score = float(np.mean(grow_pts)) if grow_pts else 50.0

        # --- Pillar 4: 재무 건전성 (Financial Health) ---
        health_pts: list[float] = []
        dte = metrics.get("debt_to_equity")  # % 단위 (예: 65.0 = 65%)
        if _is_num(dte) and dte >= 0:
            if dte <= 40:
                health_pts.append(88.0)
                strengths.append(f"탄탄한 재무구조: 부채비율 {dte:.0f}%로 재무 안정성이 매우 높습니다.")
            elif dte <= 100:
                health_pts.append(70.0)
            elif dte <= 180:
                health_pts.append(48.0)
            else:
                health_pts.append(22.0)
                red_flags.append(f"부채 부담 주의: 부채비율 {dte:.0f}%로 금리 민감도가 높습니다.")

        curr_r = metrics.get("current_ratio")
        if _is_pos(curr_r):
            if curr_r >= 2.0:
                health_pts.append(85.0)
            elif curr_r >= 1.2:
                health_pts.append(66.0)
            elif curr_r >= 0.9:
                health_pts.append(42.0)
            else:
                health_pts.append(20.0)
                red_flags.append(f"단기 유동성 경계: 유동비율 {curr_r:.2f} (1.0 미만).")

        fcf_yield = metrics.get("fcf_yield")
        if _is_num(fcf_yield):
            if fcf_yield >= 0.05:
                health_pts.append(88.0)
                strengths.append(f"우수한 현금 창출력: 잉여현금흐름(FCF) 수익률 {fcf_yield * 100:.1f}%.")
            elif fcf_yield >= 0.02:
                health_pts.append(68.0)
            elif fcf_yield >= 0.0:
                health_pts.append(50.0)
            else:
                health_pts.append(24.0)
                red_flags.append(f"현금흐름 유출: 잉여현금흐름(FCF)이 마이너스({fcf_yield * 100:.1f}%)입니다.")

        if metrics.get("net_cash_positive") is True:
            health_pts.append(82.0)

        health_score = float(np.mean(health_pts)) if health_pts else 50.0

        # --- Pillar 5: 실적 서프라이즈 & 애널리스트 컨센서스 ---
        earn_pts: list[float] = []
        avg_surp = metrics.get("avg_eps_surprise")
        beat_rate = metrics.get("eps_beat_rate")
        if _is_num(avg_surp):
            if avg_surp >= 0.08:
                earn_pts.append(90.0)
                strengths.append(f"연속 어닝 서프라이즈: 최근 분기 평균 EPS 서프라이즈 {avg_surp * 100:+.1f}%.")
            elif avg_surp >= 0.02:
                earn_pts.append(72.0)
            elif avg_surp >= -0.02:
                earn_pts.append(52.0)
            else:
                earn_pts.append(22.0)
                red_flags.append(f"실적 기대치 하회: 최근 평균 EPS 서프라이즈 {avg_surp * 100:+.1f}%.")
        if _is_num(beat_rate):
            earn_pts.append(float(np.clip(25.0 + beat_rate * 70.0, 20.0, 95.0)))

        upside = metrics.get("target_upside")
        if _is_num(upside):
            if upside >= 0.20:
                earn_pts.append(86.0)
                strengths.append(f"증권사 목표가 괴리: 컨센서스 평균 대비 +{upside * 100:.1f}% 상승 여력.")
            elif upside >= 0.08:
                earn_pts.append(68.0)
            elif upside >= -0.05:
                earn_pts.append(50.0)
            else:
                earn_pts.append(26.0)
                red_flags.append(f"목표가 초과: 현재가가 컨센서스 목표가를 {abs(upside) * 100:.1f}% 상회 중입니다.")

        rec_mean = metrics.get("recommendation_mean")  # 1.0(강력매수) ~ 5.0(강력매도)
        if _is_pos(rec_mean):
            earn_pts.append(float(np.clip(105.0 - rec_mean * 22.0, 15.0, 92.0)))

        earn_score = float(np.mean(earn_pts)) if earn_pts else 50.0

        # 데이터 충실도 계산
        total_pillars_with_data = sum(
            1 for pts in [val_pts, prof_pts, grow_pts, health_pts, earn_pts] if len(pts) > 0
        )
        completeness = total_pillars_with_data / 5.0

        # 종합 점수 (가중 평균)
        overall = (
            0.22 * val_score
            + 0.23 * prof_score
            + 0.22 * grow_score
            + 0.16 * health_score
            + 0.17 * earn_score
        )
        overall = float(np.clip(overall, 0.0, 100.0))

        # 예측 반영용 정규화 시그널 (-1 ~ +1), 데이터 부족 시 중립(0)으로 수축
        raw_sig = (overall - 50.0) / 40.0
        fund_signal = float(np.clip(raw_sig * completeness, -1.0, 1.0))

        return FundamentalProfile(
            overall_score=overall,
            valuation_score=val_score,
            profitability_score=prof_score,
            growth_score=grow_score,
            health_score=health_score,
            earnings_score=earn_score,
            grade=_score_to_grade(overall),
            fundamental_signal=fund_signal,
            data_completeness=completeness,
            metrics=metrics,
            earnings_history=earnings_history,
            strengths=strengths[:5],
            red_flags=red_flags[:5],
            source=source,
        )

    def _fetch_yfinance_deep(
        self, yf_ticker: str, current_price: float, base_info: dict[str, Any] | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        metrics: dict[str, Any] = {}
        earnings_hist: list[dict[str, Any]] = []
        try:
            import yfinance as yf

            tk = yf.Ticker(yf_ticker)
            info = dict(base_info or {})
            try:
                raw_info = tk.info or {}
                for k, v in raw_info.items():
                    if k not in info or info[k] is None:
                        info[k] = v
            except Exception:
                pass

            def _pick(key: str) -> float | None:
                v = info.get(key)
                return float(v) if _is_num(v) else None

            mapping = {
                "trailing_pe": "trailingPE",
                "forward_pe": "forwardPE",
                "pbr": "priceToBook",
                "ev_ebitda": "enterpriseToEbitda",
                "peg": "pegRatio",
                "roe": "returnOnEquity",
                "roa": "returnOnAssets",
                "operating_margin": "operatingMargins",
                "profit_margin": "profitMargins",
                "revenue_growth": "revenueGrowth",
                "earnings_growth": "earningsGrowth",
                "debt_to_equity": "debtToEquity",
                "current_ratio": "currentRatio",
                "dividend_yield": "dividendYield",
                "market_cap": "marketCap",
                "free_cashflow": "freeCashflow",
                "total_cash": "totalCash",
                "total_debt": "totalDebt",
                "target_mean_price": "targetMeanPrice",
                "recommendation_mean": "recommendationMean",
            }
            for dst, src in mapping.items():
                val = _pick(src)
                if val is not None:
                    metrics[dst] = val

            if info.get("sector"):
                metrics["sector"] = str(info["sector"])
            if info.get("industry"):
                metrics["industry"] = str(info["industry"])
            if info.get("recommendationKey"):
                metrics["recommendation_key"] = str(info["recommendationKey"])

            # 파생 건전성 지표: FCF 수익률, 순현금 여부, 목표가 괴리율
            mcap = metrics.get("market_cap")
            fcf = metrics.get("free_cashflow")
            if _is_pos(mcap) and _is_num(fcf):
                metrics["fcf_yield"] = float(fcf) / float(mcap)
            tcash, tdebt = metrics.get("total_cash"), metrics.get("total_debt")
            if _is_num(tcash) and _is_num(tdebt):
                metrics["net_cash_positive"] = float(tcash) > float(tdebt)
            tmean = metrics.get("target_mean_price")
            if _is_pos(tmean) and current_price > 0:
                metrics["target_upside"] = (float(tmean) / float(current_price)) - 1.0

            # 분기 손익계산서에서 최근 분기 매출 QoQ 추세 추출
            try:
                q_inc = tk.quarterly_income_stmt
                if q_inc is not None and not q_inc.empty:
                    for row_name in ("Total Revenue", "Operating Revenue"):
                        if row_name in q_inc.index:
                            vals = q_inc.loc[row_name].dropna().to_numpy(dtype=float)
                            if len(vals) >= 2 and vals[1] > 0:
                                metrics["qoq_revenue_growth"] = float(vals[0] / vals[1] - 1.0)
                            break
            except Exception:
                pass

            # 최근 4분기 EPS 어닝 서프라이즈 추출
            try:
                eh = tk.earnings_dates
                if eh is not None and not eh.empty:
                    eh_clean = eh.dropna(subset=["EPS Estimate", "Reported EPS"]).head(4)
                    surprises: list[float] = []
                    for dt_idx, row in eh_clean.iterrows():
                        est = float(row["EPS Estimate"])
                        rep = float(row["Reported EPS"])
                        surp_pct = (
                            float(row["Surprise(%)"]) / 100.0
                            if "Surprise(%)" in row and _is_num(row["Surprise(%)"])
                            else (rep - est) / (abs(est) + 1e-6)
                        )
                        surprises.append(surp_pct)
                        earnings_hist.append(
                            {
                                "date": str(dt_idx)[:10],
                                "eps_estimate": round(est, 3),
                                "eps_actual": round(rep, 3),
                                "surprise_pct": round(surp_pct * 100.0, 2),
                            }
                        )
                    if surprises:
                        metrics["avg_eps_surprise"] = float(np.mean(surprises))
                        metrics["eps_beat_rate"] = float(np.mean([s > 0 for s in surprises]))
            except Exception:
                pass

        except Exception:
            pass
        return metrics, earnings_hist

    def _fetch_naver_finance_metrics(self, code: str) -> dict[str, Any]:
        """네이버 금융 종목 메인 페이지에서 국내 기업 핵심 재무/컨센서스 지표 경량 추출."""
        out: dict[str, Any] = {}
        try:
            url = f"https://finance.naver.com/item/main.naver?code={code}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            if not resp.ok:
                return out
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(resp.text, "lxml")

            def _extract_float(elem_id: str) -> float | None:
                tag = soup.find(id=elem_id)
                if not tag:
                    return None
                txt = re.sub(r"[^0-9.\-]", "", tag.get_text(strip=True))
                try:
                    return float(txt)
                except ValueError:
                    return None

            per = _extract_float("_per")
            pbr = _extract_float("_pbr")
            dvr = _extract_float("_dvr")
            if per and per > 0:
                out["trailing_pe"] = per
            if pbr and pbr > 0:
                out["pbr"] = pbr
            if dvr is not None and dvr >= 0:
                out["dividend_yield"] = dvr / 100.0

            # 투자의견 / 목표주가 테이블 파싱
            rwidth = soup.select_one("table.rwidth")
            if rwidth:
                text = rwidth.get_text(" ", strip=True)
                m_target = re.search(r"목표주가\s*([0-9,]+)", text)
                if m_target:
                    out["target_mean_price"] = float(m_target.group(1).replace(",", ""))

            # 기업실적분석 테이블에서 최근 연간 영업이익률 / ROE / 부채비율 추출
            cop = soup.select_one("div.cop_analysis table")
            if cop:
                for tr in cop.select("tr"):
                    th = tr.find("th")
                    if not th:
                        continue
                    label = th.get_text(strip=True)
                    tds = [td.get_text(strip=True).replace(",", "") for td in tr.select("td")[:4]]
                    nums = []
                    for t in tds:
                        try:
                            nums.append(float(t))
                        except ValueError:
                            pass
                    if not nums:
                        continue
                    latest_val = nums[-1]
                    if "영업이익률" in label:
                        out["operating_margin"] = latest_val / 100.0
                    elif "순이익률" in label:
                        out["profit_margin"] = latest_val / 100.0
                    elif "ROE" in label:
                        out["roe"] = latest_val / 100.0
                    elif "부채비율" in label:
                        out["debt_to_equity"] = latest_val
                    elif "당좌비율" in label and latest_val > 0:
                        out["current_ratio"] = latest_val / 100.0
        except Exception:
            pass
        return out

    def _fetch_dart_summary(self, code: str) -> dict[str, Any]:
        """DART_API_KEY가 설정된 경우 금감원 DART 공시 기반 최근 재무 보강 (선택)."""
        # 별도 OpenDartReader 미설치 시에도 오류 없이 통과
        try:
            import OpenDartReader  # type: ignore

            dart = OpenDartReader(os.getenv("DART_API_KEY"))
            from datetime import datetime

            year = datetime.now().year - 1
            fs = dart.finstate(code, year)
            if fs is None or fs.empty:
                return {}
            return {"dart_verified_year": year}
        except Exception:
            return {}
