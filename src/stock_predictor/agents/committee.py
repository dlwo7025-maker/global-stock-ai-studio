"""TradingAgents & ai-hedge-fund 영감의 5인 AI 투자 위원회 모듈 (v2 심층 재무 + 이벤트 이슈 통합).

- 기업 재정 5대 기둥(FundamentalProfile)과 6대 이벤트 뉴스 분석(NewsIntelligenceReport)을
  애널리스트 평가와 Bull/Bear 토론 및 CIO 메모에 깊게 통합합니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..data.features import FeatureDataset
from ..data.market_data import MarketDataBundle
from ..models.ensemble import EnsembleForecast
from ..trading.winrate_evolver import load_champion_artifact, predict_stock_win_confidence


@dataclass
class AgentOpinion:
    """개별 애널리스트 에이전트의 평가 결과."""

    role: str
    persona: str
    stance: str            # "강세" | "중립" | "약세"
    score: float           # -100 ~ +100
    summary: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class CommitteeVerdict:
    """5인 투자 위원회 최종 합의 결과."""

    action: str                      # "강력 매수" | "매수" | "관망(중립)" | "매도" | "강력 매도"
    composite_score: float           # -100 ~ +100
    recommended_position_pct: float  # 권장 포지션 비중 (0 ~ 100%)
    risk_level: str                  # "낮음" | "보통" | "높음" | "매우 높음"
    opinions: list[AgentOpinion]
    bull_case: list[str]
    bear_case: list[str]
    news_sentiment_score: float      # -1 ~ +1
    cio_memo: str
    llm_enhanced: bool = False
    win_confidence_pct: float = 50.0
    champion_filter_passed: bool = True
    champion_filter_note: str = ""


def _score_to_stance(score: float) -> str:
    if score >= 18.0:
        return "강세"
    if score <= -18.0:
        return "약세"
    return "중립"


class InvestmentCommittee:
    """5인 전문 에이전트 기반 투자 위원회."""

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm

    def evaluate(
        self,
        bundle: MarketDataBundle,
        dataset: FeatureDataset,
        forecast: EnsembleForecast,
    ) -> CommitteeVerdict:
        latest = dataset.latest_X.iloc[0]

        quant = self._quant_ml_strategist(forecast)
        tech = self._technical_analyst(latest)
        fund = self._fundamental_analyst(bundle)
        macro_news, news_score = self._macro_flow_issue_analyst(bundle, latest)
        risk, risk_level, vol_cap = self._risk_manager(dataset, forecast)

        opinions = [quant, tech, fund, macro_news, risk]

        # 가중 종합 점수: 퀀트(30%) + 기업재정(24%) + 거시·이슈(22%) + 기술적(19%) + 리스크(5%)
        composite = (
            0.30 * quant.score
            + 0.24 * fund.score
            + 0.22 * macro_news.score
            + 0.19 * tech.score
            + 0.05 * risk.score
        )
        composite = float(np.clip(composite, -100.0, 100.0))

        if composite >= 40.0 and forecast.up_probability >= 0.55:
            action = "강력 매수"
        elif composite >= 15.0 and forecast.up_probability >= 0.51:
            action = "매수"
        elif composite <= -40.0 and forecast.up_probability <= 0.45:
            action = "강력 매도"
        elif composite <= -15.0 and forecast.up_probability <= 0.49:
            action = "매도"
        else:
            action = "관망(중립)"

        # [실전 학습 규칙] 당일/단기 과열 급등주 추격매수 원천 차단 (Anti-Chasing Filter)
        ret_1d = float(np.expm1(float(latest.get("ret_1d", 0.0))))
        ret_5d = float(np.expm1(float(latest.get("ret_5d", 0.0))))
        rsi_7 = float(latest.get("rsi_7", 0.5)) * 100.0
        overheated = (ret_1d >= 0.085) or (ret_5d >= 0.20 and rsi_7 >= 75.0) or (rsi_7 >= 82.0)
        overheat_note = ""
        if overheated and "매수" in action:
            action = "관망(과열대기)"
            overheat_note = (
                f"단기 급등 과열 차단(1일 {ret_1d*100:+.1f}% · 5일 {ret_5d*100:+.1f}% · RSI7 {rsi_7:.0f}) — "
                f"고점 추격매수 대신 눌림목 진입 대기"
            )

        # [자가진화 챔피언 승률 필터] 모의투자 훈련을 통해 검증된 2차 승률 판별기(Meta-AI) 적용
        champ_art = load_champion_artifact()
        win_conf_pct, champ_passed, champ_note = predict_stock_win_confidence(
            dataset.latest_X,
            forecast.expected_log_return,
            forecast.up_probability,
            artifact=champ_art,
        )
        if champ_art is not None:
            if not champ_passed and "매수" in action:
                action = "관망(승률필터 대기)"
            elif champ_passed and win_conf_pct >= 62.0 and composite >= 14.0:
                composite = float(np.clip(composite + 5.0, -100.0, 100.0))
                if composite >= 40.0 and forecast.up_probability >= 0.55 and "과열" not in action:
                    action = "강력 매수"

        if "매수" in action and composite > 5.0 and forecast.up_probability > 0.5:
            raw_pos = np.clip((composite / 72.0) * 100.0, 10.0, 100.0)
            recommended_pos = float(np.round(min(raw_pos, vol_cap), 1))
        else:
            recommended_pos = 0.0

        # Bull / Bear 토론 논거: 재무 강점/위험 + 이슈 촉매/악재 우선 배치
        bull_case: list[str] = []
        bear_case: list[str] = []
        if overheat_note:
            bear_case.append(overheat_note)
        if champ_art is not None:
            if champ_passed:
                bull_case.append(champ_note)
            else:
                bear_case.append(champ_note)

        if bundle.fundamental_profile:
            bull_case.extend(bundle.fundamental_profile.strengths[:2])
            bear_case.extend(bundle.fundamental_profile.red_flags[:2])
        if bundle.news_report:
            bull_case.extend(bundle.news_report.key_catalysts[:2])
            bear_case.extend(bundle.news_report.key_risks[:2])
        if bundle.alt_signal:
            for b_txt in bundle.alt_signal.easy_bullets_ko[:2]:
                if bundle.alt_signal.combined_signal >= 0.08:
                    bull_case.append(b_txt)
                elif bundle.alt_signal.combined_signal <= -0.08:
                    bear_case.append(b_txt)

        for op in opinions:
            if op.score >= 12 and op.reasons:
                bull_case.append(op.reasons[0])
            elif op.score <= -12 and op.reasons:
                bear_case.append(op.reasons[0])

        # 중복 제거
        bull_case = list(dict.fromkeys(bull_case))[:5]
        bear_case = list(dict.fromkeys(bear_case))[:5]
        if not bull_case:
            bull_case.append("뚜렷한 단기 촉매보다는 점진적 바닥 다지기 구간입니다.")
        if not bear_case:
            bear_case.append("단기 급격한 악재 신호는 제한적이나 시장 변동성에 유의해야 합니다.")

        cio_memo = self._build_rule_cio_memo(
            bundle, forecast, action, composite, recommended_pos, risk_level, bull_case, bear_case
        )
        llm_enhanced = False

        if self.use_llm and os.getenv("OPENAI_API_KEY"):
            llm_text = self._try_llm_cio_synthesis(
                bundle, forecast, opinions, action, composite, recommended_pos, risk_level
            )
            if llm_text:
                cio_memo = llm_text
                llm_enhanced = True

        return CommitteeVerdict(
            action=action,
            composite_score=composite,
            recommended_position_pct=recommended_pos,
            risk_level=risk_level,
            opinions=opinions,
            bull_case=bull_case,
            bear_case=bear_case,
            news_sentiment_score=news_score,
            cio_memo=cio_memo,
            llm_enhanced=llm_enhanced,
            win_confidence_pct=win_conf_pct,
            champion_filter_passed=champ_passed,
            champion_filter_note=champ_note,
        )

    def _technical_analyst(self, latest: "pd.Series") -> AgentOpinion:
        score = 0.0
        reasons: list[str] = []

        sma5_20 = float(latest.get("sma_5_20_cross", 0.0))
        sma20_60 = float(latest.get("sma_20_60_cross", 0.0))
        rsi14 = float(latest.get("rsi_14", 0.5)) * 100.0
        macd_hist = float(latest.get("macd_hist_pct", 0.0))
        bb_z = float(latest.get("bb_zscore_20", 0.0))
        adx = float(latest.get("adx_14", 0.2)) * 100.0

        if sma5_20 > 0 and sma20_60 > 0:
            score += 32.0
            reasons.append("이동평균선 정배열(5일>20일>60일)로 상승 추세가 유지되고 있습니다.")
        elif sma5_20 < 0 and sma20_60 < 0:
            score -= 32.0
            reasons.append("이동평균선 역배열(5일<20일<60일)로 하락 압력이 우세합니다.")
        else:
            score += 12.0 if sma5_20 > 0 else -12.0
            reasons.append("단기·중기 이동평균선이 교차하며 추세 전환을 모색 중입니다.")

        if macd_hist > 0:
            score += 18.0
            reasons.append("MACD 히스토그램 양(+)전환으로 단기 매수 모멘텀이 강화 중입니다.")
        else:
            score -= 18.0
            reasons.append("MACD 히스토그램 음(-)의 영역으로 모멘텀 둔화 신호가 나타납니다.")

        if rsi14 >= 72:
            score -= 22.0
            reasons.append(f"RSI(14)={rsi14:.1f} 과매수 구간으로 단기 차익실현 압력에 유의해야 합니다.")
        elif rsi14 <= 30:
            score += 20.0
            reasons.append(f"RSI(14)={rsi14:.1f} 과매도 영역으로 기술적 반등 여력이 있습니다.")
        else:
            score += (rsi14 - 50.0) * 0.4
            reasons.append(f"RSI(14)는 {rsi14:.1f}로 중립 구간입니다.")

        if bb_z > 2.0:
            score -= 12.0
        elif bb_z < -1.8:
            score += 12.0
        if adx > 25:
            reasons.append(f"ADX={adx:.1f}로 현재 방향성의 추세 강도가 뚜렷합니다.")

        score = float(np.clip(score, -100.0, 100.0))
        stance = _score_to_stance(score)
        return AgentOpinion(
            role="기술적 분석가",
            persona="차트·모멘텀 트레이더",
            stance=stance,
            score=score,
            summary=f"추세·오실레이터 {stance} ({score:+.0f}점, RSI {rsi14:.1f})",
            reasons=reasons,
        )

    def _fundamental_analyst(self, bundle: MarketDataBundle) -> AgentOpinion:
        prof = bundle.fundamental_profile
        if prof is None or prof.data_completeness <= 0:
            return AgentOpinion(
                role="기업 재정·실적 분석가",
                persona="워런 버핏 · 재무제표 & 어닝 분석",
                stance="중립",
                score=0.0,
                summary="기업 재무 데이터 미제공(ETF 또는 조회 제한)으로 중립 평가",
                reasons=["세부 재무지표가 제공되지 않아 가격·수급·이슈 중심으로 판단해야 합니다."],
            )

        # 0~100 점수를 -100 ~ +100 스케일로 변환
        score = float(np.clip((prof.overall_score - 50.0) * 2.2, -100.0, 100.0))
        stance = _score_to_stance(score)
        reasons = list(prof.strengths[:3]) + list(prof.red_flags[:2])
        if not reasons:
            reasons.append(
                f"밸류({prof.valuation_score:.0f}) · 수익성({prof.profitability_score:.0f}) · "
                f"성장({prof.growth_score:.0f}) · 재무건전성({prof.health_score:.0f}) · 실적({prof.earnings_score:.0f})"
            )
        return AgentOpinion(
            role="기업 재정·실적 분석가",
            persona="워런 버핏 · 재무제표 & 어닝 분석",
            stance=stance,
            score=score,
            summary=f"재무등급 {prof.grade} ({prof.overall_score:.0f}/100점, 스탠스 {stance})",
            reasons=reasons,
        )

    def _macro_flow_issue_analyst(
        self, bundle: MarketDataBundle, latest: "pd.Series"
    ) -> tuple[AgentOpinion, float]:
        score = 0.0
        reasons: list[str] = []

        vix = float(latest.get("vix_level", 0.18)) * 100.0
        rel_alpha_20 = float(latest.get("rel_alpha_20d", 0.0)) * 100.0
        us10y_diff = float(latest.get("us10y_diff_20d", 0.0)) * 100.0
        fx_ret_20 = float(latest.get("usdkrw_ret_20d", 0.0)) * 100.0

        if vix < 16:
            score += 12.0
            reasons.append(f"VIX 공포지수({vix:.1f}) 안정 구간으로 위험자산 선호 환경입니다.")
        elif vix > 25:
            score -= 18.0
            reasons.append(f"VIX 공포지수({vix:.1f}) 상승으로 글로벌 변동성 경계가 필요합니다.")

        if rel_alpha_20 > 3.0:
            score += 16.0
            reasons.append(f"최근 20일 시장 대비 +{rel_alpha_20:.1f}% 초과성과를 기록하는 주도주 흐름입니다.")
        elif rel_alpha_20 < -3.0:
            score -= 16.0
            reasons.append(f"최근 20일 시장 대비 {rel_alpha_20:.1f}% 부진한 상대 약세입니다.")

        if us10y_diff > 0.35:
            score -= 10.0
            reasons.append(f"미국 10년물 국채금리 20일 +{us10y_diff:.2f}%p 상승으로 밸류에이션 부담.")
        elif us10y_diff < -0.25:
            score += 8.0

        if bundle.symbol.market == "KR":
            if fx_ret_20 > 2.5:
                score -= 10.0
                reasons.append(f"원/달러 환율 +{fx_ret_20:.1f}% 급등으로 외국인 수급에 부담 요인입니다.")
            elif fx_ret_20 < -1.5:
                score += 8.0
            f_flow = latest.get("kr_foreign_flow_5d", None)
            if f_flow is not None and not np.isnan(f_flow):
                if float(f_flow) > 0.05:
                    score += 14.0
                    reasons.append("최근 5거래일 외국인 순매수가 유입되고 있습니다.")
                elif float(f_flow) < -0.05:
                    score -= 14.0
                    reasons.append("최근 5거래일 외국인 순매도 출회가 부담입니다.")

        # 심층 뉴스·이벤트 분석(NewsIntelligenceReport) 반영 (강한 비중)
        nr = bundle.news_report
        news_score = nr.sentiment_score if nr else 0.0
        if nr and nr.article_count > 0:
            issue_signal = 0.55 * nr.sentiment_score + 0.45 * nr.event_shock_score
            score += issue_signal * 55.0
            reasons.append(
                f"최신 이슈 {nr.article_count}건 분석(주력 테마: {nr.dominant_category}) — "
                f"긍정 {nr.positive_count} / 부정 {nr.negative_count} / 중립 {nr.neutral_count} (감성 {nr.sentiment_score:+.2f})"
            )
            if nr.llm_summary:
                reasons.append(f"LLM 이슈 요약: {nr.llm_summary}")
            elif nr.key_catalysts:
                reasons.append(f"주요 호재: {nr.key_catalysts[0]}")
            elif nr.key_risks:
                reasons.append(f"주요 악재: {nr.key_risks[0]}")

        # 폴리마켓 예측시장 + 레딧 민심 + 내부자/외인기관 큰손 수급 반영
        alt = bundle.alt_signal
        if alt is not None:
            score += alt.combined_signal * 42.0
            reasons.extend(alt.easy_bullets_ko[:2])

        if not reasons:
            reasons.append("거시경제 및 최신 이슈가 전반적으로 중립 범위입니다.")

        score = float(np.clip(score, -100.0, 100.0))
        stance = _score_to_stance(score)
        return (
            AgentOpinion(
                role="이슈·뉴스·거시 분석가",
                persona="이벤트 드리븐 & 매크로 리서처",
                stance=stance,
                score=score,
                summary=f"이슈·거시 종합 {stance} ({score:+.0f}점, 뉴스감성 {news_score:+.2f}, VIX {vix:.1f})",
                reasons=reasons,
            ),
            float(news_score),
        )

    def _quant_ml_strategist(self, forecast: EnsembleForecast) -> AgentOpinion:
        ret_score = np.clip(forecast.expected_pct_return * 100.0 * 9.0, -55.0, 55.0)
        prob_score = (forecast.up_probability - 0.5) * 140.0
        agree_boost = (forecast.model_agreement - 0.5) * 20.0 * (1 if forecast.up_probability >= 0.5 else -1)
        score = float(np.clip(ret_score + prob_score + agree_boost, -100.0, 100.0))
        stance = _score_to_stance(score)

        best_model = max(forecast.model_metrics, key=lambda m: m.weight)
        tags = []
        if forecast.cross_sectional_used:
            tags.append("다중종목 통합알파 결합")
        if forecast.memory_calibrated:
            tags.append("실전채점 자가보정")
        tag_str = f" ({', '.join(tags)})" if tags else ""

        reasons = [
            f"하이브리드 예측{tag_str}: 향후 {forecast.horizon}거래일 기대수익률 {forecast.expected_pct_return * 100:+.2f}%, "
            f"상승확률 {forecast.up_probability * 100:.1f}%.",
            f"참여 모델 {len(forecast.model_metrics)}개 중 {forecast.model_agreement * 100:.0f}%가 동일 방향에 합의했습니다.",
            f"핵심 가중 모델: '{best_model.display_name}' (비중 {best_model.weight * 100:.1f}%, OOF 적중률 {best_model.direction_accuracy * 100:.1f}%).",
        ]
        return AgentOpinion(
            role="AI 퀀트 예측가",
            persona="통합 알파 + ML/DL 앙상블",
            stance=stance,
            score=score,
            summary=f"{forecast.horizon}일 예측 {forecast.expected_pct_return * 100:+.2f}% · 상승확률 {forecast.up_probability * 100:.1f}%",
            reasons=reasons,
        )

    def _risk_manager(
        self, dataset: FeatureDataset, forecast: EnsembleForecast
    ) -> tuple[AgentOpinion, str, float]:
        rv20 = dataset.latest_rv20 * 100.0
        atr_pct = dataset.latest_atr_pct * 100.0
        dd252 = float(dataset.latest_X.iloc[0].get("drawdown_252d", 0.0)) * 100.0
        downside_pct = (forecast.target_price_low / forecast.current_price - 1.0) * 100.0

        if rv20 >= 55:
            risk_level, vol_cap, score = "매우 높음", 35.0, -45.0
        elif rv20 >= 38:
            risk_level, vol_cap, score = "높음", 55.0, -22.0
        elif rv20 >= 22:
            risk_level, vol_cap, score = "보통", 80.0, 0.0
        else:
            risk_level, vol_cap, score = "낮음", 100.0, 18.0

        reasons = [
            f"최근 20일 연환산 변동성 {rv20:.1f}%(일평균 ATR {atr_pct:.2f}%) → 리스크 등급 '{risk_level}'.",
            f"향후 {forecast.horizon}거래일 80% 신뢰구간 하방 리스크는 {downside_pct:+.2f}% 수준입니다.",
            f"52주 고점 대비 낙폭 {dd252:.1f}%, 변동성 기준 최대 권장 비중은 {vol_cap:.0f}%입니다.",
        ]
        opinion = AgentOpinion(
            role="리스크 매니저",
            persona="변동성·하방 위험 통제관",
            stance=_score_to_stance(score),
            score=score,
            summary=f"위험도 '{risk_level}' (연 변동성 {rv20:.1f}%, 최대 권장 비중 {vol_cap:.0f}%)",
            reasons=reasons,
        )
        return opinion, risk_level, vol_cap

    @staticmethod
    def _build_rule_cio_memo(
        bundle: MarketDataBundle,
        forecast: EnsembleForecast,
        action: str,
        composite: float,
        rec_pos: float,
        risk_level: str,
        bull_case: list[str],
        bear_case: list[str],
    ) -> str:
        cur = "원" if bundle.symbol.currency == "KRW" else "USD"
        fmt_price = (
            f"{forecast.target_price_median:,.0f}{cur}"
            if cur == "원"
            else f"${forecast.target_price_median:,.2f}"
        )
        bd = forecast.signal_breakdown
        bd_str = ""
        if bd:
            bd_str = (
                f"\n• 예측 분해: 단일ML({bd.get('단일종목_ML_DL_앙상블(%)', 0):+.2f}%) "
                f"+ 통합알파({bd.get('다중종목_통합알파_보정(%)', 0):+.2f}%) "
                f"+ 기업재무({bd.get('기업_재정실적_보정(%)', 0):+.2f}%) "
                f"+ 실시간이슈({bd.get('실시간_이슈뉴스_보정(%)', 0):+.2f}%) "
                f"+ 예측시장·큰손({bd.get('예측시장_큰손민심_보정(%)', 0):+.2f}%)"
            )
        return (
            f"[CIO 최종 종합 의견: {action} | 종합점수 {composite:+.1f}/100]\n"
            f"{bundle.symbol.name}({bundle.symbol.code}) 5인 위원회 심사 결과, 향후 {forecast.horizon}거래일 예상 목표가는 "
            f"{fmt_price}({forecast.expected_pct_return * 100:+.2f}%), 상승 확률은 {forecast.up_probability * 100:.1f}%입니다."
            f"{bd_str}\n"
            f"• 핵심 강세/촉매 근거: {bull_case[0]}\n"
            f"• 핵심 리스크 요인: {bear_case[0]}\n"
            f"• 리스크 등급 '{risk_level}' 기준 권장 비중은 {rec_pos:.0f}%이며, 손절 기준선은 {forecast.target_price_low:,.2f} 이탈 여부로 관리하세요."
        )

    def _try_llm_cio_synthesis(
        self,
        bundle: MarketDataBundle,
        forecast: EnsembleForecast,
        opinions: list[AgentOpinion],
        action: str,
        composite: float,
        rec_pos: float,
        risk_level: str,
    ) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI()
            opinions_text = "\n".join(
                f"- {op.role}({op.stance}, {op.score:+.0f}점): {op.summary} / 근거: {'; '.join(op.reasons[:2])}"
                for op in opinions
            )
            news_text = "\n".join(f"- {n['title']}" for n in bundle.news[:6]) or "최신 뉴스 없음"
            prompt = (
                f"당신은 글로벌 헤지펀드의 최고투자책임자(CIO)입니다. 아래 5인 애널리스트 보고서(재무·이슈·차트·퀀트)와 최신 뉴스를 종합해 "
                f"{bundle.symbol.name}({bundle.symbol.code}, {bundle.symbol.market})의 핵심 한국어 투자 메모(5~7문장)를 작성하세요.\n"
                f"정량 결과: 최종의견={action}, 종합점수={composite:+.1f}, {forecast.horizon}일 기대수익률={forecast.expected_pct_return*100:+.2f}%, "
                f"상승확률={forecast.up_probability*100:.1f}%, 위험등급={risk_level}, 권장비중={rec_pos:.0f}%, "
                f"예측기여도={forecast.signal_breakdown}.\n\n"
                f"[애널리스트 의견]\n{opinions_text}\n\n[최신 뉴스]\n{news_text}"
            )
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=480,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None
