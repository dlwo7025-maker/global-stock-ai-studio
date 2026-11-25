"""국내·해외 주식 통합 분석 · 다중종목 알파 학습 · 심층 재무/이슈 융합 · 자가채점 파이프라인."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .agents.committee import CommitteeVerdict, InvestmentCommittee
from .backtest.engine import BacktestResult, WalkForwardBacktester
from .config import KR_UNIVERSE_CODES, PredictorConfig, US_UNIVERSE_TICKERS
from .data.features import FeatureDataset, FeatureEngine
from .data.market_data import MarketDataBundle, MarketDataLoader
from .models.cross_sectional import (
    CrossSectionalAlphaTrainer,
    CrossSectionalArtifact,
    load_cross_sectional_artifact,
)
from .models.ensemble import EnsembleForecast, EnsembleModelManager
from .storage.memory_db import PredictionMemoryDB, ScorecardSummary


@dataclass
class StockAnalysisReport:
    """단일 종목의 통합 분석·예측·백테스트·재무·이슈 최종 리포트."""

    bundle: MarketDataBundle
    dataset: FeatureDataset
    forecast: EnsembleForecast
    backtest: BacktestResult
    verdict: CommitteeVerdict

    def to_summary_dict(self) -> dict[str, object]:
        sym = self.bundle.symbol
        fund_score = self.bundle.fundamental_profile.overall_score if self.bundle.fundamental_profile else 50.0
        fund_grade = self.bundle.fundamental_profile.grade if self.bundle.fundamental_profile else "미제공"
        news_score = self.bundle.news_report.sentiment_score if self.bundle.news_report else 0.0
        news_cnt = self.bundle.news_report.article_count if self.bundle.news_report else 0
        alt_score = self.bundle.alt_signal.combined_signal if self.bundle.alt_signal else 0.0
        alt_summary = (
            " | ".join(self.bundle.alt_signal.easy_bullets_ko[:2])
            if (self.bundle.alt_signal and self.bundle.alt_signal.easy_bullets_ko)
            else "예측시장·큰손 중립"
        )
        return {
            "시장": "국내(KR)" if sym.market == "KR" else "해외(US)",
            "종목명": sym.name,
            "티커": sym.code,
            "통화": sym.currency,
            "기준일": self.dataset.latest_date.strftime("%Y-%m-%d"),
            "현재가": round(self.forecast.current_price, 2),
            f"{self.forecast.horizon}일_예상가": round(self.forecast.target_price_median, 2),
            f"{self.forecast.horizon}일_기대수익률(%)": round(self.forecast.expected_pct_return * 100, 2),
            "상승확률(%)": round(self.forecast.up_probability * 100, 1),
            "재무점수": round(fund_score, 1),
            "재무등급": fund_grade,
            "이슈감성": round(news_score, 2),
            "이슈건수": news_cnt,
            "예측시장_큰손점수": round(alt_score, 2),
            "큰손_한줄요약": alt_summary,
            "실전승률_점수(%)": round(self.verdict.win_confidence_pct, 1),
            "승률필터_통과": "Y" if self.verdict.champion_filter_passed else "N",
            "승률필터_설명": self.verdict.champion_filter_note,
            "통합알파적용": "Y" if self.forecast.cross_sectional_used else "N",
            "하한_10%": round(self.forecast.target_price_low, 2),
            "상한_90%": round(self.forecast.target_price_high, 2),
            "모델합의도(%)": round(self.forecast.model_agreement * 100, 0),
            "신뢰도": round(self.forecast.confidence_score, 1),
            "OOF_방향적중률(%)": round(self.backtest.direction_accuracy * 100, 1),
            "OOF_RankIC": round(self.backtest.rank_ic, 3),
            "백테스트_전략수익률(%)": round(self.backtest.strategy_total_return * 100, 2),
            "백테스트_단순보유(%)": round(self.backtest.benchmark_total_return * 100, 2),
            "전략_Sharpe": round(self.backtest.strategy_sharpe, 2),
            "전략_MDD(%)": round(self.backtest.strategy_mdd * 100, 2),
            "위원회_점수": round(self.verdict.composite_score, 1),
            "최종의견": self.verdict.action,
            "권장비중(%)": round(self.verdict.recommended_position_pct, 1),
            "리스크": self.verdict.risk_level,
            "데이터소스": self.bundle.data_source,
        }


class GlobalStockPipeline:
    """국내·해외 주식 통합 예측 · 다중종목 알파 학습 · 자가채점 파이프라인."""

    def __init__(
        self,
        config: PredictorConfig | None = None,
        memory_db: PredictionMemoryDB | None = None,
    ) -> None:
        self.config = config or PredictorConfig()
        self.loader = MarketDataLoader(
            use_cache=self.config.use_cache,
            cache_ttl_hours=self.config.cache_ttl_hours,
            use_llm_news=self.config.use_llm_agents,
        )
        self.feature_engine = FeatureEngine(horizon=self.config.horizon)
        self.model_manager = EnsembleModelManager(config=self.config)
        self.backtester = WalkForwardBacktester(config=self.config)
        self.committee = InvestmentCommittee(use_llm=self.config.use_llm_agents)
        self.memory_db = memory_db or PredictionMemoryDB()

    def train_universe(
        self,
        market: str = "KR",
        symbols: list[str] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> CrossSectionalArtifact:
        """국내(KR) 또는 해외(US) 다중 종목 유니버스를 통합 수집·학습해 Cross-Sectional 알파 모델 저장."""
        mkt = market.upper()
        target_symbols = symbols or (KR_UNIVERSE_CODES if mkt == "KR" else US_UNIVERSE_TICKERS)
        bundles: list[MarketDataBundle] = []

        for idx, sym in enumerate(target_symbols, start=1):
            if progress_cb:
                progress_cb(f"[{mkt}] 유니버스 가격 데이터 수집 ({idx}/{len(target_symbols)}): {sym}")
            try:
                # 통합 시계열 학습에는 Point-in-Time 가격·거시 데이터만 빠르게 수집 (Look-Ahead 방지 + 고속화)
                b = self.loader.load_bundle(
                    sym, lookback_years=self.config.lookback_years, include_deep_info=False
                )
                bundles.append(b)
            except Exception:
                continue

        trainer = CrossSectionalAlphaTrainer(config=self.config)
        return trainer.train_from_bundles(bundles, market=mkt, save=True, progress_cb=progress_cb)

    def analyze(self, query: str) -> StockAnalysisReport:
        """단일 종목 심층 수집(가격+거시+재무+이슈) -> 하이브리드 예측 -> 백테스트 -> 위원회 -> DB 기록."""
        bundle = self.loader.load_bundle(
            query, lookback_years=self.config.lookback_years, include_deep_info=True
        )
        return self.analyze_bundle(bundle)

    def analyze_bundle(
        self,
        bundle: MarketDataBundle,
        cs_artifact: CrossSectionalArtifact | None = None,
    ) -> StockAnalysisReport:
        """준비된 MarketDataBundle로 분석 수행 (오프라인 테스트 및 커스텀 아티팩트 주입 지원)."""
        dataset = self.feature_engine.build(bundle)

        # 저장된 다중종목 Cross-Sectional 통합 알파 모델 자동 로드
        loaded_cs = cs_artifact
        if loaded_cs is None and self.config.use_cross_sectional:
            loaded_cs = load_cross_sectional_artifact(bundle.symbol.market, self.config.horizon)

        # SQLite 실전 채점 기록 기반 온라인 보정 가중치 조회
        cal_mults = (
            self.memory_db.get_model_calibration_multipliers(bundle.symbol.market)
            if self.config.use_memory_calibration
            else None
        )

        forecast = self.model_manager.fit_validate_and_predict(
            dataset=dataset,
            full_close=bundle.ohlcv["Close"],
            cs_artifact=loaded_cs,
            fundamental_profile=bundle.fundamental_profile,
            news_report=bundle.news_report,
            alt_signal=bundle.alt_signal,
            calibration_multipliers=cal_mults,
        )
        backtest = self.backtester.run(forecast.oof_predictions, market=bundle.symbol.market)
        verdict = self.committee.evaluate(bundle, dataset, forecast)

        # 자가학습용 예측 기록 자동 저장
        if self.config.record_predictions:
            try:
                target_dt = forecast.forecast_path["date"].iloc[-1]
                fund_s = bundle.fundamental_profile.overall_score if bundle.fundamental_profile else 50.0
                news_s = bundle.news_report.sentiment_score if bundle.news_report else 0.0
                m_entries = [
                    (m.name, m.latest_expected_return, m.latest_up_prob)
                    for m in forecast.model_metrics
                ]
                self.memory_db.log_prediction(
                    market=bundle.symbol.market,
                    code=bundle.symbol.code,
                    name=bundle.symbol.name,
                    horizon=forecast.horizon,
                    as_of_date=dataset.latest_date,
                    target_date=target_dt,
                    base_price=forecast.current_price,
                    pred_price=forecast.target_price_median,
                    pred_pct_ret=forecast.expected_pct_return,
                    up_prob=forecast.up_probability,
                    fund_score=fund_s,
                    news_score=news_s,
                    committee_score=verdict.composite_score,
                    action=verdict.action,
                    model_entries=m_entries,
                )
            except Exception:
                pass

        return StockAnalysisReport(
            bundle=bundle,
            dataset=dataset,
            forecast=forecast,
            backtest=backtest,
            verdict=verdict,
        )

    def evaluate_past_predictions(self) -> tuple[int, ScorecardSummary]:
        """만기일이 지난 과거 예측들을 실제 최신 주가로 자동 채점하고 성적표 반환."""
        cache_ohlcv: dict[tuple[str, str], pd.DataFrame] = {}

        def _resolve_price(market: str, code: str, target_date_str: str) -> float | None:
            key = (market, code)
            if key not in cache_ohlcv:
                try:
                    b = self.loader.load_bundle(code, lookback_years=1, include_deep_info=False)
                    cache_ohlcv[key] = b.ohlcv
                except Exception:
                    return None
            df = cache_ohlcv[key]
            target_ts = pd.Timestamp(target_date_str)
            after = df.loc[df.index >= target_ts, "Close"]
            if after.empty:
                return None
            return float(after.iloc[0])

        newly_scored = self.memory_db.evaluate_due_predictions(_resolve_price)
        return newly_scored, self.memory_db.get_scorecard_summary()

    def screen(
        self,
        queries: list[str],
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> tuple[pd.DataFrame, list[StockAnalysisReport], list[tuple[str, str]]]:
        """국내·해외 여러 종목을 일괄 예측하고 위원회 종합 점수 순으로 랭킹."""
        reports: list[StockAnalysisReport] = []
        errors: list[tuple[str, str]] = []
        total = len(queries)
        for idx, q in enumerate(queries, start=1):
            if progress_cb:
                progress_cb(idx, total, q)
            try:
                reports.append(self.analyze(q))
            except Exception as exc:
                errors.append((q, str(exc)))

        if not reports:
            return pd.DataFrame(), reports, errors

        rows = [r.to_summary_dict() for r in reports]
        df = pd.DataFrame(rows).sort_values(
            by=["위원회_점수", f"{self.config.horizon}일_기대수익률(%)"],
            ascending=[False, False],
        ).reset_index(drop=True)
        return df, reports, errors
