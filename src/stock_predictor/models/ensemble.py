"""동적 성과 가중 앙상블 + Cross-Sectional 통합 알파 + 기업 재무/실시간 이슈 융합 예측 관리자.

예측 융합 구조:
1. Purged Walk-Forward 단일 종목 5종 앙상블 (LightGBM, XGBoost, Ridge, ExtraTrees, Temporal Neural)
2. (학습 완료 시) 다중 종목 통합 Cross-Sectional 알파 모델 결합 (시장 대비 초과수익 신호)
3. (옵션) SQLite 실전 채점 기록 기반 모델 가중치 온라인 보정 (Self-Learning)
4. (옵션) 기업 재정·실적 퀄리티 오버레이 (FundamentalProfile) + 실시간 이슈/뉴스 이벤트 쇼크 오버레이
   -> 변동성 스케일(sigma_h)에 비례해 안전 범위 내에서 반영하고, 각 축의 기여도를 투명하게 분해(Attribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import PredictorConfig
from ..data.alt_intelligence import StockAltSignal
from ..data.features import FeatureDataset
from ..data.fundamentals import FundamentalProfile
from ..data.news_intelligence import NewsIntelligenceReport
from .base import BaseStockModel
from .cross_sectional import CrossSectionalArtifact, predict_single_with_artifact
from .deep_ts import ChronosFoundationAdapter, TemporalNeuralStockModel
from .tabular import (
    ExtraTreesStockModel,
    LightGBMStockModel,
    LinearQuantModel,
    XGBoostStockModel,
)


@dataclass
class ModelMetrics:
    """단일 모델의 Out-of-Sample 검증 성과 지표."""

    name: str
    display_name: str
    weight: float
    direction_accuracy: float
    rank_ic: float
    mae: float
    rmse: float
    latest_expected_return: float
    latest_up_prob: float
    latest_target_price: float


@dataclass
class EnsembleForecast:
    """최종 앙상블 예측 및 Out-of-Sample 검증 시계열 묶음."""

    expected_log_return: float
    expected_pct_return: float
    up_probability: float
    current_price: float
    target_price_median: float
    target_price_low: float          # 10% 분위 (하방 리스크 구간)
    target_price_high: float         # 90% 분위 (상방 잠재 구간)
    model_agreement: float           # 모델 간 방향 합의도 (0~1)
    confidence_score: float          # 종합 신뢰도 (0~100)
    horizon: int
    forecast_path: pd.DataFrame      # 일별 예측 경로 (day_offset, median, low, high)
    model_metrics: list[ModelMetrics]
    oof_predictions: pd.DataFrame    # Out-of-Sample 시계열 (actual_ret, pred_ret, pred_prob, close)
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    chronos_used: bool = False
    # v2 투명한 예측 기여도 분해(Attribution)
    cross_sectional_used: bool = False
    memory_calibrated: bool = False
    signal_breakdown: dict[str, float] = field(default_factory=dict)


def _logit(p: float) -> float:
    p_c = float(np.clip(p, 0.02, 0.98))
    return float(np.log(p_c / (1.0 - p_c)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -12.0, 12.0))))


class EnsembleModelManager:
    """다중 이질적 모델 + Cross-Sectional 알파 + 재무/이슈 하이브리드 앙상블 매니저."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def _create_models(self) -> list[BaseStockModel]:
        models: list[BaseStockModel] = [
            LightGBMStockModel(random_state=self.config.random_seed),
            XGBoostStockModel(random_state=self.config.random_seed + 1),
            LinearQuantModel(random_state=self.config.random_seed + 2),
            ExtraTreesStockModel(random_state=self.config.random_seed + 3),
        ]
        if self.config.use_deep_learning:
            models.append(TemporalNeuralStockModel(random_state=self.config.random_seed + 4))
        return models

    def _build_walk_forward_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Purged Expanding Walk-Forward 인덱스 분할 생성 (학습과 검증 사이에 horizon일 Embargo 적용)."""
        embargo = max(self.config.horizon, 1)
        min_train = min(self.config.min_train_rows, max(int(n_samples * 0.45), 80))
        available_test = n_samples - min_train - embargo
        if available_test < 30:
            split_point = max(int(n_samples * 0.7), 60)
            train_end = max(split_point - embargo, 40)
            return [(np.arange(0, train_end), np.arange(split_point, n_samples))]

        n_splits = max(2, min(self.config.n_splits, available_test // 25))
        fold_size = available_test // n_splits
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_splits):
            val_start = min_train + embargo + k * fold_size
            val_end = n_samples if k == n_splits - 1 else min(val_start + fold_size, n_samples)
            train_end = val_start - embargo
            if train_end >= 60 and val_end - val_start >= 10:
                splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
        return splits

    def fit_validate_and_predict(
        self,
        dataset: FeatureDataset,
        full_close: pd.Series,
        cs_artifact: CrossSectionalArtifact | None = None,
        fundamental_profile: FundamentalProfile | None = None,
        news_report: NewsIntelligenceReport | None = None,
        alt_signal: StockAltSignal | None = None,
        calibration_multipliers: dict[str, float] | None = None,
    ) -> EnsembleForecast:
        """1) Purged Walk-Forward OOF 검증 -> 2) 동적·자가보정 가중치 -> 3) 통합 알파/재무/이슈 융합 예측."""
        X = dataset.train_X
        y_ret = dataset.train_y_ret
        y_dir = dataset.train_y_dir
        splits = self._build_walk_forward_splits(len(X))

        model_specs = self._create_models()
        model_names = [m.name for m in model_specs]
        display_map = {m.name: m.display_name for m in model_specs}

        oof_ret: dict[str, pd.Series] = {
            name: pd.Series(np.nan, index=X.index, dtype=float) for name in model_names
        }
        oof_prob: dict[str, pd.Series] = {
            name: pd.Series(np.nan, index=X.index, dtype=float) for name in model_names
        }

        # 1. Purged Walk-Forward Out-of-Sample 검증
        for train_idx, val_idx in splits:
            X_tr, y_r_tr, y_d_tr = X.iloc[train_idx], y_ret.iloc[train_idx], y_dir.iloc[train_idx]
            X_va = X.iloc[val_idx]
            fold_models = self._create_models()
            for model in fold_models:
                model.fit(X_tr, y_r_tr, y_d_tr)
                pred = model.predict(X_va)
                oof_ret[model.name].iloc[val_idx] = pred.expected_return
                oof_prob[model.name].iloc[val_idx] = pred.up_prob

        # 2. 모델별 OOF 성과 평가 & Softmax 동적 가중치 산출
        raw_scores: dict[str, float] = {}
        eval_stats: dict[str, dict[str, float]] = {}

        for name in model_names:
            valid = oof_ret[name].notna()
            actual_r = y_ret.loc[valid].to_numpy(dtype=float)
            actual_d = y_dir.loc[valid].to_numpy(dtype=int)
            pred_r = oof_ret[name].loc[valid].to_numpy(dtype=float)
            pred_p = oof_prob[name].loc[valid].to_numpy(dtype=float)

            dir_acc = float(np.mean((pred_p >= 0.5).astype(int) == actual_d)) if len(actual_d) > 0 else 0.5
            if len(actual_r) > 5 and np.std(pred_r) > 1e-12:
                ic_val = float(spearmanr(pred_r, actual_r).statistic)
                if np.isnan(ic_val):
                    ic_val = 0.0
            else:
                ic_val = 0.0
            mae = float(np.mean(np.abs(actual_r - pred_r))) if len(actual_r) > 0 else 0.05
            rmse = float(np.sqrt(np.mean((actual_r - pred_r) ** 2))) if len(actual_r) > 0 else 0.07

            score = (dir_acc - 0.5) * 3.0 + ic_val * 2.5 - rmse * 1.0
            raw_scores[name] = score
            eval_stats[name] = {
                "direction_accuracy": dir_acc,
                "rank_ic": ic_val,
                "mae": mae,
                "rmse": rmse,
            }

        score_vals = np.array([raw_scores[n] for n in model_names], dtype=float)
        score_vals = score_vals - np.max(score_vals)
        soft_w = np.exp(score_vals * 2.0)
        soft_w = soft_w / np.sum(soft_w)
        uniform_w = np.ones_like(soft_w) / len(soft_w)
        blended_w = 0.65 * soft_w + 0.35 * uniform_w
        weights = {n: float(w) for n, w in zip(model_names, blended_w)}

        # SQLite 실전 채점 기록 기반 온라인 가중치 보정 (Self-Learning)
        memory_calibrated = False
        if self.config.use_memory_calibration and calibration_multipliers:
            applied = False
            for n in list(weights.keys()):
                if n in calibration_multipliers:
                    weights[n] *= float(calibration_multipliers[n])
                    applied = True
            if applied:
                w_total = sum(weights.values()) + 1e-9
                weights = {n: v / w_total for n, v in weights.items()}
                memory_calibrated = True

        # 백테스트용 OOF 시계열 (누수 방지를 위해 균등 앙상블 사용)
        valid_any = oof_ret[model_names[0]].notna()
        oof_df = pd.DataFrame(index=X.index[valid_any])
        oof_df["close"] = dataset.train_close.loc[oof_df.index]
        oof_df["future_close"] = dataset.train_future_close.loc[oof_df.index]
        oof_df["actual_ret"] = y_ret.loc[oof_df.index]
        oof_df["actual_dir"] = y_dir.loc[oof_df.index]

        ret_matrix = np.column_stack([oof_ret[n].loc[oof_df.index].to_numpy() for n in model_names])
        prob_matrix = np.column_stack([oof_prob[n].loc[oof_df.index].to_numpy() for n in model_names])
        oof_df["pred_ret"] = np.nanmean(ret_matrix, axis=1)
        oof_df["pred_prob"] = np.nanmean(prob_matrix, axis=1)
        oof_df["pred_std"] = np.nanstd(ret_matrix, axis=1)

        # 3. 전체 가용 학습 데이터로 최종 단일종목 모델 적합 & 최신 예측
        final_models = self._create_models()
        latest_preds: dict[str, tuple[float, float]] = {}
        imp_accumulator: dict[str, float] = {f: 0.0 for f in dataset.feature_names}
        imp_weight_sum = 0.0

        for model in final_models:
            model.fit(X, y_ret, y_dir)
            pred = model.predict(dataset.latest_X)
            latest_preds[model.name] = (float(pred.expected_return[0]), float(pred.up_prob[0]))
            f_imp = model.get_feature_importance(dataset.feature_names)
            if f_imp:
                w = weights[model.name]
                for fname, val in f_imp.items():
                    imp_accumulator[fname] = imp_accumulator.get(fname, 0.0) + w * val
                imp_weight_sum += w

        # 단일 종목 5종 앙상블 기본 예측치 (기여도 추적용)
        single_base_log_ret = sum(weights[n] * latest_preds[n][0] for n in model_names)

        # 4. 다중 종목 통합 Cross-Sectional 알파 모델 결합 (Qlib 핵심 엔진)
        cross_sectional_used = False
        cs_delta_log_ret = 0.0
        if self.config.use_cross_sectional and cs_artifact is not None:
            # 벤치마크 최근 추세 드리프트(보수적 감쇠)와 통합 알파 합산
            bm_ret20 = float(dataset.latest_X.iloc[0].get("bm_ret_20d", 0.0))
            bm_drift = float(np.clip(bm_ret20 * (dataset.horizon / 20.0) * 0.25, -0.02, 0.02))
            cs_out = predict_single_with_artifact(cs_artifact, dataset.latest_X, bm_expected_drift=bm_drift)

            cs_name = "cross_sectional_alpha"
            # 통합 알파 모델의 OOF Rank IC가 양수(+)일 때만 높은 비중(25~42%) 부여, 약하면 자동 축소(10%)
            cs_ic = cs_artifact.metrics.mean_daily_rank_ic
            if cs_ic > 0.005:
                cs_weight = float(np.clip(0.28 + cs_ic * 2.5, 0.25, 0.42))
            else:
                cs_weight = float(np.clip(0.12 + cs_ic * 2.0, 0.08, 0.15))
            for k in list(weights.keys()):
                weights[k] *= (1.0 - cs_weight)
            weights[cs_name] = cs_weight
            display_map[cs_name] = f"Cross-Sectional Alpha ({cs_artifact.metrics.n_symbols}종목 통합)"
            latest_preds[cs_name] = (cs_out["expected_total_log_return"], cs_out["outperform_prob"])
            eval_stats[cs_name] = {
                "direction_accuracy": cs_artifact.metrics.excess_dir_accuracy,
                "rank_ic": cs_artifact.metrics.mean_daily_rank_ic,
                "mae": float(np.mean(np.abs(oof_df["actual_ret"]))),
                "rmse": float(np.sqrt(np.mean(oof_df["actual_ret"] ** 2))),
            }
            model_names.append(cs_name)
            cross_sectional_used = True

        # 선택적 Amazon Chronos 시계열 파운데이션 모델 결합
        chronos_used = False
        if self.config.use_chronos:
            chronos = ChronosFoundationAdapter()
            c_res = chronos.forecast_close_series(full_close, horizon=dataset.horizon)
            if c_res is not None:
                chronos_used = True
                for k in list(weights.keys()):
                    weights[k] *= 0.85
                weights[chronos.name] = 0.15
                display_map[chronos.name] = chronos.display_name
                latest_preds[chronos.name] = (c_res["expected_return"], c_res["up_prob"])
                eval_stats[chronos.name] = {
                    "direction_accuracy": 0.5,
                    "rank_ic": 0.0,
                    "mae": float(np.mean(np.abs(oof_df["actual_ret"]))),
                    "rmse": float(np.sqrt(np.mean(oof_df["actual_ret"] ** 2))),
                }
                model_names.append(chronos.name)

        # 5. 모델 앙상블 순수 예측값 산출
        model_ens_log_ret = sum(weights[n] * latest_preds[n][0] for n in model_names)
        model_ens_up_prob = sum(weights[n] * latest_preds[n][1] for n in model_names)
        if cross_sectional_used:
            cs_delta_log_ret = model_ens_log_ret - single_base_log_ret

        # 잔차 변동성 + 최근 실현변동성으로 종목의 H일 표준편차(sigma_h) 산출
        oof_residuals = (oof_df["actual_ret"] - oof_df["pred_ret"]).dropna()
        resid_std = float(oof_residuals.std()) if len(oof_residuals) > 10 else 0.04
        horizon_vol = dataset.latest_rv20 * np.sqrt(dataset.horizon / 252.0)
        pred_spread = float(np.std([latest_preds[n][0] for n in model_names]))
        sigma_h = max(0.55 * resid_std + 0.45 * horizon_vol, pred_spread, 0.01)

        # 6. 기업 재정·실적 퀄리티 & 실시간 이슈·뉴스 이벤트 오버레이 반영
        # (변동성 스케일 sigma_h에 비례해 합리적 범위 안에서만 보정, Log-Odds 확률 업데이트)
        fund_log_adj = 0.0
        fund_logit_adj = 0.0
        if (
            self.config.use_fundamental_overlay
            and fundamental_profile is not None
            and fundamental_profile.data_completeness > 0
        ):
            # 기간이 길수록(10~20일) 재무 펀더멘털 영향력 증가, 단기(1~5일)는 완만 반영
            horizon_scale = float(np.clip(np.sqrt(dataset.horizon / 5.0), 0.55, 1.6))
            fund_sig = fundamental_profile.fundamental_signal  # -1 ~ +1
            fund_log_adj = float(np.clip(fund_sig * sigma_h * 0.22 * horizon_scale, -0.04, 0.04))
            fund_logit_adj = float(fund_sig * 0.35 * horizon_scale)

        news_log_adj = 0.0
        news_logit_adj = 0.0
        if (
            self.config.use_news_overlay
            and news_report is not None
            and news_report.article_count > 0
        ):
            # 이슈/뉴스 충격은 단기(1~5일)에 강하게 반영, 고임팩트 이벤트(실적/수주/유증 등) 가중
            news_combined = float(
                np.clip(0.55 * news_report.sentiment_score + 0.45 * news_report.event_shock_score, -1.0, 1.0)
            )
            # 기사 수가 너무 적을 때(1~2건) 과도한 변동 방지
            coverage = min(news_report.article_count / 5.0, 1.0)
            short_scale = float(np.clip(np.sqrt(5.0 / max(dataset.horizon, 1)), 0.65, 1.3))
            news_log_adj = float(np.clip(news_combined * coverage * sigma_h * 0.26 * short_scale, -0.05, 0.05))
            news_logit_adj = float(news_combined * coverage * 0.45 * short_scale)

        alt_log_adj = 0.0
        alt_logit_adj = 0.0
        if self.config.use_alt_overlay and alt_signal is not None:
            # 폴리마켓 예측시장 + 레딧 민심 + 임원/외인기관 큰손 수급 신호 반영
            alt_sig = float(np.clip(alt_signal.combined_signal, -1.0, 1.0))
            alt_log_adj = float(np.clip(alt_sig * sigma_h * 0.24, -0.04, 0.04))
            alt_logit_adj = float(alt_sig * 0.40)

        ens_log_ret = float(model_ens_log_ret + fund_log_adj + news_log_adj + alt_log_adj)
        ens_up_prob = float(
            np.clip(_sigmoid(_logit(model_ens_up_prob) + fund_logit_adj + news_logit_adj + alt_logit_adj), 0.05, 0.95)
        )

        ens_pct_ret = float(np.expm1(ens_log_ret))
        cur_price = dataset.latest_close
        target_med = cur_price * float(np.exp(ens_log_ret))
        target_low = cur_price * float(np.exp(ens_log_ret - 1.28 * sigma_h))
        target_high = cur_price * float(np.exp(ens_log_ret + 1.28 * sigma_h))

        # 모델 간 합의도
        ens_sign = 1 if ens_up_prob >= 0.5 else -1
        agree_count = sum(1 for n in model_names if (1 if latest_preds[n][1] >= 0.5 else -1) == ens_sign)
        agreement = agree_count / len(model_names)

        # 종합 신뢰도 점수 (0~100)
        oof_acc = float(np.mean((oof_df["pred_prob"] >= 0.5).astype(int) == oof_df["actual_dir"]))
        cs_bonus = 6.0 if (cross_sectional_used and cs_artifact and cs_artifact.metrics.mean_daily_rank_ic > 0.01) else 0.0
        info_bonus = 4.0 if (fundamental_profile and fundamental_profile.data_completeness >= 0.6) else 0.0
        prob_edge = abs(ens_up_prob - 0.5) * 2.0
        raw_conf = 35.0 + (oof_acc - 0.5) * 110.0 + prob_edge * 34.0 + (agreement - 0.5) * 28.0 + cs_bonus + info_bonus
        confidence = float(np.clip(raw_conf, 15.0, 95.0))

        # 일별 예측 경로(Fan Chart)
        future_bdays = pd.bdate_range(start=dataset.latest_date, periods=dataset.horizon + 1)
        path_rows = []
        for step in range(dataset.horizon + 1):
            frac = step / dataset.horizon
            sqrt_frac = np.sqrt(frac)
            path_rows.append(
                {
                    "date": future_bdays[step],
                    "day_offset": step,
                    "median_price": cur_price * float(np.exp(ens_log_ret * frac)),
                    "low_price": cur_price * float(np.exp(ens_log_ret * frac - 1.28 * sigma_h * sqrt_frac)),
                    "high_price": cur_price * float(np.exp(ens_log_ret * frac + 1.28 * sigma_h * sqrt_frac)),
                }
            )
        forecast_path = pd.DataFrame(path_rows)

        metrics_list = [
            ModelMetrics(
                name=n,
                display_name=display_map[n],
                weight=weights[n],
                direction_accuracy=eval_stats[n]["direction_accuracy"],
                rank_ic=eval_stats[n]["rank_ic"],
                mae=eval_stats[n]["mae"],
                rmse=eval_stats[n]["rmse"],
                latest_expected_return=float(np.expm1(latest_preds[n][0])),
                latest_up_prob=latest_preds[n][1],
                latest_target_price=cur_price * float(np.exp(latest_preds[n][0])),
            )
            for n in model_names
        ]

        top_features = (
            sorted(
                [(k, v / imp_weight_sum) for k, v in imp_accumulator.items()],
                key=lambda item: item[1],
                reverse=True,
            )[:15]
            if imp_weight_sum > 0
            else []
        )

        # 각 축별 기대수익률 기여도 Breakdown (% 단위)
        signal_breakdown = {
            "단일종목_ML_DL_앙상블(%)": round(float(np.expm1(single_base_log_ret)) * 100.0, 2),
            "다중종목_통합알파_보정(%)": round(float(cs_delta_log_ret) * 100.0, 2),
            "기업_재정실적_보정(%)": round(float(fund_log_adj) * 100.0, 2),
            "실시간_이슈뉴스_보정(%)": round(float(news_log_adj) * 100.0, 2),
            "예측시장_큰손민심_보정(%)": round(float(alt_log_adj) * 100.0, 2),
            "최종_기대수익률(%)": round(ens_pct_ret * 100.0, 2),
        }

        return EnsembleForecast(
            expected_log_return=ens_log_ret,
            expected_pct_return=ens_pct_ret,
            up_probability=ens_up_prob,
            current_price=cur_price,
            target_price_median=target_med,
            target_price_low=target_low,
            target_price_high=target_high,
            model_agreement=float(agreement),
            confidence_score=confidence,
            horizon=dataset.horizon,
            forecast_path=forecast_path,
            model_metrics=metrics_list,
            oof_predictions=oof_df,
            feature_importance=top_features,
            chronos_used=chronos_used,
            cross_sectional_used=cross_sectional_used,
            memory_calibrated=memory_calibrated,
            signal_breakdown=signal_breakdown,
        )
