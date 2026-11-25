"""정형(Tabular) 퀀트 예측 모델 모음.

- LightGBMStockModel: Qlib 스타일 핵심 GBDT (Huber 손실로 급등락 아웃라이어 완화)
- XGBoostStockModel: 강한 L1/L2 정규화가 적용된 Gradient Boosted Trees
- LinearQuantModel: Winsorizing + RobustScaler + Ridge/Logistic (과적합 방지 선형 앵커)
- ExtraTreesStockModel: Randomized Bagging (부스팅 계열의 분산을 상쇄하는 초고속 앙상블)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .base import BaseStockModel, ModelPrediction

warnings.filterwarnings("ignore", category=UserWarning)


def _clip_returns(y: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    """극단적 아웃라이어(상한가/하한가/분할 오류 등)를 Winsorize하여 학습 안정화."""
    arr = y.to_numpy(dtype=float)
    lo, hi = np.quantile(arr, [lower_q, upper_q])
    return np.clip(arr, lo, hi)


def _recency_weights(n: int, decay: float = 0.9985) -> np.ndarray:
    """최신 시장 레짐에 살짝 더 높은 가중치를 부여하는 지수 감쇠 샘플 가중치."""
    if n <= 1:
        return np.ones(n, dtype=float)
    exponents = np.arange(n - 1, -1, -1, dtype=float)
    w = decay ** exponents
    return w / np.mean(w)


class LightGBMStockModel(BaseStockModel):
    """LightGBM 기반 회귀(기대수익률) + 분류(상승확률) 모델."""

    name = "lightgbm"
    display_name = "LightGBM (Qlib GBDT)"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.reg = None
        self.clf = None

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "LightGBMStockModel":
        import lightgbm as lgb

        y_ret_clean = _clip_returns(y_ret)
        y_dir_clean = y_dir.to_numpy(dtype=int)
        weights = _recency_weights(len(X))

        self.reg = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )
        self.reg.fit(X, y_ret_clean, sample_weight=weights)

        self.clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
        )
        self.clf.fit(X, y_dir_clean, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        exp_ret = self.reg.predict(X)
        probs = self.clf.predict_proba(X)[:, 1]
        return ModelPrediction(
            expected_return=np.asarray(exp_ret, dtype=float),
            up_prob=np.clip(np.asarray(probs, dtype=float), 0.01, 0.99),
        )

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.reg is None or self.clf is None:
            return {}
        imp_r = np.asarray(self.reg.feature_importances_, dtype=float)
        imp_c = np.asarray(self.clf.feature_importances_, dtype=float)
        imp = (imp_r / (imp_r.sum() + 1e-9) + imp_c / (imp_c.sum() + 1e-9)) / 2.0
        return {f: float(v) for f, v in zip(feature_names, imp)}


class XGBoostStockModel(BaseStockModel):
    """XGBoost 기반 회귀 + 분류 모델."""

    name = "xgboost"
    display_name = "XGBoost (Regularized Tree)"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.reg = None
        self.clf = None

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "XGBoostStockModel":
        import xgboost as xgb

        y_ret_clean = _clip_returns(y_ret)
        y_dir_clean = y_dir.to_numpy(dtype=int)
        weights = _recency_weights(len(X))

        self.reg = xgb.XGBRegressor(
            objective="reg:pseudohubererror",
            tree_method="hist",
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=12,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=8.0,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.reg.fit(X, y_ret_clean, sample_weight=weights)

        self.clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=180,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=12,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=8.0,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.clf.fit(X, y_dir_clean, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        exp_ret = self.reg.predict(X)
        probs = self.clf.predict_proba(X)[:, 1]
        return ModelPrediction(
            expected_return=np.asarray(exp_ret, dtype=float),
            up_prob=np.clip(np.asarray(probs, dtype=float), 0.01, 0.99),
        )

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.reg is None or self.clf is None:
            return {}
        imp_r = np.asarray(self.reg.feature_importances_, dtype=float)
        imp_c = np.asarray(self.clf.feature_importances_, dtype=float)
        imp = (imp_r / (imp_r.sum() + 1e-9) + imp_c / (imp_c.sum() + 1e-9)) / 2.0
        return {f: float(v) for f, v in zip(feature_names, imp)}


class LinearQuantModel(BaseStockModel):
    """정규화 선형 모델 (RidgeCV + L2 LogisticRegression): 노이즈가 큰 금융 시계열의 과적합 억제 앵커."""

    name = "linear_quant"
    display_name = "Ridge/Logistic (Linear Factor)"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.reg: Pipeline | None = None
        self.clf: Pipeline | None = None
        self._clip_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def _winsorize_X(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        arr = X.to_numpy(dtype=float)
        if fit:
            lo = np.nanpercentile(arr, 1, axis=0)
            hi = np.nanpercentile(arr, 99, axis=0)
            self._clip_bounds = (lo, hi)
        assert self._clip_bounds is not None
        return np.clip(arr, self._clip_bounds[0], self._clip_bounds[1])

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "LinearQuantModel":
        Xw = self._winsorize_X(X, fit=True)
        y_ret_clean = _clip_returns(y_ret)
        y_dir_clean = y_dir.to_numpy(dtype=int)
        weights = _recency_weights(len(X))

        self.reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", RidgeCV(alphas=np.logspace(0, 4, 15))),
            ]
        )
        self.reg.fit(Xw, y_ret_clean, model__sample_weight=weights)

        self.clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(C=0.08, max_iter=500, random_state=self.random_state)),
            ]
        )
        self.clf.fit(Xw, y_dir_clean, model__sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        assert self.reg is not None and self.clf is not None
        Xw = self._winsorize_X(X, fit=False)
        exp_ret = self.reg.predict(Xw)
        probs = self.clf.predict_proba(Xw)[:, 1]
        return ModelPrediction(
            expected_return=np.asarray(exp_ret, dtype=float),
            up_prob=np.clip(np.asarray(probs, dtype=float), 0.01, 0.99),
        )

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.reg is None:
            return {}
        coef = np.abs(np.asarray(self.reg.named_steps["model"].coef_, dtype=float))
        total = coef.sum() + 1e-9
        return {f: float(v / total) for f, v in zip(feature_names, coef)}


class ExtraTreesStockModel(BaseStockModel):
    """ExtraTrees(Randomized Bagging) 듀얼 헤드 모델: 부스팅 계열의 분산을 상쇄하는 초고속 배깅 앙상블."""

    name = "extra_trees"
    display_name = "ExtraTrees (Randomized Bagging)"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.reg: Pipeline | None = None
        self.clf: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "ExtraTreesStockModel":
        y_ret_clean = _clip_returns(y_ret)
        y_dir_clean = y_dir.to_numpy(dtype=int)
        weights = _recency_weights(len(X))

        self.reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=160,
                        max_depth=8,
                        min_samples_leaf=10,
                        max_features=0.7,
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self.reg.fit(X, y_ret_clean, model__sample_weight=weights)

        self.clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=160,
                        max_depth=8,
                        min_samples_leaf=10,
                        max_features=0.7,
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        self.clf.fit(X, y_dir_clean, model__sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        assert self.reg is not None and self.clf is not None
        exp_ret = self.reg.predict(X)
        probs = self.clf.predict_proba(X)[:, 1]
        return ModelPrediction(
            expected_return=np.asarray(exp_ret, dtype=float),
            up_prob=np.clip(np.asarray(probs, dtype=float), 0.01, 0.99),
        )

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.reg is None or self.clf is None:
            return {}
        imp_r = np.asarray(self.reg.named_steps["model"].feature_importances_, dtype=float)
        imp_c = np.asarray(self.clf.named_steps["model"].feature_importances_, dtype=float)
        imp = (imp_r / (imp_r.sum() + 1e-9) + imp_c / (imp_c.sum() + 1e-9)) / 2.0
        return {f: float(v) for f, v in zip(feature_names, imp)}
