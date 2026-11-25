"""모든 예측 모델이 따르는 공통 베이스 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelPrediction:
    """단일 모델의 배치 예측 결과."""

    expected_return: np.ndarray   # 예측 로그수익률
    up_prob: np.ndarray           # 상승 확률 (0~1)


class BaseStockModel(ABC):
    """회귀(수익률) + 분류(상승확률) 듀얼 헤드 주식 예측 모델 추상 클래스."""

    name: str = "base"
    display_name: str = "Base Model"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "BaseStockModel":
        """학습 데이터로 모델을 적합."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """입력 피처에 대해 기대수익률과 상승확률을 예측."""

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """모델의 피처 중요도 반환 (미지원 모델은 빈 딕셔너리)."""
        return {}
