"""시계열 신경망(딥러닝) & 시계열 파운데이션 모델(Chronos) 어댑터.

1. TemporalNeuralStockModel:
   - 시계열 지연 윈도우(Lag Sequence) 피처를 구성해 가격·모멘텀·변동성의 순차적 패턴 학습
   - PyTorch 설치 시: 1D-CNN + BiGRU + Temporal Attention 하이브리드 신경망 실행
   - PyTorch 미설치(경량 CPU 환경) 시: Scikit-Learn Multi-Scale Sequence MLP로 자동 동작
2. ChronosFoundationAdapter:
   - Amazon Chronos (chronos-forecasting) 설치 및 활성화 시 제로샷 시계열 파운데이션 모델 예측 결합
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .base import BaseStockModel, ModelPrediction
from .tabular import _clip_returns

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


if HAS_TORCH:

    class _TemporalAttentionNet(nn.Module):
        """경량 1D-CNN + GRU + Temporal Attention 듀얼 헤드 신경망."""

        def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.attn = nn.Linear(hidden_dim * 2, 1)
            self.head_ret = nn.Sequential(
                nn.Linear(hidden_dim * 2, 16),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(16, 1),
            )
            self.head_dir = nn.Sequential(
                nn.Linear(hidden_dim * 2, 16),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (B, T, F)
            c = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.gru(c)
            weights = torch.softmax(self.attn(out), dim=1)
            ctx = torch.sum(out * weights, dim=1)
            return self.head_ret(ctx).squeeze(-1), self.head_dir(ctx).squeeze(-1)


class TemporalNeuralStockModel(BaseStockModel):
    """시계열 신경망 모델 (PyTorch TCN-GRU-Attention 또는 Multi-Scale Sequence MLP)."""

    name = "temporal_neural"
    display_name = "Temporal Deep Net (CNN-GRU-Attn)" if HAS_TORCH else "Temporal Neural Net (Sequence MLP)"

    def __init__(self, seq_len: int = 10, random_state: int = 42) -> None:
        self.seq_len = seq_len
        self.random_state = random_state
        self._scaler: Pipeline | None = None
        self._torch_net: Any = None
        self._mlp_reg: Pipeline | None = None
        self._mlp_clf: Pipeline | None = None
        self._clip_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._history_tail: np.ndarray | None = None
        self._selected_cols: list[str] = []

    def _select_sequence_cols(self, X: pd.DataFrame) -> list[str]:
        preferred = [
            "ret_1d", "ret_5d", "ret_20d", "dist_sma_20", "dist_sma_60",
            "rsi_14", "macd_hist_pct", "bb_zscore_20", "atr_14_pct",
            "rv_20d", "vol_zscore_20", "bm_ret_5d", "vix_level",
            "drawdown_60d", "obv_norm_10",
        ]
        cols = [c for c in preferred if c in X.columns]
        if len(cols) < 6:
            cols = list(X.columns[: min(15, len(X.columns))])
        return cols

    def _build_lag_matrix(self, arr_scaled: np.ndarray, pad_prefix: np.ndarray | None = None) -> np.ndarray:
        """2D 배열을 시계열 지연 특징(t, t-1, t-3, t-5)으로 확장."""
        if pad_prefix is not None and len(pad_prefix) > 0:
            full = np.vstack([pad_prefix, arr_scaled])
            offset = len(pad_prefix)
        else:
            full = arr_scaled
            offset = 0

        lags = [0, 1, 3, 5]
        n_total, n_feat = full.shape
        blocks = []
        for lag in lags:
            shifted = np.zeros_like(full)
            if lag == 0:
                shifted[:] = full
            else:
                shifted[lag:] = full[:-lag]
                shifted[:lag] = full[0]
            blocks.append(shifted[offset:])
        return np.hstack(blocks)

    def _build_3d_sequences(self, arr_scaled: np.ndarray, pad_prefix: np.ndarray | None = None) -> np.ndarray:
        """PyTorch용 (N, seq_len, F) 3D 시퀀스 텐서 생성."""
        if pad_prefix is not None and len(pad_prefix) > 0:
            full = np.vstack([pad_prefix, arr_scaled])
            offset = len(pad_prefix)
        else:
            full = arr_scaled
            offset = 0

        n_out = len(arr_scaled)
        n_feat = full.shape[1]
        seqs = np.zeros((n_out, self.seq_len, n_feat), dtype=np.float32)
        for i in range(n_out):
            end_pos = offset + i + 1
            start_pos = end_pos - self.seq_len
            if start_pos >= 0:
                seqs[i] = full[start_pos:end_pos]
            else:
                pad_len = -start_pos
                seqs[i, :pad_len] = full[0]
                seqs[i, pad_len:] = full[:end_pos]
        return seqs

    def fit(self, X: pd.DataFrame, y_ret: pd.Series, y_dir: pd.Series) -> "TemporalNeuralStockModel":
        self._selected_cols = self._select_sequence_cols(X)
        raw = X[self._selected_cols].to_numpy(dtype=float)
        lo = np.nanpercentile(raw, 1, axis=0)
        hi = np.nanpercentile(raw, 99, axis=0)
        self._clip_bounds = (lo, hi)
        raw_clipped = np.clip(raw, lo, hi)

        self._scaler = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        scaled = np.clip(self._scaler.fit_transform(raw_clipped), -5.0, 5.0)
        self._history_tail = scaled[-self.seq_len :].copy()

        y_ret_clean = _clip_returns(y_ret)
        y_dir_clean = y_dir.to_numpy(dtype=int)

        if HAS_TORCH and len(X) >= 80:
            self._fit_torch(scaled, y_ret_clean, y_dir_clean)
        else:
            self._fit_mlp(scaled, y_ret_clean, y_dir_clean)
        return self

    def _fit_torch(self, scaled: np.ndarray, y_ret: np.ndarray, y_dir: np.ndarray) -> None:
        torch.manual_seed(self.random_state)
        seqs = self._build_3d_sequences(scaled)
        X_t = torch.tensor(seqs, dtype=torch.float32)
        yr_t = torch.tensor(y_ret, dtype=torch.float32)
        yd_t = torch.tensor(y_dir, dtype=torch.float32)

        net = _TemporalAttentionNet(input_dim=seqs.shape[2], hidden_dim=32)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        huber = nn.HuberLoss(delta=0.05)
        bce = nn.BCEWithLogitsLoss()

        net.train()
        batch_size = 64
        n = len(X_t)
        for _ in range(18):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                p_ret, p_logit = net(X_t[idx])
                loss = huber(p_ret, yr_t[idx]) * 10.0 + bce(p_logit, yd_t[idx]) * 0.5
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        net.eval()
        self._torch_net = net

    def _fit_mlp(self, scaled: np.ndarray, y_ret: np.ndarray, y_dir: np.ndarray) -> None:
        X_lag = self._build_lag_matrix(scaled)
        self._mlp_reg = Pipeline(
            [
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(48, 24),
                        activation="relu",
                        alpha=2.0,
                        learning_rate_init=0.0015,
                        max_iter=250,
                        early_stopping=True,
                        n_iter_no_change=12,
                        validation_fraction=0.15,
                        random_state=self.random_state,
                    ),
                )
            ]
        )
        self._mlp_reg.fit(X_lag, y_ret)

        self._mlp_clf = Pipeline(
            [
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(48, 24),
                        activation="relu",
                        alpha=2.5,
                        learning_rate_init=0.0015,
                        max_iter=250,
                        early_stopping=True,
                        n_iter_no_change=12,
                        validation_fraction=0.15,
                        random_state=self.random_state,
                    ),
                )
            ]
        )
        self._mlp_clf.fit(X_lag, y_dir)

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        assert self._scaler is not None and self._clip_bounds is not None
        raw = X[self._selected_cols].to_numpy(dtype=float)
        raw_clipped = np.clip(raw, self._clip_bounds[0], self._clip_bounds[1])
        scaled = np.clip(self._scaler.transform(raw_clipped), -5.0, 5.0)

        if self._torch_net is not None:
            seqs = self._build_3d_sequences(scaled, pad_prefix=self._history_tail)
            with torch.no_grad():
                p_ret, p_logit = self._torch_net(torch.tensor(seqs, dtype=torch.float32))
                exp_ret = p_ret.numpy().astype(float)
                up_prob = torch.sigmoid(p_logit).numpy().astype(float)
            return ModelPrediction(
                expected_return=exp_ret,
                up_prob=np.clip(up_prob, 0.01, 0.99),
            )

        assert self._mlp_reg is not None and self._mlp_clf is not None
        X_lag = self._build_lag_matrix(scaled, pad_prefix=self._history_tail)
        exp_ret = self._mlp_reg.predict(X_lag)
        raw_prob = np.asarray(self._mlp_clf.predict_proba(X_lag)[:, 1], dtype=float)
        # 신경망 확률 과신(Overconfidence) 완화를 위한 온도 스무딩
        up_prob = 0.5 + (raw_prob - 0.5) * 0.65
        return ModelPrediction(
            expected_return=np.asarray(exp_ret, dtype=float),
            up_prob=np.clip(up_prob, 0.05, 0.95),
        )


class ChronosFoundationAdapter:
    """Amazon Chronos 시계열 파운데이션 모델 플러그인 (선택적 설치 시 동작)."""

    name = "chronos_foundation"
    display_name = "Amazon Chronos (Zero-Shot Foundation)"

    def __init__(self, model_id: str = "amazon/chronos-t5-tiny") -> None:
        self.model_id = model_id
        self._pipeline: Any = None
        self.available = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            self.available = True
        except Exception:
            self.available = False

    def forecast_close_series(self, close_series: pd.Series, horizon: int = 5) -> dict[str, float] | None:
        """종가 시계열을 입력받아 향후 horizon일의 예상 수익률·분위수 반환."""
        if not self.available or self._pipeline is None or len(close_series) < 30:
            return None
        try:
            import torch

            ctx = torch.tensor(close_series.tail(180).to_numpy(dtype=np.float32))
            forecast = self._pipeline.predict(ctx, prediction_length=horizon, num_samples=40)
            samples = forecast[0].numpy()  # (num_samples, horizon)
            last_close = float(close_series.iloc[-1])
            end_prices = samples[:, -1]
            med_price = float(np.median(end_prices))
            low_price = float(np.quantile(end_prices, 0.1))
            high_price = float(np.quantile(end_prices, 0.9))
            up_prob = float(np.mean(end_prices > last_close))
            exp_ret = float(np.log(max(med_price, 1e-6) / max(last_close, 1e-6)))
            return {
                "expected_return": exp_ret,
                "up_prob": float(np.clip(up_prob, 0.01, 0.99)),
                "low_price": low_price,
                "median_price": med_price,
                "high_price": high_price,
            }
        except Exception:
            return None
