"""엄격한 Out-of-Sample Walk-Forward 백테스팅 엔진.

- 중첩 트렌치(Overlapping Tranche) 일별 포트폴리오 모델:
  horizon=H일 전략에서 수익률이 H배 과대 계상되는 오류 없이,
  매일 자본의 1/H씩 신규 시그널에 따라 리밸런싱하여 일간 실제 수익률로 정확히 복리 계산.
- 국내/해외 시장별 현실적 수수료·증권거래세·슬리피지 반영.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import MARKET_COST_BPS, PredictorConfig


@dataclass
class BacktestResult:
    """Out-of-Sample 워크포워드 백테스팅 성과 결과."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    # 예측 정확도 지표
    direction_accuracy: float
    rank_ic: float
    rmse: float
    # 전략 vs Buy & Hold 수익/리스크 지표
    strategy_total_return: float
    benchmark_total_return: float
    excess_return: float
    strategy_cagr: float
    benchmark_cagr: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_sortino: float
    strategy_mdd: float
    benchmark_mdd: float
    win_rate: float
    profit_factor: float
    avg_exposure: float          # 평균 주식 보유 비중 (0~1)
    total_turnover: float        # 총 회전율 (편도 기준)
    # 시각화용 시계열 DataFrame
    equity_curve: pd.DataFrame   # strategy_equity, benchmark_equity, strategy_dd, benchmark_dd, position


class WalkForwardBacktester:
    """Out-of-Sample 예측 기반 거래비용 반영 백테스터."""

    def __init__(self, config: PredictorConfig) -> None:
        self.config = config

    def run(self, oof_df: pd.DataFrame, market: str = "US") -> BacktestResult:
        """OOF 예측 DataFrame(close, actual_ret, actual_dir, pred_ret, pred_prob)으로 시뮬레이션 수행."""
        df = oof_df.sort_index().copy()
        costs = MARKET_COST_BPS.get(market, MARKET_COST_BPS["US"])
        buy_cost = costs["commission"] + costs["slippage"]
        sell_cost = costs["commission"] + costs["sell_tax"] + costs["slippage"]

        # 1. 예측 품질 지표 (OOF 전체)
        actual_r = df["actual_ret"].to_numpy(dtype=float)
        actual_d = df["actual_dir"].to_numpy(dtype=int)
        pred_r = df["pred_ret"].to_numpy(dtype=float)
        pred_p = df["pred_prob"].to_numpy(dtype=float)

        dir_acc = float(np.mean((pred_p >= 0.5).astype(int) == actual_d)) if len(actual_d) > 0 else 0.5
        if len(actual_r) > 5 and np.std(pred_r) > 1e-12:
            ic = float(spearmanr(pred_r, actual_r).statistic)
            if np.isnan(ic):
                ic = 0.0
        else:
            ic = 0.0
        rmse = float(np.sqrt(np.mean((actual_r - pred_r) ** 2))) if len(actual_r) > 0 else 0.0

        # 2. 일별 시그널 생성 -> H일 중첩 트렌치 목표 비중 산출
        long_sig = (df["pred_ret"] >= self.config.long_threshold) & (
            df["pred_prob"] >= self.config.prob_threshold
        )
        if self.config.allow_short:
            short_sig = (df["pred_ret"] <= -self.config.long_threshold) & (
                df["pred_prob"] <= (1.0 - self.config.prob_threshold)
            )
            raw_signal = pd.Series(0.0, index=df.index)
            raw_signal.loc[long_sig] = 1.0
            raw_signal.loc[short_sig] = -1.0
        else:
            raw_signal = long_sig.astype(float)

        # 신뢰도 강도에 따른 소프트 포지션 사이징 (0.5 ~ 1.0)
        strength = np.clip((df["pred_prob"] - 0.5) / 0.15, 0.5, 1.0)
        sized_signal = raw_signal * np.where(raw_signal != 0, strength, 0.0)

        h = max(self.config.horizon, 1)
        # t일 장 마감 시그널은 t+1일 수익률부터 적용 (shift(1)로 실행 지연 반영!)
        target_weight = sized_signal.rolling(window=h, min_periods=1).mean()
        held_weight = target_weight.shift(1).fillna(0.0)

        # 실제 일간 단순 수익률
        daily_ret = df["close"].pct_change().fillna(0.0)
        weight_change = held_weight.diff().fillna(held_weight)

        # 매수/매도 방향별 거래비용 정밀 차감
        daily_cost = np.where(
            weight_change > 0,
            weight_change * buy_cost,
            np.abs(weight_change) * sell_cost,
        )
        strat_daily_ret = held_weight * daily_ret - daily_cost
        bm_daily_ret = daily_ret.copy()

        # 누적 자산 곡선 (초기 자본 = 1.0)
        strat_eq = (1.0 + strat_daily_ret).cumprod()
        bm_eq = (1.0 + bm_daily_ret).cumprod()

        strat_dd = strat_eq / strat_eq.cummax() - 1.0
        bm_dd = bm_eq / bm_eq.cummax() - 1.0

        # 3. 성과 지표 계산
        n_days = len(df)
        years = max(n_days / 252.0, 1.0 / 252.0)

        strat_total = float(strat_eq.iloc[-1] - 1.0)
        bm_total = float(bm_eq.iloc[-1] - 1.0)
        strat_cagr = float((max(strat_eq.iloc[-1], 1e-6)) ** (1.0 / years) - 1.0)
        bm_cagr = float((max(bm_eq.iloc[-1], 1e-6)) ** (1.0 / years) - 1.0)

        strat_std = float(strat_daily_ret.std())
        bm_std = float(bm_daily_ret.std())
        strat_sharpe = float((strat_daily_ret.mean() / strat_std) * np.sqrt(252)) if strat_std > 1e-9 else 0.0
        bm_sharpe = float((bm_daily_ret.mean() / bm_std) * np.sqrt(252)) if bm_std > 1e-9 else 0.0

        downside = strat_daily_ret[strat_daily_ret < 0]
        down_std = float(downside.std()) if len(downside) > 2 else 0.0
        strat_sortino = float((strat_daily_ret.mean() / down_std) * np.sqrt(252)) if down_std > 1e-9 else 0.0

        # 활성 포지션 거래일 기준 승률 & 손익비
        active_rets = strat_daily_ret[held_weight.abs() > 1e-6]
        if len(active_rets) > 0:
            win_rate = float((active_rets > 0).mean())
            gross_profit = float(active_rets[active_rets > 0].sum())
            gross_loss = float(np.abs(active_rets[active_rets < 0].sum()))
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else (2.0 if gross_profit > 0 else 0.0)
        else:
            win_rate = 0.0
            profit_factor = 0.0

        curve_df = pd.DataFrame(
            {
                "close": df["close"],
                "strategy_equity": strat_eq,
                "benchmark_equity": bm_eq,
                "strategy_dd": strat_dd,
                "benchmark_dd": bm_dd,
                "position": held_weight,
                "pred_ret": df["pred_ret"],
                "pred_prob": df["pred_prob"],
            },
            index=df.index,
        )

        return BacktestResult(
            start_date=df.index[0],
            end_date=df.index[-1],
            n_days=n_days,
            direction_accuracy=dir_acc,
            rank_ic=ic,
            rmse=rmse,
            strategy_total_return=strat_total,
            benchmark_total_return=bm_total,
            excess_return=strat_total - bm_total,
            strategy_cagr=strat_cagr,
            benchmark_cagr=bm_cagr,
            strategy_sharpe=strat_sharpe,
            benchmark_sharpe=bm_sharpe,
            strategy_sortino=strat_sortino,
            strategy_mdd=float(strat_dd.min()),
            benchmark_mdd=float(bm_dd.min()),
            win_rate=win_rate,
            profit_factor=float(min(profit_factor, 99.0)),
            avg_exposure=float(held_weight.abs().mean()),
            total_turnover=float(weight_change.abs().sum()),
            equity_curve=curve_df,
        )
