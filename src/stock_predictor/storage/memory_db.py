"""SQLite 기반 예측 기록 · 만기 자동 채점 · 자가 학습(Online Calibration) 피드백 루프.

- 실행할 때마다 예측 결과(종목, 예상가, 상승확률, 개별 모델 예측, 재무/이슈 점수)를 자동 기록
- 만기 거래일이 지난 과거 예측들의 실제 종가를 조회해 자동 채점(Hit / Miss / 실제 수익률)
- 실전 채점 성적이 쌓이면 최근 잘 맞히는 모델에 자동으로 가중치 보정(Online Calibration) 제공
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from ..config import DATA_DIR, MEMORY_DB_PATH


@dataclass
class ScorecardSummary:
    """자가 채점 누적 성적표 요약."""

    total_logged: int = 0
    evaluated_count: int = 0
    pending_count: int = 0
    hit_rate: float = 0.0
    avg_predicted_return: float = 0.0
    avg_actual_return: float = 0.0
    avg_abs_error: float = 0.0
    model_hit_rates: dict[str, tuple[float, int]] = field(default_factory=dict)  # model -> (hit_rate, n)


class PredictionMemoryDB:
    """예측 히스토리 저장 및 실전 성과 채점 DB."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or MEMORY_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    as_of_date TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    base_price REAL NOT NULL,
                    pred_price REAL NOT NULL,
                    pred_pct_ret REAL NOT NULL,
                    up_prob REAL NOT NULL,
                    fund_score REAL,
                    news_score REAL,
                    committee_score REAL,
                    action TEXT,
                    evaluated INTEGER NOT NULL DEFAULT 0,
                    actual_price REAL,
                    actual_pct_ret REAL,
                    is_hit INTEGER,
                    evaluated_at TEXT,
                    UNIQUE(market, code, horizon, as_of_date)
                );

                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    market TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    pred_pct_ret REAL NOT NULL,
                    up_prob REAL NOT NULL,
                    evaluated INTEGER NOT NULL DEFAULT 0,
                    is_hit INTEGER,
                    abs_error REAL,
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
                );
                """
            )

    def log_prediction(
        self,
        market: str,
        code: str,
        name: str,
        horizon: int,
        as_of_date: pd.Timestamp,
        target_date: pd.Timestamp,
        base_price: float,
        pred_price: float,
        pred_pct_ret: float,
        up_prob: float,
        fund_score: float,
        news_score: float,
        committee_score: float,
        action: str,
        model_entries: list[tuple[str, float, float]],  # (model_name, pred_pct_ret, up_prob)
    ) -> int:
        """최신 예측 기록 (동일 종목·동일 기준일·동일 기간은 최신 실행으로 업서트)."""
        now_str = datetime.now().isoformat(timespec="seconds")
        as_of_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
        target_str = pd.Timestamp(target_date).strftime("%Y-%m-%d")

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, evaluated FROM predictions WHERE market=? AND code=? AND horizon=? AND as_of_date=?",
                (market, code, horizon, as_of_str),
            ).fetchone()

            if existing and existing["evaluated"] == 1:
                return int(existing["id"])

            if existing:
                pred_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE predictions
                    SET created_at=?, name=?, target_date=?, base_price=?, pred_price=?,
                        pred_pct_ret=?, up_prob=?, fund_score=?, news_score=?,
                        committee_score=?, action=?
                    WHERE id=?
                    """,
                    (
                        now_str, name, target_str, float(base_price), float(pred_price),
                        float(pred_pct_ret), float(up_prob), float(fund_score),
                        float(news_score), float(committee_score), action, pred_id,
                    ),
                )
                conn.execute("DELETE FROM model_predictions WHERE prediction_id=?", (pred_id,))
            else:
                cur = conn.execute(
                    """
                    INSERT INTO predictions (
                        created_at, market, code, name, horizon, as_of_date, target_date,
                        base_price, pred_price, pred_pct_ret, up_prob, fund_score, news_score,
                        committee_score, action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_str, market, code, name, int(horizon), as_of_str, target_str,
                        float(base_price), float(pred_price), float(pred_pct_ret),
                        float(up_prob), float(fund_score), float(news_score),
                        float(committee_score), action,
                    ),
                )
                pred_id = int(cur.lastrowid)

            for m_name, m_ret, m_prob in model_entries:
                conn.execute(
                    """
                    INSERT INTO model_predictions (prediction_id, market, model_name, pred_pct_ret, up_prob)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (pred_id, market, m_name, float(m_ret), float(m_prob)),
                )
            return pred_id

    def evaluate_due_predictions(
        self,
        price_resolver: Callable[[str, str, str], float | None],
        as_of_today: str | None = None,
    ) -> int:
        """만기일이 도래한 미채점 예측들을 실제 종가로 자동 채점.

        price_resolver(market, code, target_date_str) -> 실제 만기(또는 이후 첫 거래일) 종가
        """
        today_str = as_of_today or datetime.now().strftime("%Y-%m-%d")
        evaluated_now = 0

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE evaluated=0 AND target_date <= ?",
                (today_str,),
            ).fetchall()

            for r in rows:
                actual_price = price_resolver(r["market"], r["code"], r["target_date"])
                if actual_price is None or actual_price <= 0:
                    continue

                base_p = float(r["base_price"])
                actual_ret = (actual_price / base_p) - 1.0
                actual_up = 1 if actual_ret > 0 else 0
                pred_up = 1 if float(r["up_prob"]) >= 0.5 else 0
                is_hit = 1 if pred_up == actual_up else 0
                now_str = datetime.now().isoformat(timespec="seconds")

                conn.execute(
                    """
                    UPDATE predictions
                    SET evaluated=1, actual_price=?, actual_pct_ret=?, is_hit=?, evaluated_at=?
                    WHERE id=?
                    """,
                    (float(actual_price), float(actual_ret), is_hit, now_str, r["id"]),
                )

                m_rows = conn.execute(
                    "SELECT id, pred_pct_ret, up_prob FROM model_predictions WHERE prediction_id=?",
                    (r["id"],),
                ).fetchall()
                for mr in m_rows:
                    m_up = 1 if float(mr["up_prob"]) >= 0.5 else 0
                    m_hit = 1 if m_up == actual_up else 0
                    m_err = abs(actual_ret - float(mr["pred_pct_ret"]))
                    conn.execute(
                        "UPDATE model_predictions SET evaluated=1, is_hit=?, abs_error=? WHERE id=?",
                        (m_hit, float(m_err), mr["id"]),
                    )
                evaluated_now += 1
        return evaluated_now

    def get_model_calibration_multipliers(self, market: str, min_samples: int = 5) -> dict[str, float]:
        """실전 채점 성과가 누적된 모델들의 보정 배수(0.75 ~ 1.30) 반환 (Online Self-Learning)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_name, AVG(is_hit) as hit_rate, COUNT(*) as cnt
                FROM model_predictions
                WHERE market=? AND evaluated=1
                GROUP BY model_name
                """,
                (market,),
            ).fetchall()

        multipliers: dict[str, float] = {}
        for r in rows:
            cnt = int(r["cnt"])
            if cnt < min_samples:
                continue
            hit = float(r["hit_rate"])
            # 적중률 50% 기준: 60%면 ~1.20x, 40%면 ~0.82x (과도한 흔들림 방지)
            mult = float(np.clip(1.0 + (hit - 0.5) * 2.0, 0.75, 1.30))
            multipliers[str(r["model_name"])] = mult
        return multipliers

    def get_scorecard_summary(self) -> ScorecardSummary:
        with self._connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            ev_rows = conn.execute(
                "SELECT pred_pct_ret, actual_pct_ret, is_hit FROM predictions WHERE evaluated=1"
            ).fetchall()
            m_rows = conn.execute(
                """
                SELECT model_name, AVG(is_hit) as hit_rate, COUNT(*) as cnt
                FROM model_predictions
                WHERE evaluated=1
                GROUP BY model_name
                """
            ).fetchall()

        ev_cnt = len(ev_rows)
        if ev_cnt > 0:
            hit_rate = float(np.mean([r["is_hit"] for r in ev_rows]))
            avg_pred = float(np.mean([r["pred_pct_ret"] for r in ev_rows]))
            avg_act = float(np.mean([r["actual_pct_ret"] for r in ev_rows]))
            avg_err = float(np.mean([abs(r["actual_pct_ret"] - r["pred_pct_ret"]) for r in ev_rows]))
        else:
            hit_rate = avg_pred = avg_act = avg_err = 0.0

        m_hits = {str(r["model_name"]): (float(r["hit_rate"]), int(r["cnt"])) for r in m_rows}
        return ScorecardSummary(
            total_logged=total,
            evaluated_count=ev_cnt,
            pending_count=total - ev_cnt,
            hit_rate=hit_rate,
            avg_predicted_return=avg_pred,
            avg_actual_return=avg_act,
            avg_abs_error=avg_err,
            model_hit_rates=m_hits,
        )

    def get_recent_records(self, limit: int = 50) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT as_of_date, target_date, market, code, name, horizon,
                       base_price, pred_price, pred_pct_ret, up_prob,
                       fund_score, news_score, action, evaluated,
                       actual_price, actual_pct_ret, is_hit
                FROM predictions
                ORDER BY as_of_date DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
