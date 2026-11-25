"""국내·해외 통합 AI 모의투자(Paper Trading) 엔진 & 학습 데이터셋 생성기.

구성:
1. PaperTradingDB (SQLite):
   - 라이브 모의 계좌(초기 1억 원) 잔고, 보유 종목, 매수/매도 주문 로그, 일별 자산 스냅샷 관리
   - 과거 멀티종목 워크포워드 모의투자 시뮬레이션 거래 로그, 일별 수익률 곡선, 학습용 피처-결과 데이터셋 저장
2. LivePaperTrader:
   - AI 스크리닝 결과(매수 의견·권장 비중·손절선·목표가)에 따라 실제 모의 매수/리밸런싱 집행
   - 수동 모의 매수/매도 주문, 최신 시세·환율 기반 실시간 평가손익(KRW 통합) 갱신
3. HistoricalPaperSimulator:
   - 국내·해외 다중 종목 실제 주가 + 환율로 최근 N개월~1년 간 Strict Walk-Forward 모의투자 수행
   - 매주(5거래일) 리밸런싱 + 손절(-8%)/익절(+16%) + 거래비용/세금/슬리피지 + 변동성 비중 조절 반영
   - 모든 모의 매매에서 [진입 시점 퀀트 피처 + AI 예측치 + 실제 실현 수익률/승패] 학습 데이터셋을
     CSV/JSON/SQLite로 축적해 모델 개선에 재활용 가능하도록 저장
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from ..config import DATA_DIR, MARKET_COST_BPS, PredictorConfig
from ..data.features import FeatureEngine
from ..data.market_data import MarketDataBundle
from ..models.tabular import ExtraTreesStockModel, LightGBMStockModel, LinearQuantModel

DEFAULT_INITIAL_CASH_KRW = 100_000_000.0  # 초기 모의투자 자본: 1억 원
PAPER_DB_PATH = DATA_DIR / "paper_trading.db"
PAPER_DATASET_DIR = DATA_DIR / "paper_trading_dataset"


@dataclass
class SimulationReport:
    """과거 워크포워드 멀티종목 AI 모의투자 시뮬레이션 결과 요약."""

    run_id: str
    created_at: str
    start_date: str
    end_date: str
    n_days: int
    n_symbols: int
    initial_capital_krw: float
    final_equity_krw: float
    total_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    benchmark_sharpe: float
    max_drawdown_pct: float
    benchmark_mdd_pct: float
    total_trades: int
    closed_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_fees_krw: float
    avg_invested_pct: float
    symbols_used: list[str] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    symbol_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    learning_dataset_path: str = ""


class PaperTradingDB:
    """라이브 모의투자 계좌 + 시뮬레이션 트랙레코드 SQLite 저장소."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or PAPER_DB_PATH
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
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    created_at TEXT NOT NULL,
                    initial_cash_krw REAL NOT NULL,
                    cash_krw REAL NOT NULL,
                    fx_usdkrw REAL NOT NULL DEFAULT 1380.0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    shares REAL NOT NULL,
                    avg_cost_local REAL NOT NULL,
                    avg_cost_krw REAL NOT NULL,
                    current_price_local REAL NOT NULL,
                    stop_loss_local REAL,
                    target_price_local REAL,
                    ai_action TEXT,
                    ai_score REAL,
                    fund_grade TEXT,
                    rationale TEXT,
                    opened_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(market, code)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    executed_at TEXT NOT NULL,
                    side TEXT NOT NULL,          -- BUY / SELL
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    shares REAL NOT NULL,
                    price_local REAL NOT NULL,
                    fx_rate REAL NOT NULL,
                    gross_krw REAL NOT NULL,
                    fee_krw REAL NOT NULL,
                    net_krw REAL NOT NULL,
                    realized_pnl_krw REAL,
                    realized_pnl_pct REAL,
                    reason TEXT,
                    ai_score REAL,
                    up_prob REAL
                );

                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snap_date TEXT NOT NULL UNIQUE,
                    total_equity_krw REAL NOT NULL,
                    cash_krw REAL NOT NULL,
                    invested_krw REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    n_positions INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulation_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS simulation_equity (
                    run_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    equity_krw REAL NOT NULL,
                    benchmark_krw REAL NOT NULL,
                    cash_krw REAL NOT NULL,
                    invested_pct REAL NOT NULL,
                    drawdown_pct REAL NOT NULL,
                    benchmark_dd_pct REAL NOT NULL,
                    n_holdings INTEGER NOT NULL,
                    PRIMARY KEY (run_id, date)
                );

                CREATE TABLE IF NOT EXISTS simulation_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    shares REAL NOT NULL,
                    invested_krw REAL NOT NULL,
                    pnl_krw REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    fee_krw REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    pred_ret_pct REAL NOT NULL,
                    pred_prob_pct REAL NOT NULL,
                    holding_days INTEGER NOT NULL,
                    entry_rationale TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS live_account_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    total_equity_krw REAL NOT NULL,
                    cash_krw REAL NOT NULL,
                    invested_krw REAL NOT NULL,
                    unrealized_pnl_krw REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    fx_usdkrw REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS live_position_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_at TEXT NOT NULL,
                    market TEXT NOT NULL,
                    code TEXT NOT NULL,
                    name TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    price_local REAL NOT NULL,
                    prev_price_local REAL NOT NULL,
                    tick_change_pct REAL NOT NULL,
                    day_change_pct REAL,
                    pnl_krw REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    direction TEXT NOT NULL,
                    status_note TEXT
                );
                """
            )

    def ensure_account(self, initial_cash_krw: float = DEFAULT_INITIAL_CASH_KRW, fx_usdkrw: float = 1380.0) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO account (id, created_at, initial_cash_krw, cash_krw, fx_usdkrw, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                    (now, float(initial_cash_krw), float(initial_cash_krw), float(fx_usdkrw), now),
                )
                row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        return dict(row)

    def reset_account(self, initial_cash_krw: float = DEFAULT_INITIAL_CASH_KRW, fx_usdkrw: float = 1380.0) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM daily_snapshots")
            conn.execute("DELETE FROM account")
            conn.execute(
                "INSERT INTO account (id, created_at, initial_cash_krw, cash_krw, fx_usdkrw, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                (now, float(initial_cash_krw), float(initial_cash_krw), float(fx_usdkrw), now),
            )
            row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        return dict(row)

    def get_account(self) -> dict[str, Any]:
        return self.ensure_account()

    def get_positions_df(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM positions ORDER BY (shares * current_price_local) DESC").fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_orders_df(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_snapshots_df(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM daily_snapshots ORDER BY snap_date ASC").fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_live_account_ticks_df(self, limit: int = 300) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM live_account_ticks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in reversed(rows)])
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        return df

    def get_live_position_ticks_df(self, limit: int = 120) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM live_position_ticks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_latest_position_tick_map(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*
                FROM live_position_ticks t
                INNER JOIN (
                    SELECT code, MAX(id) AS max_id
                    FROM live_position_ticks
                    GROUP BY code
                ) m ON t.id = m.max_id
                """
            ).fetchall()
        return {str(r["code"]): dict(r) for r in rows}

    def save_simulation(self, report: SimulationReport) -> None:
        summary = {
            "run_id": report.run_id,
            "created_at": report.created_at,
            "start_date": report.start_date,
            "end_date": report.end_date,
            "n_days": report.n_days,
            "n_symbols": report.n_symbols,
            "initial_capital_krw": report.initial_capital_krw,
            "final_equity_krw": report.final_equity_krw,
            "total_return_pct": report.total_return_pct,
            "benchmark_return_pct": report.benchmark_return_pct,
            "excess_return_pct": report.excess_return_pct,
            "cagr_pct": report.cagr_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "benchmark_sharpe": report.benchmark_sharpe,
            "max_drawdown_pct": report.max_drawdown_pct,
            "benchmark_mdd_pct": report.benchmark_mdd_pct,
            "total_trades": report.total_trades,
            "closed_trades": report.closed_trades,
            "win_rate_pct": report.win_rate_pct,
            "avg_win_pct": report.avg_win_pct,
            "avg_loss_pct": report.avg_loss_pct,
            "profit_factor": report.profit_factor,
            "total_fees_krw": report.total_fees_krw,
            "avg_invested_pct": report.avg_invested_pct,
            "symbols_used": report.symbols_used,
            "learning_dataset_path": report.learning_dataset_path,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO simulation_runs (run_id, created_at, summary_json) VALUES (?, ?, ?)",
                (report.run_id, report.created_at, json.dumps(summary, ensure_ascii=False)),
            )
            conn.execute("DELETE FROM simulation_equity WHERE run_id=?", (report.run_id,))
            conn.execute("DELETE FROM simulation_trades WHERE run_id=?", (report.run_id,))

            if not report.equity_curve.empty:
                eq_rows = [
                    (
                        report.run_id,
                        pd.Timestamp(dt).strftime("%Y-%m-%d"),
                        float(r["equity_krw"]),
                        float(r["benchmark_krw"]),
                        float(r["cash_krw"]),
                        float(r["invested_pct"]),
                        float(r["drawdown_pct"]),
                        float(r["benchmark_dd_pct"]),
                        int(r["n_holdings"]),
                    )
                    for dt, r in report.equity_curve.iterrows()
                ]
                conn.executemany(
                    """
                    INSERT INTO simulation_equity
                    (run_id, date, equity_krw, benchmark_krw, cash_krw, invested_pct, drawdown_pct, benchmark_dd_pct, n_holdings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    eq_rows,
                )

            if not report.trade_log.empty:
                t_rows = [
                    (
                        report.run_id,
                        str(r["entry_date"]),
                        str(r["exit_date"]),
                        str(r["market"]),
                        str(r["code"]),
                        str(r["name"]),
                        float(r["entry_price"]),
                        float(r["exit_price"]),
                        float(r["shares"]),
                        float(r["invested_krw"]),
                        float(r["pnl_krw"]),
                        float(r["pnl_pct"]),
                        float(r["fee_krw"]),
                        str(r["exit_reason"]),
                        float(r["pred_ret_pct"]),
                        float(r["pred_prob_pct"]),
                        int(r["holding_days"]),
                        str(r.get("entry_rationale", "")),
                    )
                    for _, r in report.trade_log.iterrows()
                ]
                conn.executemany(
                    """
                    INSERT INTO simulation_trades
                    (run_id, entry_date, exit_date, market, code, name, entry_price, exit_price,
                     shares, invested_krw, pnl_krw, pnl_pct, fee_krw, exit_reason, pred_ret_pct, pred_prob_pct, holding_days, entry_rationale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    t_rows,
                )

    def load_latest_simulation(self) -> SimulationReport | None:
        with self._connect() as conn:
            run_row = conn.execute("SELECT * FROM simulation_runs ORDER BY created_at DESC LIMIT 1").fetchone()
            if run_row is None:
                return None
            meta = json.loads(run_row["summary_json"])
            run_id = meta["run_id"]
            eq_rows = conn.execute("SELECT * FROM simulation_equity WHERE run_id=? ORDER BY date ASC", (run_id,)).fetchall()
            tr_rows = conn.execute("SELECT * FROM simulation_trades WHERE run_id=? ORDER BY exit_date DESC, id DESC", (run_id,)).fetchall()

        eq_df = pd.DataFrame([dict(r) for r in eq_rows])
        if not eq_df.empty:
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            eq_df = eq_df.set_index("date").drop(columns=["run_id"], errors="ignore")

        tr_df = pd.DataFrame([dict(r) for r in tr_rows]).drop(columns=["id", "run_id"], errors="ignore") if tr_rows else pd.DataFrame()
        sym_pnl = _build_symbol_pnl_summary(tr_df)

        return SimulationReport(
            run_id=meta["run_id"],
            created_at=meta["created_at"],
            start_date=meta["start_date"],
            end_date=meta["end_date"],
            n_days=int(meta["n_days"]),
            n_symbols=int(meta["n_symbols"]),
            initial_capital_krw=float(meta["initial_capital_krw"]),
            final_equity_krw=float(meta["final_equity_krw"]),
            total_return_pct=float(meta["total_return_pct"]),
            benchmark_return_pct=float(meta["benchmark_return_pct"]),
            excess_return_pct=float(meta["excess_return_pct"]),
            cagr_pct=float(meta["cagr_pct"]),
            sharpe_ratio=float(meta["sharpe_ratio"]),
            benchmark_sharpe=float(meta["benchmark_sharpe"]),
            max_drawdown_pct=float(meta["max_drawdown_pct"]),
            benchmark_mdd_pct=float(meta["benchmark_mdd_pct"]),
            total_trades=int(meta["total_trades"]),
            closed_trades=int(meta["closed_trades"]),
            win_rate_pct=float(meta["win_rate_pct"]),
            avg_win_pct=float(meta["avg_win_pct"]),
            avg_loss_pct=float(meta["avg_loss_pct"]),
            profit_factor=float(meta["profit_factor"]),
            total_fees_krw=float(meta["total_fees_krw"]),
            avg_invested_pct=float(meta["avg_invested_pct"]),
            symbols_used=list(meta.get("symbols_used", [])),
            equity_curve=eq_df,
            trade_log=tr_df,
            symbol_pnl=sym_pnl,
            learning_dataset_path=str(meta.get("learning_dataset_path", "")),
        )


def _build_symbol_pnl_summary(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log is None or trade_log.empty:
        return pd.DataFrame()
    grouped = (
        trade_log.groupby(["market", "code", "name"], as_index=False)
        .agg(
            거래횟수=("pnl_krw", "count"),
            승률=("pnl_krw", lambda s: round(float((s > 0).mean() * 100.0), 1)),
            누적손익_원=("pnl_krw", lambda s: round(float(s.sum()), 0)),
            평균수익률=("pnl_pct", lambda s: round(float(s.mean()), 2)),
            평균보유일=("holding_days", lambda s: round(float(s.mean()), 1)),
        )
        .sort_values("누적손익_원", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


class LivePaperTrader:
    """1억 원 라이브 모의투자 계좌 매매 집행 및 포트폴리오 평가 관리기."""

    def __init__(self, db: PaperTradingDB | None = None, fx_usdkrw: float = 1380.0) -> None:
        self.db = db or PaperTradingDB()
        self.default_fx = fx_usdkrw
        self.db.ensure_account(DEFAULT_INITIAL_CASH_KRW, fx_usdkrw=fx_usdkrw)

    def _fx_for_currency(self, currency: str, fx_usdkrw: float | None = None) -> float:
        if currency == "KRW":
            return 1.0
        if fx_usdkrw and fx_usdkrw > 500:
            return float(fx_usdkrw)
        return float(self.db.get_account().get("fx_usdkrw", self.default_fx))

    def buy(
        self,
        market: str,
        code: str,
        name: str,
        currency: str,
        price_local: float,
        budget_krw: float,
        fx_usdkrw: float | None = None,
        stop_loss_local: float | None = None,
        target_price_local: float | None = None,
        reason: str = "AI 매수 시그널",
        ai_action: str = "매수",
        ai_score: float = 0.0,
        up_prob: float = 0.5,
    ) -> dict[str, Any]:
        """지정된 원화 예산으로 모의 매수 체결 (수수료·슬리피지 반영)."""
        acc = self.db.get_account()
        cash = float(acc["cash_krw"])
        fx = self._fx_for_currency(currency, fx_usdkrw)
        costs = MARKET_COST_BPS.get(market, MARKET_COST_BPS["US"])
        buy_cost_rate = costs["commission"] + costs["slippage"]

        usable_krw = min(float(budget_krw), cash)
        if usable_krw < 10_000 or price_local <= 0:
            raise ValueError("주문 가능 현금이 부족하거나 주가가 유효하지 않습니다.")

        # 수수료 포함해 예산 내 매수 가능 수량 산출 (국내=정수주, 해외=소수점 3자리 지원)
        raw_shares = (usable_krw / (1.0 + buy_cost_rate)) / (price_local * fx)
        shares = float(int(raw_shares)) if market == "KR" else round(float(raw_shares), 3)
        if shares <= 0:
            raise ValueError(f"예산({usable_krw:,.0f}원)으로 1주 이상 매수할 수 없습니다 (주가 {price_local:,.2f}).")

        gross_krw = shares * price_local * fx
        fee_krw = gross_krw * buy_cost_rate
        net_krw = gross_krw + fee_krw
        if net_krw > cash:
            # 1주 줄여 재계산
            shares = max(shares - (1.0 if market == "KR" else 0.01), 0.0)
            gross_krw = shares * price_local * fx
            fee_krw = gross_krw * buy_cost_rate
            net_krw = gross_krw + fee_krw
        if shares <= 0 or net_krw > cash:
            raise ValueError("수수료 포함 주문 금액이 가용 현금을 초과합니다.")

        now = datetime.now().isoformat(timespec="seconds")
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE account SET cash_krw = cash_krw - ?, fx_usdkrw = ?, updated_at = ? WHERE id=1",
                (net_krw, fx if currency == "USD" else acc["fx_usdkrw"], now),
            )
            pos = conn.execute("SELECT * FROM positions WHERE market=? AND code=?", (market, code)).fetchone()
            if pos:
                old_sh = float(pos["shares"])
                new_sh = old_sh + shares
                new_avg_local = (old_sh * float(pos["avg_cost_local"]) + shares * price_local) / new_sh
                new_avg_krw = (old_sh * float(pos["avg_cost_krw"]) + net_krw) / new_sh
                conn.execute(
                    """
                    UPDATE positions
                    SET shares=?, avg_cost_local=?, avg_cost_krw=?, current_price_local=?,
                        stop_loss_local=COALESCE(?, stop_loss_local),
                        target_price_local=COALESCE(?, target_price_local),
                        ai_action=?, ai_score=?, rationale=?, updated_at=?
                    WHERE id=?
                    """,
                    (new_sh, new_avg_local, new_avg_krw, price_local, stop_loss_local, target_price_local, ai_action, ai_score, reason, now, pos["id"]),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO positions (
                        market, code, name, currency, shares, avg_cost_local, avg_cost_krw,
                        current_price_local, stop_loss_local, target_price_local,
                        ai_action, ai_score, fund_grade, rationale, opened_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        market, code, name, currency, shares, price_local, net_krw / shares,
                        price_local, stop_loss_local, target_price_local, ai_action, ai_score, "", reason, now, now,
                    ),
                )

            conn.execute(
                """
                INSERT INTO orders (
                    executed_at, side, market, code, name, currency, shares, price_local,
                    fx_rate, gross_krw, fee_krw, net_krw, realized_pnl_krw, realized_pnl_pct,
                    reason, ai_score, up_prob
                ) VALUES (?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                """,
                (
                    now, market, code, name, currency, shares, price_local, fx,
                    gross_krw, fee_krw, net_krw, reason, ai_score, up_prob,
                ),
            )

        self.record_snapshot()
        return {
            "side": "BUY",
            "code": code,
            "name": name,
            "shares": shares,
            "price_local": price_local,
            "net_krw": net_krw,
            "fee_krw": fee_krw,
        }

    def sell(
        self,
        market: str,
        code: str,
        price_local: float,
        sell_ratio: float = 1.0,
        fx_usdkrw: float | None = None,
        reason: str = "수동/리밸런싱 매도",
    ) -> dict[str, Any]:
        """보유 종목의 일정 비율(기본 전량 1.0)을 모의 매도 체결 (거래세·수수료·슬리피지 반영)."""
        with self.db._connect() as conn:
            pos = conn.execute("SELECT * FROM positions WHERE market=? AND code=?", (market, code)).fetchone()
            if pos is None:
                raise ValueError(f"보유 중인 종목이 아닙니다: {code}")

            total_sh = float(pos["shares"])
            sell_sh = total_sh if sell_ratio >= 0.999 else (float(int(total_sh * sell_ratio)) if market == "KR" else round(total_sh * sell_ratio, 3))
            if sell_sh <= 0:
                raise ValueError("매도 수량이 0 이하입니다.")

            currency = str(pos["currency"])
            fx = self._fx_for_currency(currency, fx_usdkrw)
            costs = MARKET_COST_BPS.get(market, MARKET_COST_BPS["US"])
            sell_cost_rate = costs["commission"] + costs["sell_tax"] + costs["slippage"]

            gross_krw = sell_sh * price_local * fx
            fee_krw = gross_krw * sell_cost_rate
            net_krw = gross_krw - fee_krw
            cost_basis_krw = sell_sh * float(pos["avg_cost_krw"])
            realized_pnl_krw = net_krw - cost_basis_krw
            realized_pnl_pct = (realized_pnl_krw / (cost_basis_krw + 1e-9)) * 100.0

            now = datetime.now().isoformat(timespec="seconds")
            conn.execute("UPDATE account SET cash_krw = cash_krw + ?, updated_at = ? WHERE id=1", (net_krw, now))

            rem_sh = total_sh - sell_sh
            if rem_sh <= 1e-6:
                conn.execute("DELETE FROM positions WHERE id=?", (pos["id"],))
            else:
                conn.execute(
                    "UPDATE positions SET shares=?, current_price_local=?, updated_at=? WHERE id=?",
                    (rem_sh, price_local, now, pos["id"]),
                )

            conn.execute(
                """
                INSERT INTO orders (
                    executed_at, side, market, code, name, currency, shares, price_local,
                    fx_rate, gross_krw, fee_krw, net_krw, realized_pnl_krw, realized_pnl_pct,
                    reason, ai_score, up_prob
                ) VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    now, market, code, str(pos["name"]), currency, sell_sh, price_local, fx,
                    gross_krw, fee_krw, net_krw, realized_pnl_krw, realized_pnl_pct,
                    reason, float(pos["ai_score"] or 0.0),
                ),
            )

        self.record_snapshot()
        return {
            "side": "SELL",
            "code": code,
            "name": str(pos["name"]),
            "shares": sell_sh,
            "net_krw": net_krw,
            "realized_pnl_krw": realized_pnl_krw,
            "realized_pnl_pct": realized_pnl_pct,
        }

    def execute_ai_portfolio_from_screening(
        self,
        screening_df: pd.DataFrame,
        fx_usdkrw: float = 1380.0,
        max_total_invest_ratio: float | None = None,
        max_single_position_ratio: float = 0.18,
    ) -> list[dict[str, Any]]:
        """AI 스크리닝 결과 중 '매수/강력 매수' 종목들을 폴리마켓 날씨 비중·변동성 비율대로 자동 분할 매수."""
        if screening_df is None or screening_df.empty:
            return []

        if max_total_invest_ratio is None:
            try:
                from ..data.alt_intelligence import AltMarketIntelligenceEngine

                pulse = AltMarketIntelligenceEngine().get_macro_pulse(use_cache=True)
                max_total_invest_ratio = float(pulse.recommended_invest_cap)
            except Exception:
                max_total_invest_ratio = 0.70

        buys = screening_df[screening_df["최종의견"].astype(str).str.contains("매수", na=False)].copy()
        if buys.empty:
            # 매수 종목이 없으면 점수 상위 중 권장비중 > 0 종목 선택
            buys = screening_df[screening_df["권장비중(%)"] > 0].head(3).copy()
        if buys.empty:
            return []

        acc = self.db.get_account()
        total_eq = self.get_portfolio_summary(fx_usdkrw=fx_usdkrw)["total_equity_krw"]
        # 권장비중 합으로 정규화하되 총 투자 한도와 종목별 상한 준수
        w_raw = buys["권장비중(%)"].astype(float).clip(lower=10.0).to_numpy()
        w_norm = w_raw / np.sum(w_raw)
        target_total_budget = min(total_eq * max_total_invest_ratio, float(acc["cash_krw"]))

        executed: list[dict[str, Any]] = []
        existing = self.db.get_positions_df()
        held_codes = set(existing["code"].astype(str)) if not existing.empty else set()

        ret_col = next((c for c in buys.columns if "예상가" in c), None)
        risk_caps = {"낮음": 0.16, "보통": 0.16, "높음": 0.11, "매우 높음": 0.07}
        for (_, row), weight in zip(buys.iterrows(), w_norm):
            code = str(row["티커"])
            if code in held_codes:
                continue
            mkt = "KR" if "KR" in str(row["시장"]) else "US"
            risk_lvl = str(row.get("리스크", "보통"))
            pos_cap = min(max_single_position_ratio, risk_caps.get(risk_lvl, 0.11))
            budget = min(target_total_budget * float(weight), total_eq * pos_cap)
            try:
                res = self.buy(
                    market=mkt,
                    code=code,
                    name=str(row["종목명"]),
                    currency=str(row["통화"]),
                    price_local=float(row["현재가"]),
                    budget_krw=budget,
                    fx_usdkrw=fx_usdkrw,
                    stop_loss_local=float(row["하한_10%"]) if "하한_10%" in row else None,
                    target_price_local=float(row[ret_col]) if ret_col else None,
                    reason=(
                        f"AI {row['최종의견']} (점수 {float(row['위원회_점수']):+.1f}, "
                        f"상승확률 {float(row['상승확률(%)']):.1f}%, 재무 {row.get('재무등급', '')}, "
                        f"이슈 {float(row.get('이슈감성', 0)):+.2f}"
                        + (f" · {row['큰손_한줄요약']}" if row.get("큰손_한줄요약") else "")
                        + ")"
                    ),
                    ai_action=str(row["최종의견"]),
                    ai_score=float(row["위원회_점수"]),
                    up_prob=float(row["상승확률(%)"]) / 100.0,
                )
                # 재무등급 업데이트
                with self.db._connect() as conn:
                    conn.execute(
                        "UPDATE positions SET fund_grade=? WHERE market=? AND code=?",
                        (str(row.get("재무등급", "")), mkt, code),
                    )
                executed.append(res)
            except Exception:
                continue
        return executed

    def refresh_prices(self, price_map: dict[str, float], fx_usdkrw: float | None = None) -> None:
        """보유 종목의 현재가 및 환율을 최신 시장가로 업데이트하고 일별 스냅샷 갱신."""
        now = datetime.now().isoformat(timespec="seconds")
        with self.db._connect() as conn:
            if fx_usdkrw and fx_usdkrw > 500:
                conn.execute("UPDATE account SET fx_usdkrw=?, updated_at=? WHERE id=1", (float(fx_usdkrw), now))
            for code, price in price_map.items():
                if price and price > 0:
                    conn.execute(
                        "UPDATE positions SET current_price_local=?, updated_at=? WHERE code=?",
                        (float(price), now, str(code)),
                    )
        self.record_snapshot()

    def get_portfolio_summary(self, fx_usdkrw: float | None = None) -> dict[str, Any]:
        acc = self.db.get_account()
        fx = float(fx_usdkrw) if (fx_usdkrw and fx_usdkrw > 500) else float(acc.get("fx_usdkrw", self.default_fx))
        cash_krw = float(acc["cash_krw"])
        initial_krw = float(acc["initial_cash_krw"])
        pos_df = self.db.get_positions_df()
        orders_df = self.db.get_orders_df(limit=500)

        invested_val_krw = 0.0
        cost_basis_krw = 0.0
        enriched_rows: list[dict[str, Any]] = []

        if not pos_df.empty:
            for _, p in pos_df.iterrows():
                p_fx = 1.0 if p["currency"] == "KRW" else fx
                val_krw = float(p["shares"]) * float(p["current_price_local"]) * p_fx
                cost_krw = float(p["shares"]) * float(p["avg_cost_krw"])
                pnl_krw = val_krw - cost_krw
                pnl_pct = (pnl_krw / (cost_krw + 1e-9)) * 100.0
                avg_local = float(p["avg_cost_local"])
                cur_local = float(p["current_price_local"])
                local_ret_pct = ((cur_local / avg_local) - 1.0) * 100.0 if avg_local > 0 else 0.0
                invested_val_krw += val_krw
                cost_basis_krw += cost_krw
                enriched_rows.append(
                    {
                        "시장": "국내(KR)" if p["market"] == "KR" else "해외(US)",
                        "종목명": p["name"],
                        "티커": p["code"],
                        "통화": p["currency"],
                        "보유수량": round(float(p["shares"]), 3),
                        "매입단가": round(avg_local, 2),
                        "현재가": round(cur_local, 2),
                        "평가금액(원)": round(val_krw, 0),
                        "평가손익(원)": round(pnl_krw, 0),
                        "주가수익률(%)": round(local_ret_pct, 2),
                        "수익률(%)": round(pnl_pct, 2),
                        "목표가": round(float(p["target_price_local"]), 2) if p["target_price_local"] else None,
                        "손절가": round(float(p["stop_loss_local"]), 2) if p["stop_loss_local"] else None,
                        "재무등급": p["fund_grade"] or "-",
                        "AI점수": round(float(p["ai_score"] or 0.0), 1),
                        "투자근거": p["rationale"] or "",
                    }
                )

        total_equity = cash_krw + invested_val_krw
        total_return_pct = (total_equity / initial_krw - 1.0) * 100.0
        unrealized_pnl = invested_val_krw - cost_basis_krw
        realized_pnl = float(orders_df["realized_pnl_krw"].dropna().sum()) if not orders_df.empty else 0.0
        total_fees = float(orders_df["fee_krw"].dropna().sum()) if not orders_df.empty else 0.0

        enriched_df = pd.DataFrame(enriched_rows)
        if not enriched_df.empty:
            enriched_df["비중(%)"] = (enriched_df["평가금액(원)"] / (total_equity + 1e-9) * 100.0).round(1)

        return {
            "initial_cash_krw": initial_krw,
            "cash_krw": cash_krw,
            "invested_krw": invested_val_krw,
            "total_equity_krw": total_equity,
            "total_return_pct": total_return_pct,
            "unrealized_pnl_krw": unrealized_pnl,
            "realized_pnl_krw": realized_pnl,
            "total_fees_krw": total_fees,
            "fx_usdkrw": fx,
            "n_positions": len(enriched_rows),
            "positions_df": enriched_df,
        }

    def record_snapshot(self, snap_date: str | None = None) -> None:
        summary = self.get_portfolio_summary()
        dt_str = snap_date or datetime.now().strftime("%Y-%m-%d")
        with self.db._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_snapshots
                (snap_date, total_equity_krw, cash_krw, invested_krw, return_pct, n_positions)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dt_str,
                    summary["total_equity_krw"],
                    summary["cash_krw"],
                    summary["invested_krw"],
                    summary["total_return_pct"],
                    summary["n_positions"],
                ),
            )


class HistoricalPaperSimulator:
    """국내·해외 멀티종목 Strict Walk-Forward AI 모의투자 시뮬레이터 & 학습 데이터셋 생성기.

    - 실제 과거 1년(약 250거래일) 동안 AI 모델이 미래를 전혀 보지 않고(expanding window + embargo)
      매주(5거래일마다) 재학습·예측해 상위 종목을 매수하고, 손절(-8%)/익절(+18%)/리밸런싱을 수행.
    - 매 매매마다 진입 시점의 핵심 피처, AI 예측치, 실제 실현 수익률을 학습 데이터셋으로 자동 축적!
    """

    def __init__(
        self,
        config: PredictorConfig | None = None,
        initial_capital_krw: float = DEFAULT_INITIAL_CASH_KRW,
        max_positions: int = 5,
        rebalance_every: int = 5,
        retrain_every: int = 20,
        stop_loss_pct: float = -0.08,
        take_profit_pct: float = 0.18,
    ) -> None:
        self.config = config or PredictorConfig(horizon=5)
        self.initial_capital_krw = initial_capital_krw
        self.max_positions = max_positions
        self.rebalance_every = rebalance_every
        self.retrain_every = retrain_every
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.feature_engine = FeatureEngine(horizon=self.config.horizon)

    def run(
        self,
        bundles: list[MarketDataBundle],
        sim_days: int = 240,
        db: PaperTradingDB | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> SimulationReport:
        """멀티종목(국내+해외) 번들 리스트로 실제 워크포워드 모의투자 실행 및 데이터셋 생성."""
        if len(bundles) < 3:
            raise ValueError("멀티종목 모의투자 시뮬레이션에는 최소 3개 이상의 종목이 필요합니다.")

        if progress_cb:
            progress_cb(f"모의투자 패널 데이터 구축 중 ({len(bundles)}개 종목)...")

        # 1) 모든 종목 피처·가격·환율 정렬
        sym_meta: dict[str, dict[str, Any]] = {}
        price_map: dict[str, pd.Series] = {}
        panel_parts: list[pd.DataFrame] = []
        fx_series: pd.Series | None = None

        for b in bundles:
            try:
                ds = self.feature_engine.build(b)
                if len(ds.train_X) < 160:
                    continue
                code = b.symbol.code
                sym_meta[code] = {
                    "market": b.symbol.market,
                    "code": code,
                    "name": b.symbol.name,
                    "currency": b.symbol.currency,
                }
                price_map[code] = b.ohlcv["Close"].copy()
                if fx_series is None and b.macro is not None and "usdkrw" in b.macro.columns:
                    fx_series = b.macro["usdkrw"].dropna().sort_index()

                sub = ds.features_all.copy()
                sub["_date"] = sub.index
                sub["_code"] = code
                sub["_target_ret"] = np.log(
                    (b.ohlcv["Close"].shift(-self.config.horizon) / b.ohlcv["Close"]).reindex(sub.index).clip(lower=1e-6)
                )
                panel_parts.append(sub.reset_index(drop=True))
            except Exception:
                continue

        if len(panel_parts) < 3:
            raise RuntimeError("시뮬레이션 가능한 유효 종목 수가 부족합니다.")

        meta_cols = {"_date", "_code", "_target_ret"}
        common_feats = set(panel_parts[0].columns) - meta_cols
        for p in panel_parts[1:]:
            common_feats &= (set(p.columns) - meta_cols)
        feature_cols = sorted(common_feats)

        panel = pd.concat([p[feature_cols + list(meta_cols)] for p in panel_parts], ignore_index=True)
        panel = panel.sort_values(["_date", "_code"]).reset_index(drop=True)

        # 일별 Cross-Sectional 순위 피처 추가
        rank_cols = [c for c in ["ret_5d", "ret_20d", "rsi_14", "rv_20d", "dist_sma_20", "drawdown_60d"] if c in feature_cols]
        for col in rank_cols:
            rk = f"cs_rank_{col}"
            panel[rk] = panel.groupby("_date")[col].rank(pct=True)
            feature_cols.append(rk)

        # Excess Alpha 타겟
        panel["_alpha_ret"] = panel["_target_ret"] - panel.groupby("_date")["_target_ret"].transform("median")
        panel["_alpha_dir"] = (panel["_alpha_ret"] > 0).astype(int)

        # 가격 매트릭스 & 시뮬레이션 날짜 선정
        close_df = pd.DataFrame(price_map).sort_index().ffill()
        valid_dates = sorted(
            dt for dt in panel["_date"].unique()
            if dt in close_df.index and close_df.loc[dt].notna().sum() >= 3
        )
        if len(valid_dates) < 160:
            raise RuntimeError("워크포워드 시뮬레이션을 위한 공통 거래일이 부족합니다.")

        actual_sim_days = min(sim_days, max(len(valid_dates) - 140, 60))
        sim_dates = valid_dates[-actual_sim_days:]

        if fx_series is not None and not fx_series.empty:
            fx_aligned = fx_series.reindex(close_df.index, method="ffill").bfill().fillna(1380.0)
        else:
            fx_aligned = pd.Series(1380.0, index=close_df.index)

        # 벤치마크: 참여 종목 전체 동일가중 Buy & Hold (원화 환산)
        bm_values = pd.DataFrame(index=sim_dates)
        for code, meta in sym_meta.items():
            s = close_df[code].reindex(sim_dates).ffill().bfill()
            fx_mult = 1.0 if meta["currency"] == "KRW" else fx_aligned.reindex(sim_dates).ffill().bfill()
            krw_p = s * fx_mult
            bm_values[code] = krw_p / krw_p.iloc[0]
        bm_equity_series = bm_values.mean(axis=1) * self.initial_capital_krw

        # 2) Strict Expanding Walk-Forward 일별 모의투자 루프
        cash = float(self.initial_capital_krw)
        # holdings[code] = dict(shares, entry_price, entry_krw, entry_date, pred_ret, pred_prob, entry_feats, days_held)
        holdings: dict[str, dict[str, Any]] = {}
        trade_records: list[dict[str, Any]] = []
        learning_rows: list[dict[str, Any]] = []
        eq_rows: list[dict[str, Any]] = []
        total_fees = 0.0

        models: list[Any] = []
        embargo = max(self.config.horizon, 1)

        for day_i, dt in enumerate(sim_dates):
            dt_ts = pd.Timestamp(dt)
            dt_str = dt_ts.strftime("%Y-%m-%d")
            fx_today = float(fx_aligned.loc[dt_ts])

            # (a) 주기적 재학습 (과거 데이터만 사용, embargo 적용!)
            if day_i % self.retrain_every == 0 or not models:
                if progress_cb:
                    progress_cb(f"AI 모의투자 진행 중 ({day_i + 1}/{len(sim_dates)}일 · {dt_str}) — 모델 재학습...")
                dt_pos = valid_dates.index(dt)
                cutoff_date = valid_dates[max(0, dt_pos - embargo - 1)]
                tr = panel[(panel["_date"] <= cutoff_date) & (panel["_alpha_ret"].notna())]
                if len(tr) >= 120:
                    X_tr = tr[feature_cols].fillna(0.0)
                    yr_tr = tr["_alpha_ret"]
                    yd_tr = tr["_alpha_dir"]
                    models = [
                        LightGBMStockModel(random_state=self.config.random_seed),
                        LinearQuantModel(random_state=self.config.random_seed + 1),
                        ExtraTreesStockModel(random_state=self.config.random_seed + 2),
                    ]
                    for m in models:
                        m.fit(X_tr, yr_tr, yd_tr)

            # (b) 일일 손절/익절 체크
            for code in list(holdings.keys()):
                h = holdings[code]
                h["days_held"] += 1
                px = float(close_df.loc[dt_ts, code])
                if np.isnan(px) or px <= 0:
                    continue
                cur_ret = (px / h["entry_price"]) - 1.0
                exit_reason = None
                if cur_ret <= self.stop_loss_pct:
                    exit_reason = "손절(Stop-Loss)"
                elif cur_ret >= self.take_profit_pct:
                    exit_reason = "익절(Take-Profit)"

                if exit_reason:
                    cash, fee = self._close_position(
                        code, h, px, dt_str, fx_today, sym_meta[code], exit_reason,
                        cash, trade_records, learning_rows,
                    )
                    total_fees += fee
                    del holdings[code]

            # (c) 리밸런싱 주기에 따라 AI 예측 기반 교체 매매
            if day_i % self.rebalance_every == 0 and models:
                today_slice = panel[panel["_date"] == dt].copy()
                if not today_slice.empty:
                    X_today = today_slice[feature_cols].fillna(0.0)
                    preds_ret = np.mean([m.predict(X_today).expected_return for m in models], axis=0)
                    preds_prob = np.mean([m.predict(X_today).up_prob for m in models], axis=0)
                    today_slice["_pred_ret"] = preds_ret
                    today_slice["_pred_prob"] = preds_prob
                    # 변동성 패널티 반영한 리스크 조정 알파 스코어
                    rv = today_slice["rv_20d"].clip(lower=0.15, upper=1.2).to_numpy(dtype=float) if "rv_20d" in today_slice.columns else np.full(len(today_slice), 0.3)
                    today_slice["_score"] = preds_ret / np.sqrt(rv) + (preds_prob - 0.5) * 0.08
                    ranked = today_slice.sort_values("_score", ascending=False)

                    # 약세장 방어: VIX가 매우 높거나(>32) 상승확률>0.52 종목이 적으면 현금 비중 확대
                    vix_now = float(today_slice["vix_level"].iloc[0] * 100.0) if "vix_level" in today_slice.columns else 18.0
                    invest_cap = 0.65 if vix_now >= 28.0 else 0.90

                    # 매수 후보: 하락 추세(60일선 -5% 하회) 제외 + 기대 알파 양수/상승확률 우위 종목 선별
                    if "dist_sma_60" in ranked.columns:
                        trend_ok = ranked["dist_sma_60"] >= -0.05
                    else:
                        trend_ok = pd.Series(True, index=ranked.index)
                    candidates = ranked[trend_ok & (ranked["_pred_ret"] > 0.0) & (ranked["_pred_prob"] >= 0.52)].head(self.max_positions)
                    # 회전율·수수료 절감 버퍼: 기존 보유 종목은 상위 (max_positions + 3)위 이내이고 추세 유지 시 계속 보유
                    keep_buffer = set(ranked[trend_ok].head(self.max_positions + 3)["_code"].astype(str))
                    target_codes = set(candidates["_code"].astype(str))

                    # 탈락 종목 청산 (버퍼 밖으로 밀렸거나 추세 이탈한 종목만 매도해 거래세·수수료 절감)
                    for code in list(holdings.keys()):
                        if code not in target_codes and code not in keep_buffer:
                            px = float(close_df.loc[dt_ts, code])
                            if not np.isnan(px) and px > 0:
                                cash, fee = self._close_position(
                                    code, holdings[code], px, dt_str, fx_today, sym_meta[code],
                                    "추세 이탈 / AI 순위 하락", cash, trade_records, learning_rows,
                                )
                                total_fees += fee
                                del holdings[code]

                    # 현재 총자산 산출 후 신규 후보 종목 진입 (변동성 역가중 포지션 사이징)
                    cur_equity = cash + sum(
                        h["shares"] * float(close_df.loc[dt_ts, c]) * (1.0 if sym_meta[c]["currency"] == "KRW" else fx_today)
                        for c, h in holdings.items()
                    )
                    new_cands = candidates[~candidates["_code"].isin(holdings.keys())]
                    if not new_cands.empty and cash > cur_equity * 0.05:
                        max_invest_total = cur_equity * invest_cap
                        cur_invested = cur_equity - cash
                        avail_budget = max(min(cash, max_invest_total - cur_invested), 0.0)
                        if avail_budget > 200_000:
                            inv_vol = 1.0 / (new_cands["rv_20d"].clip(lower=0.15, upper=1.2).to_numpy(dtype=float) if "rv_20d" in new_cands.columns else np.ones(len(new_cands)))
                            weights = inv_vol / np.sum(inv_vol)
                            for (_, row), w in zip(new_cands.iterrows(), weights):
                                code = str(row["_code"])
                                px = float(close_df.loc[dt_ts, code])
                                if np.isnan(px) or px <= 0:
                                    continue
                                meta = sym_meta[code]
                                fx_m = 1.0 if meta["currency"] == "KRW" else fx_today
                                buy_rate = MARKET_COST_BPS[meta["market"]]["commission"] + MARKET_COST_BPS[meta["market"]]["slippage"]
                                budget = min(avail_budget * float(w), cur_equity * 0.24, cash)
                                raw_sh = (budget / (1.0 + buy_rate)) / (px * fx_m)
                                sh = float(int(raw_sh)) if meta["market"] == "KR" else round(float(raw_sh), 3)
                                if sh <= 0:
                                    continue
                                gross = sh * px * fx_m
                                fee = gross * buy_rate
                                net = gross + fee
                                if net > cash:
                                    continue
                                cash -= net
                                total_fees += fee
                                snap_feats = {
                                    f: round(float(row[f]), 5)
                                    for f in ["ret_5d", "ret_20d", "rsi_14", "rv_20d", "macd_hist_pct", "bb_zscore_20", "rel_alpha_20d", "vix_level"]
                                    if f in row.index
                                }
                                entry_rationale = self._build_trade_rationale(
                                    float(row["_pred_ret"]), float(row["_pred_prob"]), snap_feats
                                )
                                holdings[code] = {
                                    "shares": sh,
                                    "entry_price": px,
                                    "entry_krw": net,
                                    "entry_date": dt_str,
                                    "pred_ret": float(row["_pred_ret"]),
                                    "pred_prob": float(row["_pred_prob"]),
                                    "entry_feats": snap_feats,
                                    "entry_rationale": entry_rationale,
                                    "days_held": 0,
                                }

            # (d) 일별 계좌 평가 기록
            invested_now = sum(
                h["shares"] * float(close_df.loc[dt_ts, c]) * (1.0 if sym_meta[c]["currency"] == "KRW" else fx_today)
                for c, h in holdings.items()
                if not np.isnan(close_df.loc[dt_ts, c])
            )
            eq_now = cash + invested_now
            eq_rows.append(
                {
                    "date": dt_ts,
                    "equity_krw": eq_now,
                    "benchmark_krw": float(bm_equity_series.loc[dt_ts]),
                    "cash_krw": cash,
                    "invested_pct": (invested_now / (eq_now + 1e-9)) * 100.0,
                    "n_holdings": len(holdings),
                }
            )

        # 시뮬레이션 종료일 잔여 포지션 평가 청산 기록 (데이터셋 완성용)
        last_dt = pd.Timestamp(sim_dates[-1])
        last_str = last_dt.strftime("%Y-%m-%d")
        last_fx = float(fx_aligned.loc[last_dt])
        for code in list(holdings.keys()):
            px = float(close_df.loc[last_dt, code])
            if not np.isnan(px) and px > 0:
                cash, fee = self._close_position(
                    code, holdings[code], px, last_str, last_fx, sym_meta[code],
                    "기간종료 평가청산", cash, trade_records, learning_rows,
                )
                total_fees += fee
                del holdings[code]

        # 3) 성과 지표 집계
        eq_df = pd.DataFrame(eq_rows).set_index("date")
        eq_df["drawdown_pct"] = (eq_df["equity_krw"] / eq_df["equity_krw"].cummax() - 1.0) * 100.0
        eq_df["benchmark_dd_pct"] = (eq_df["benchmark_krw"] / eq_df["benchmark_krw"].cummax() - 1.0) * 100.0

        daily_ret = eq_df["equity_krw"].pct_change().fillna(0.0)
        bm_ret = eq_df["benchmark_krw"].pct_change().fillna(0.0)
        n_days = len(eq_df)
        years = max(n_days / 252.0, 1.0 / 252.0)

        final_eq = float(eq_df["equity_krw"].iloc[-1])
        tot_ret = (final_eq / self.initial_capital_krw - 1.0) * 100.0
        bm_tot = (float(eq_df["benchmark_krw"].iloc[-1]) / self.initial_capital_krw - 1.0) * 100.0
        cagr = ((max(final_eq / self.initial_capital_krw, 1e-6)) ** (1.0 / years) - 1.0) * 100.0
        sharpe = float((daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252))
        bm_sharpe = float((bm_ret.mean() / (bm_ret.std() + 1e-9)) * np.sqrt(252))

        trade_df = pd.DataFrame(trade_records)
        if not trade_df.empty:
            wins = trade_df[trade_df["pnl_krw"] > 0]
            losses = trade_df[trade_df["pnl_krw"] <= 0]
            win_rate = float(len(wins) / len(trade_df) * 100.0)
            avg_win = float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
            gp = float(wins["pnl_krw"].sum())
            gl = float(np.abs(losses["pnl_krw"].sum()))
            pf = gp / gl if gl > 1e-9 else (2.5 if gp > 0 else 0.0)
        else:
            win_rate = avg_win = avg_loss = pf = 0.0

        # 4) 모의투자 자가학습 데이터셋 CSV/JSON 저장
        PAPER_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        run_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_path = PAPER_DATASET_DIR / f"ai_paper_learning_dataset_{run_id}.csv"
        if learning_rows:
            pd.DataFrame(learning_rows).to_csv(dataset_path, index=False, encoding="utf-8-sig")
            # 최신 데이터셋 심볼릭 복사본도 유지
            pd.DataFrame(learning_rows).to_csv(PAPER_DATASET_DIR / "latest_learning_dataset.csv", index=False, encoding="utf-8-sig")

        sym_pnl_df = _build_symbol_pnl_summary(trade_df)
        report = SimulationReport(
            run_id=run_id,
            created_at=datetime.now().isoformat(timespec="seconds"),
            start_date=eq_df.index[0].strftime("%Y-%m-%d"),
            end_date=eq_df.index[-1].strftime("%Y-%m-%d"),
            n_days=n_days,
            n_symbols=len(sym_meta),
            initial_capital_krw=self.initial_capital_krw,
            final_equity_krw=final_eq,
            total_return_pct=round(tot_ret, 2),
            benchmark_return_pct=round(bm_tot, 2),
            excess_return_pct=round(tot_ret - bm_tot, 2),
            cagr_pct=round(cagr, 2),
            sharpe_ratio=round(sharpe, 2),
            benchmark_sharpe=round(bm_sharpe, 2),
            max_drawdown_pct=round(float(eq_df["drawdown_pct"].min()), 2),
            benchmark_mdd_pct=round(float(eq_df["benchmark_dd_pct"].min()), 2),
            total_trades=len(trade_df) * 2,
            closed_trades=len(trade_df),
            win_rate_pct=round(win_rate, 1),
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
            profit_factor=round(min(pf, 99.0), 2),
            total_fees_krw=round(total_fees, 0),
            avg_invested_pct=round(float(eq_df["invested_pct"].mean()), 1),
            symbols_used=list(sym_meta.keys()),
            equity_curve=eq_df,
            trade_log=trade_df,
            symbol_pnl=sym_pnl_df,
            learning_dataset_path=str(dataset_path) if learning_rows else "",
        )
        if db is not None:
            db.save_simulation(report)
        return report

    @staticmethod
    def _close_position(
        code: str,
        holding: dict[str, Any],
        exit_price: float,
        exit_date: str,
        fx_today: float,
        meta: dict[str, Any],
        exit_reason: str,
        cash: float,
        trade_records: list[dict[str, Any]],
        learning_rows: list[dict[str, Any]],
    ) -> tuple[float, float]:
        fx_m = 1.0 if meta["currency"] == "KRW" else fx_today
        sell_rate = (
            MARKET_COST_BPS[meta["market"]]["commission"]
            + MARKET_COST_BPS[meta["market"]]["sell_tax"]
            + MARKET_COST_BPS[meta["market"]]["slippage"]
        )
        gross = holding["shares"] * exit_price * fx_m
        fee = gross * sell_rate
        net = gross - fee
        pnl_krw = net - holding["entry_krw"]
        pnl_pct = (pnl_krw / (holding["entry_krw"] + 1e-9)) * 100.0
        new_cash = cash + net

        trade_records.append(
            {
                "entry_date": holding["entry_date"],
                "exit_date": exit_date,
                "market": meta["market"],
                "code": code,
                "name": meta["name"],
                "entry_price": round(holding["entry_price"], 2),
                "exit_price": round(exit_price, 2),
                "shares": round(holding["shares"], 3),
                "invested_krw": round(holding["entry_krw"], 0),
                "pnl_krw": round(pnl_krw, 0),
                "pnl_pct": round(pnl_pct, 2),
                "fee_krw": round(fee, 0),
                "exit_reason": exit_reason,
                "pred_ret_pct": round(holding["pred_ret"] * 100.0, 2),
                "pred_prob_pct": round(holding["pred_prob"] * 100.0, 1),
                "holding_days": int(max(holding["days_held"], 1)),
                "entry_rationale": str(holding.get("entry_rationale", "")),
            }
        )
        row_learn = {
            "entry_date": holding["entry_date"],
            "exit_date": exit_date,
            "market": meta["market"],
            "code": code,
            "name": meta["name"],
            "pred_ret_pct": round(holding["pred_ret"] * 100.0, 3),
            "pred_prob_pct": round(holding["pred_prob"] * 100.0, 2),
            "realized_pnl_pct": round(pnl_pct, 3),
            "realized_win": 1 if pnl_krw > 0 else 0,
            "exit_reason": exit_reason,
            "holding_days": int(max(holding["days_held"], 1)),
            "entry_rationale": str(holding.get("entry_rationale", "")),
        }
        row_learn.update(holding.get("entry_feats", {}))
        learning_rows.append(row_learn)
        return new_cash, fee

    @staticmethod
    def _build_trade_rationale(pred_ret: float, pred_prob: float, feats: dict[str, float]) -> str:
        """진입 시점 피처와 AI 예측치를 바탕으로 사람이 읽기 쉬운 한국어 매수 근거 자동 생성."""
        parts = [f"AI 초과수익 예측 {pred_ret * 100:+.2f}% · 상승확률 {pred_prob * 100:.1f}%"]
        rsi = feats.get("rsi_14", 0.5) * 100.0
        rel20 = feats.get("rel_alpha_20d", 0.0) * 100.0
        ret20 = feats.get("ret_20d", 0.0) * 100.0
        rv20 = feats.get("rv_20d", 0.3) * 100.0
        if rel20 >= 3.0:
            parts.append(f"시장 대비 20일 상대강도 +{rel20:.1f}% 주도주")
        elif rsi <= 36.0:
            parts.append(f"RSI {rsi:.0f} 단기 과매도 반등 포착")
        elif ret20 > 2.0:
            parts.append(f"20일 추세 +{ret20:.1f}% 모멘텀 유지")
        if feats.get("macd_hist_pct", 0.0) > 0:
            parts.append("MACD 매수 모멘텀 양전환")
        if rv20 <= 30.0:
            parts.append(f"안정적 변동성({rv20:.0f}%)")
        return " | ".join(parts)
