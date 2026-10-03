"""SQLite 持久化：作业、当前最好方案、下界。

数据文件路径由环境变量 DISPATCH_DB_PATH 指定，默认 /data/dispatch.db，
便于挂卷。WAL 模式下读写并发良好；写操作在进程内用一把锁串行化。
方案只存 (车型, 货号列表)，重量体积等统计读回时重算，避免不一致。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id              TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    seed                INTEGER NOT NULL,
    time_limit_seconds  REAL NOT NULL,
    request_json        TEXT NOT NULL,
    lb_json             TEXT,
    best_cost           REAL,
    best_vehicles       INTEGER,
    solution_json       TEXT,
    iterations          INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds     REAL NOT NULL DEFAULT 0,
    stop_reason         TEXT,
    infeasible_reason   TEXT,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);
"""

_TERMINAL_STATUSES = {"completed", "timeout", "cancelled", "interrupted", "infeasible"}


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 写入 ----

    def insert_job(
        self,
        job_id: str,
        status: str,
        seed: int,
        time_limit: float,
        request_json: str,
        created_at: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs(job_id, status, seed, time_limit_seconds, request_json,
                                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (job_id, status, seed, time_limit, request_json, created_at, created_at),
            )

    def update_running(self, job_id: str, lb_json: str) -> None:
        self._execute(
            "UPDATE jobs SET status='running', lb_json=?, updated_at=? WHERE job_id=?",
            (lb_json, time.time(), job_id),
        )

    def update_progress(
        self,
        job_id: str,
        status: str,
        iterations: int,
        elapsed: float,
        best_cost: float,
        best_vehicles: int,
        solution_json: str,
        stop_reason: Optional[str] = None,
        infeasible_reason: Optional[str] = None,
        lb_json: Optional[str] = None,
    ) -> None:
        sets = [
            "status=?",
            "iterations=?",
            "elapsed_seconds=?",
            "best_cost=?",
            "best_vehicles=?",
            "solution_json=?",
            "updated_at=?",
        ]
        params: list[Any] = [
            status, iterations, elapsed, best_cost, best_vehicles,
            solution_json, time.time(),
        ]
        if stop_reason is not None:
            sets.append("stop_reason=?")
            params.append(stop_reason)
        if infeasible_reason is not None:
            sets.append("infeasible_reason=?")
            params.append(infeasible_reason)
        if lb_json is not None:
            sets.append("lb_json=?")
            params.append(lb_json)
        params.append(job_id)
        self._execute(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id=?", params)

    def mark_interrupted(self, job_id: str) -> None:
        """重启恢复：把未完成作业标为 interrupted（不动已有方案）。"""
        self._execute(
            """UPDATE jobs SET status='interrupted', stop_reason='interrupted', updated_at=?
               WHERE job_id=? AND status NOT IN ('completed','timeout','cancelled',
                                                 'interrupted','infeasible')""",
            (time.time(), job_id),
        )

    def _execute(self, sql: str, params: list[Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(sql, params)

    # ---- 读取 ----

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
            return [dict(r) for r in rows]

    def unfinished_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE status NOT IN ('completed','timeout','cancelled',
                                        'interrupted','infeasible')"""
            ).fetchall()
            return [dict(r) for r in rows]


def dump_solution(solution_json_ready: dict[str, Any]) -> str:
    return json.dumps(solution_json_ready, ensure_ascii=False, separators=(",", ":"))
