"""SQLite 持久化：作业与最好方案。数据文件挂在卷上，重启后作业仍在。"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,              -- running | completed | cancelled | interrupted | failed
  stop_reason TEXT,                  -- optimal | exhausted | time_limit | cancelled | interrupted
  seed INTEGER NOT NULL,
  time_limit_seconds REAL NOT NULL,
  problem_json TEXT NOT NULL,
  lower_bound_json TEXT NOT NULL,
  best_solution_json TEXT,
  best_cost REAL,
  gap REAL,
  proven_optimal INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_job(
        self, job_id: str, seed: int, time_limit: float, problem: dict, lower_bound: dict
    ) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(job_id, status, seed, time_limit_seconds, problem_json,"
                " lower_bound_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    "running",
                    seed,
                    time_limit,
                    json.dumps(problem),
                    json.dumps(lower_bound),
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def update_best(self, job_id: str, snap: dict, gap: float | None, proven: bool) -> None:
        """刷新当前最好方案；只在作业仍在运行时生效（防重启后旧线程覆写）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET best_solution_json=?, best_cost=?, gap=?, proven_optimal=?,"
                " updated_at=? WHERE job_id=? AND status='running'",
                (json.dumps(snap), snap["total_cost"], gap, int(proven), _now(), job_id),
            )
            self._conn.commit()

    def finish(
        self,
        job_id: str,
        status: str,
        stop_reason: str,
        snap: dict | None,
        gap: float | None,
        proven: bool,
        error: str | None,
    ) -> None:
        with self._lock:
            if snap is not None:
                self._conn.execute(
                    "UPDATE jobs SET status=?, stop_reason=?, best_solution_json=?, best_cost=?,"
                    " gap=?, proven_optimal=?, error=?, updated_at=?"
                    " WHERE job_id=? AND status='running'",
                    (
                        status,
                        stop_reason,
                        json.dumps(snap),
                        snap["total_cost"],
                        gap,
                        int(proven),
                        error,
                        _now(),
                        job_id,
                    ),
                )
            else:
                self._conn.execute(
                    "UPDATE jobs SET status=?, stop_reason=?, error=?, updated_at=?"
                    " WHERE job_id=? AND status='running'",
                    (status, stop_reason, error, _now(), job_id),
                )
            self._conn.commit()

    def mark_interrupted(self) -> int:
        """启动 / 关停时调用：所有未跑完的作业标为中断。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='interrupted', stop_reason='interrupted', updated_at=?"
                " WHERE status IN ('running', 'pending')",
                (_now(),),
            )
            self._conn.commit()
            return cur.rowcount

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            return cur.fetchone()

    def list_jobs(self) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            return cur.fetchall()
