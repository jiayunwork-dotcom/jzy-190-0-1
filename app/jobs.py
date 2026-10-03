"""作业管理：提交即返回作业号，后台线程持续改进，随时可查可取消。"""
from __future__ import annotations

import threading
import uuid

from .solver import (
    build_problem,
    compute_lower_bound,
    snapshot_solution,
    solve,
    verify_solution,
)
from .store import SQLiteStore


def _gap(cost: float, lb_cost: float) -> float | None:
    if lb_cost <= 0:
        return 0.0 if cost <= 0 else None
    return max(0.0, (cost - lb_cost) / lb_cost)


def _tol(x: float) -> float:
    return 1e-9 * max(1.0, abs(x))


class JobManager:
    def __init__(self, store: SQLiteStore):
        self.store = store
        self._lock = threading.Lock()
        self._handles: dict[str, dict] = {}

    # 生命周期 ------------------------------------------------------

    def recover(self) -> None:
        """服务启动：重启前没跑完的作业标为中断。"""
        self.store.mark_interrupted()

    def shutdown(self) -> None:
        """服务关停：先把未完成的标为中断（等同重启语义），再通知线程退出。"""
        self.store.mark_interrupted()
        with self._lock:
            handles = list(self._handles.values())
        for h in handles:
            h["cancel"].set()
        for h in handles:
            h["thread"].join(timeout=5.0)

    # 作业操作 ------------------------------------------------------

    def submit(self, problem_raw: dict, seed: int, time_limit: float) -> str:
        problem, sw, sv = build_problem(problem_raw)
        lb = compute_lower_bound(problem)
        job_id = uuid.uuid4().hex
        self.store.insert_job(
            job_id,
            seed,
            time_limit,
            problem_raw,
            {
                "cost": lb.cost,
                "vehicles": lb.vehicles,
                "tightest": lb.tightest,
                "details": lb.details,
            },
        )
        cancel = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(job_id, problem, sw, sv, lb.cost, seed, time_limit, cancel),
            daemon=True,
            name=f"job-{job_id[:8]}",
        )
        with self._lock:
            self._handles[job_id] = {"thread": thread, "cancel": cancel}
        thread.start()
        return job_id

    def cancel(self, job_id: str) -> None:
        with self._lock:
            handle = self._handles.get(job_id)
        if handle is not None:
            handle["cancel"].set()
            handle["thread"].join(timeout=10.0)

    def get(self, job_id: str):
        return self.store.get_job(job_id)

    def list(self):
        return self.store.list_jobs()

    # 后台执行 ------------------------------------------------------

    def _run(self, job_id, problem, sw, sv, lb_cost, seed, time_limit, cancel) -> None:
        def on_improvement(sol) -> None:
            snap = snapshot_solution(problem, sol, sw, sv)
            # 落库前独立校验：绝不交出不可行的方案
            if verify_solution(problem, snap):
                return
            cost = snap["total_cost"]
            self.store.update_best(
                job_id, snap, _gap(cost, lb_cost), cost <= lb_cost + _tol(lb_cost)
            )

        try:
            result = solve(
                problem, lb_cost, seed, time_limit, cancel.is_set, on_improvement
            )
        except Exception as exc:  # 兜底：求解器异常不拖垮服务
            self.store.finish(job_id, "failed", "error", None, None, False, repr(exc))
            return
        finally:
            with self._lock:
                self._handles.pop(job_id, None)

        if result.error is not None or result.solution is None:
            self.store.finish(
                job_id, "failed", "infeasible", None, None, False,
                result.error or "no feasible solution found",
            )
            return
        snap = snapshot_solution(problem, result.solution, sw, sv)
        if verify_solution(problem, snap):
            self.store.finish(
                job_id, "failed", "error", None, None, False,
                "internal error: solver produced an infeasible solution",
            )
            return
        cost = snap["total_cost"]
        proven = result.stop_reason == "optimal" or cost <= lb_cost + _tol(lb_cost)
        status = "cancelled" if result.stop_reason == "cancelled" else "completed"
        self.store.finish(
            job_id, status, result.stop_reason, snap, _gap(cost, lb_cost), proven, None
        )
