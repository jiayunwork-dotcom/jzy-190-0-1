"""作业生命周期管理：提交即返回作业号，后台线程 anytime 改进。

- 提交时同步计算下界（快），并立即落盘；
- 后台线程调用 solver.solve，通过回调把当前最好方案增量落盘；
- 取消只是设置事件，solver 只在 ILS 轮次边界检查，交出的始终是完整可行解；
- 重启时未到终态的作业统一标记为 interrupted（其最后一次落盘方案仍保留）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Optional

from . import config
from .bounds import compute_lower_bound
from .db import Database, dump_solution
from .models import Problem, Solution, Vehicle
from .schemas import JobRequest
from .serialize import build_problem, lower_bound_to_dict, solution_to_dict
from .solver import solve
from .verify import InfeasibleSolution, verify


class JobManager:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._cancel_events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ 启动恢复

    def recover_on_startup(self) -> None:
        for row in self.db.unfinished_jobs():
            self.db.mark_interrupted(row["job_id"])

    # ------------------------------------------------------------------ 提交

    def submit(self, req: JobRequest) -> str:
        job_id = uuid.uuid4().hex
        now = time.time()
        self.db.insert_job(
            job_id,
            status="pending",
            seed=req.seed,
            time_limit=req.time_limit_seconds,
            request_json=req.model_dump_json(),
            created_at=now,
        )
        thread = threading.Thread(
            target=self._run,
            args=(job_id, req.model_dump()),
            name=f"dispatch-{job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._cancel_events[job_id] = threading.Event()
            self._threads[job_id] = thread
        thread.start()
        return job_id

    # ------------------------------------------------------------------ 取消

    def cancel(self, job_id: str) -> bool:
        """请求取消。返回 True 表示作业存在且尚未到终态。"""
        row = self.db.get_job(job_id)
        if row is None:
            return False
        if row["status"] in {"completed", "timeout", "cancelled", "interrupted", "infeasible"}:
            return False
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is not None:
            event.set()
        return True

    def is_active(self, job_id: str) -> bool:
        row = self.db.get_job(job_id)
        return row is not None and row["status"] in {"pending", "running"}

    # ------------------------------------------------------------------ 执行

    def _run(self, job_id: str, req_dict: dict[str, Any]) -> None:
        req = JobRequest.model_validate(req_dict)
        problem = build_problem(req)
        start = time.monotonic()

        # 1) 下界（提交后很快可查）
        lb = compute_lower_bound(problem)
        lb_json = json.dumps(lower_bound_to_dict(lb), ensure_ascii=False)
        if not lb.feasible:
            self.db.update_progress(
                job_id,
                status="infeasible",
                iterations=0,
                elapsed=time.monotonic() - start,
                best_cost=0.0,
                best_vehicles=0,
                solution_json="null",
                infeasible_reason=lb.reason,
                lb_json=lb_json,
            )
            return
        self.db.update_running(job_id, lb_json)

        event = self._cancel_events[job_id]

        # 2) 求解，节流落盘
        state = {"last_flush": 0.0}

        def progress_cb(sol: Solution, cost: float, iterations: int) -> None:
            now = time.monotonic()
            if now - state["last_flush"] < config.PROGRESS_FLUSH_INTERVAL and iterations > 1:
                return
            state["last_flush"] = now
            self._flush_solution(job_id, problem, sol, "running", iterations,
                                 time.monotonic() - start, lb_json)

        result = solve(
            problem,
            seed=req.seed,
            time_limit=req.time_limit_seconds,
            progress_cb=progress_cb,
            is_cancelled=event.is_set,
        )

        # 3) 终态落盘（最终方案独立校验后才写）
        if result.solution is not None:
            try:
                verify(problem, result.solution, expected_cost=result.cost)
            except InfeasibleSolution:  # 防御性：不应发生
                self.db.update_progress(
                    job_id,
                    status="interrupted",
                    iterations=result.iterations,
                    elapsed=result.elapsed_seconds,
                    best_cost=0.0,
                    best_vehicles=0,
                    solution_json="null",
                    stop_reason="internal_verification_failed",
                    lb_json=lb_json,
                )
                return
            # 求解器状态 -> 作业状态：optimal 记为 completed（已证明最优）。
            job_status = {"optimal": "completed", "timeout": "timeout",
                          "cancelled": "cancelled"}.get(result.status, result.status)
            self._flush_solution(
                job_id, problem, result.solution,
                status=job_status,
                iterations=result.iterations,
                elapsed=result.elapsed_seconds,
                lb_json=lb_json,
                stop_reason=job_status,
            )
        else:
            self.db.update_progress(
                job_id,
                status="infeasible",
                iterations=result.iterations,
                elapsed=result.elapsed_seconds,
                best_cost=0.0,
                best_vehicles=0,
                solution_json="null",
                infeasible_reason=result.reason,
                lb_json=lb_json,
            )

    def _flush_solution(
        self,
        job_id: str,
        problem: Problem,
        sol: Solution,
        status: str,
        iterations: int,
        elapsed: float,
        lb_json: str,
        stop_reason: Optional[str] = None,
    ) -> None:
        payload = solution_to_dict(problem, sol)
        self.db.update_progress(
            job_id,
            status=status,
            iterations=iterations,
            elapsed=elapsed,
            best_cost=sol.cost(),
            best_vehicles=sol.vehicle_count(),
            solution_json=dump_solution(payload),
            stop_reason=stop_reason,
            lb_json=lb_json,
        )


# ---------------------------------------------------------------------- 读回重建

def rebuild_solution(problem: Problem, solution_payload: dict[str, Any]) -> Solution:
    """从落盘 JSON 重建 Solution，并由领域逻辑重新累计重量/体积/类别。"""
    vehicles: list[Vehicle] = []
    for vdata in solution_payload["vehicles"]:
        v = Vehicle(type_id=vdata["type_id"])
        for sid in vdata["shipment_ids"]:
            ship = problem.shipment(sid)
            v.shipment_ids.append(sid)
            v.weight += ship.weight
            v.volume += ship.volume
            if ship.category is not None:
                v._cat_counts[ship.category] = v._cat_counts.get(ship.category, 0) + 1
        vehicles.append(v)
    return Solution(vehicles=vehicles, problem=problem)
