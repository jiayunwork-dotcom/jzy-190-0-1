"""FastAPI 入口：调度系统只经 HTTP 调用。

接口：
  POST /jobs              提交排车作业，立即返回 job_id
  GET  /jobs/{job_id}     查询作业：状态、当前最好方案、下界、差距
  POST /jobs/{job_id}/cancel  请求取消
  GET  /healthz           存活检查
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import config
from .db import Database
from .jobs import JobManager, rebuild_solution
from .schemas import FieldError, JobRequest, validate_problem
from .serialize import build_problem
from .solver import relative_gap
from .verify import InfeasibleSolution, verify

_db: Optional[Database] = None
_manager: Optional[JobManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _manager
    _db = Database(config.DEFAULT_DB_PATH)
    _manager = JobManager(_db)
    _manager.recover_on_startup()
    yield


app = FastAPI(
    title="城配排车服务",
    version="1.0.0",
    description="异构车型、类别互斥、独占约束的 anytime 排车作业服务",
    lifespan=lifespan,
)


def _field_error_response(errors: list[FieldError]) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "detail": [{"field": e.field, "message": e.message} for e in errors],
        },
    )


class BusinessValidationError(Exception):
    def __init__(self, errors: list[FieldError]) -> None:
        self.errors = errors


@app.exception_handler(BusinessValidationError)
async def _on_business_validation_error(request: Request, exc: BusinessValidationError) -> JSONResponse:
    return _field_error_response(exc.errors)


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = []
    for err in exc.errors():
        loc = ".".join(
            str(p) if p not in ("body",) else "" for p in err["loc"] if p != "body"
        ).strip(".")
        detail.append(
            {
                "field": loc or "body",
                "message": err["msg"],
            }
        )
    return JSONResponse(
        status_code=422,
        content={"error": "validation_failed", "detail": detail},
    )


@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest):
    assert _manager is not None
    errors = validate_problem(req)
    if errors:
        raise BusinessValidationError(errors)
    job_id = _manager.submit(req)
    return {"job_id": job_id, "status": "pending"}


def _completed_flag(status: str) -> str:
    """completed 标记：completed=已证明最优；timeout=超时停止；其它见 stop_reason。"""
    return {
        "completed": "proven_optimal",
        "timeout": "time_limit",
        "cancelled": "cancelled",
        "interrupted": "interrupted",
        "infeasible": "infeasible",
        "running": "running",
        "pending": "pending",
    }.get(status, status)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    assert _db is not None
    row = _db.get_job(job_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error": "job_not_found", "job_id": job_id})

    req = JobRequest.model_validate_json(row["request_json"])
    problem = build_problem(req)

    lb = json.loads(row["lb_json"]) if row["lb_json"] else None
    solution = json.loads(row["solution_json"]) if row["solution_json"] not in (None, "null") else None
    cost = row["best_cost"]
    gap = relative_gap(cost, lb["cost"]) if (lb is not None and cost is not None) else None

    # 读回的终态方案再独立核验一遍（进程重启也覆盖），核验不过不返回脏方案。
    if solution is not None and row["status"] in {
        "completed", "timeout", "cancelled", "interrupted"
    }:
        rebuilt = rebuild_solution(problem, solution)
        try:
            verify(problem, rebuilt, expected_cost=cost)
        except InfeasibleSolution as exc:
            return JSONResponse(
                status_code=500,
                content={"error": "stored_solution_invalid", "detail": str(exc)},
            )

    return JSONResponse(
        content={
            "job_id": job_id,
            "status": row["status"],
            "stop_reason": _completed_flag(row["status"]),
            "iterations": row["iterations"],
            "elapsed_seconds": round(row["elapsed_seconds"], 6),
            "lower_bound": lb,
            "best_cost": _clean(cost),
            "vehicle_count": row["best_vehicles"],
            "relative_gap": None if gap is None else round(gap, 9),
            "gap_percent": None if gap is None else round(gap * 100.0, 6),
            "solution": solution,
            "infeasible_reason": row["infeasible_reason"],
        }
    )


@app.post("/jobs/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str) -> JSONResponse:
    assert _manager is not None
    if _manager.db.get_job(job_id) is None:
        return JSONResponse(status_code=404, content={"error": "job_not_found", "job_id": job_id})
    ok = _manager.cancel(job_id)
    return JSONResponse(
        content={"job_id": job_id, "cancel_requested": ok, "status": "cancelling" if ok else "terminal"}
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _clean(x: Optional[float]):
    if x is None:
        return None
    r = round(float(x), 6)
    return int(r) if float(r).is_integer() else r
