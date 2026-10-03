"""HTTP 接口：调度系统只经 HTTP 调用。"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .jobs import JobManager
from .schemas import SubmitRequest
from .store import SQLiteStore
from .validation import validate_problem

MAX_TIME_LIMIT = 3600.0


def _job_response(row) -> dict:
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "stop_reason": row["stop_reason"],
        "seed": row["seed"],
        "time_limit_seconds": row["time_limit_seconds"],
        "lower_bound": json.loads(row["lower_bound_json"]),
        "best_solution": (
            json.loads(row["best_solution_json"]) if row["best_solution_json"] else None
        ),
        "best_cost": row["best_cost"],
        "gap": row["gap"],
        "proven_optimal": bool(row["proven_optimal"]),
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_app(db_path: str | None = None) -> FastAPI:
    path = db_path or os.environ.get("APP_DB_PATH", "/data/app.db")
    store = SQLiteStore(path)
    manager = JobManager(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.recover()  # 重启前没跑完的作业标为中断
        yield
        manager.shutdown()
        store.close()

    app = FastAPI(title="排车服务", version="1.0.0", lifespan=lifespan)

    @app.get("/")
    def root():
        return {"service": "fleet-scheduling", "docs": "/docs"}

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/jobs", status_code=201)
    def submit_job(req: SubmitRequest):
        problem = req.problem.model_dump()
        errors = validate_problem(problem)
        tl = req.options.time_limit_seconds
        if not (0 < tl <= MAX_TIME_LIMIT):
            errors.append(
                {
                    "field": "options.time_limit_seconds",
                    "message": f"must be in (0, {MAX_TIME_LIMIT}]",
                }
            )
        if errors:
            return JSONResponse(status_code=422, content={"detail": errors})
        job_id = manager.submit(problem, req.options.seed, tl)
        return _job_response(manager.get(job_id))

    @app.get("/jobs")
    def list_jobs():
        return {"jobs": [_job_response(row) for row in manager.list()]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        row = manager.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_response(row)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        row = manager.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        manager.cancel(job_id)
        return _job_response(manager.get(job_id))

    return app


app = create_app()
