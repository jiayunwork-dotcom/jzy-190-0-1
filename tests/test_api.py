"""HTTP 作业接口测试：提交即返回、随时查进度/下界/差距、取消、重启恢复、同种子一致。"""
from __future__ import annotations

import importlib
import time

import pytest
from fastapi.testclient import TestClient

import app.config as config
import app.main as main_mod
from app.db import Database
from app.jobs import JobManager


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DISPATCH_DB_PATH", db_path)
    importlib.reload(config)
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c, db_path


def _wait_terminal(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}").json()
        last = r
        if r["status"] in {"completed", "timeout", "cancelled", "interrupted", "infeasible"}:
            return r
        time.sleep(0.02)
    raise AssertionError(f"作业未在 {timeout}s 内结束：{last and last['status']}")


SIMPLE = {
    "shipments": [{"id": f"s{i}", "weight": w} for i, w in enumerate([6, 5, 5, 4])],
    "vehicle_types": [
        {"id": "T", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 10}
    ],
    "seed": 1,
    "time_limit_seconds": 2.0,
}


def test_submit_returns_job_id_immediately(client):
    c, _ = client
    r = c.post("/jobs", json=SIMPLE)
    assert r.status_code == 202
    body = r.json()
    assert len(body["job_id"]) == 32
    assert body["status"] == "pending"


def test_completed_job_reports_solution_lb_and_zero_gap(client):
    c, _ = client
    job_id = c.post("/jobs", json=SIMPLE).json()["job_id"]
    out = _wait_terminal(c, job_id)
    assert out["status"] == "completed"
    assert out["stop_reason"] == "proven_optimal"
    assert out["vehicle_count"] == 2
    assert out["best_cost"] == 200
    assert out["relative_gap"] == pytest.approx(0.0, abs=1e-9)
    assert out["lower_bound"]["vehicles"] == 2
    assert out["lower_bound"]["binding"] == "weight"
    # 方案完整
    assigned = [sid for v in out["solution"]["vehicles"] for sid in v["shipment_ids"]]
    assert sorted(assigned) == ["s0", "s1", "s2", "s3"]


def test_job_query_404(client):
    c, _ = client
    assert c.get("/jobs/nope").status_code == 404


def test_cancel_returns_complete_feasible_solution(client):
    c, _ = client
    payload = {
        "shipments": [
            {"id": f"x{i}", "weight": 0.1, "volume": 8} for i in range(80)
        ],
        "vehicle_types": [
            {"id": "S", "max_weight": 100, "max_volume": 10, "cost": 100, "available": 100}
        ],
        "seed": 3,
        "time_limit_seconds": 60.0,
    }
    job_id = c.post("/jobs", json=payload).json()["job_id"]
    # 等出现第一版方案后取消
    got_solution = False
    for _ in range(200):
        cur = c.get(f"/jobs/{job_id}").json()
        if cur.get("solution"):
            got_solution = True
            break
        time.sleep(0.01)
    assert got_solution
    r = c.post(f"/jobs/{job_id}/cancel")
    assert r.status_code == 200
    out = _wait_terminal(c, job_id, timeout=10.0)
    assert out["status"] == "cancelled"
    assert out["stop_reason"] == "cancelled"
    assigned = [sid for v in out["solution"]["vehicles"] for sid in v["shipment_ids"]]
    assert sorted(assigned) == sorted(s["id"] for s in payload["shipments"])


def test_timeout_status(client):
    c, _ = client
    payload = {
        "shipments": [{"id": f"x{i}", "weight": (i % 7) + 1} for i in range(120)],
        "vehicle_types": [
            {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 200}
        ],
        "seed": 5,
        "time_limit_seconds": 0.2,
    }
    job_id = c.post("/jobs", json=payload).json()["job_id"]
    out = _wait_terminal(c, job_id)
    assert out["status"] == "timeout"
    assert out["stop_reason"] == "time_limit"
    assert out["best_cost"] >= out["lower_bound"]["cost"] - 1e-9


def test_infeasible_job_reported(client):
    """通过业务校验但执行时证明不可行：两票独占货都只能用 L，而 L 仅 1 辆。"""
    c, _ = client
    payload = {
        "shipments": [
            {"id": "big", "weight": 18, "exclusive": True},
            {"id": "small", "weight": 2, "exclusive": True},
        ],
        "vehicle_types": [
            {"id": "S", "max_weight": 0.1, "max_volume": 1e9, "cost": 10, "available": 5},
            {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 100, "available": 1},
        ],
        "time_limit_seconds": 1.0,
    }
    job_id = c.post("/jobs", json=payload).json()["job_id"]
    out = _wait_terminal(c, job_id)
    assert out["status"] == "infeasible"
    assert out["infeasible_reason"] == "exclusive_capacity"


def test_validation_error_has_field_paths(client):
    c, _ = client
    r = c.post(
        "/jobs",
        json={
            "shipments": [{"id": "a", "weight": -5}],
            "vehicle_types": [
                {"id": "T", "max_weight": 10, "max_volume": 1, "cost": 100, "available": 5}
            ],
        },
    )
    assert r.status_code == 422
    fields = [d["field"] for d in r.json()["detail"]]
    assert any("weight" in f for f in fields)


def test_shipment_too_large_validation(client):
    c, _ = client
    r = c.post(
        "/jobs",
        json={
            "shipments": [{"id": "big", "weight": 99}],
            "vehicle_types": [
                {"id": "T", "max_weight": 10, "max_volume": 1, "cost": 100, "available": 5}
            ],
        },
    )
    assert r.status_code == 422
    assert any("比任何可用车型都大" in d["message"] for d in r.json()["detail"])


def test_same_seed_same_result(client):
    c, _ = client
    payload = {
        "shipments": [{"id": f"x{i}", "weight": (i % 8) + 1} for i in range(40)],
        "vehicle_types": [
            {"id": "S", "max_weight": 12, "max_volume": 1e9, "cost": 100, "available": 60},
            {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 160, "available": 20},
        ],
        "seed": 42,
        "time_limit_seconds": 0.5,
    }
    j1 = c.post("/jobs", json=payload).json()["job_id"]
    j2 = c.post("/jobs", json=payload).json()["job_id"]
    o1 = _wait_terminal(c, j1)
    o2 = _wait_terminal(c, j2)
    sig = lambda o: sorted(
        (v["type_id"], tuple(sorted(v["shipment_ids"]))) for v in o["solution"]["vehicles"]
    )
    assert o1["best_cost"] == o2["best_cost"]
    assert sig(o1) == sig(o2)


def test_restart_marks_unfinished_as_interrupted_and_keeps_done(client, tmp_path):
    c, db_path = client
    done_id = c.post("/jobs", json=SIMPLE).json()["job_id"]
    _wait_terminal(c, done_id)

    # 直接往库里塞一条 running 作业，模拟崩溃前没跑完。
    import app.main as m
    db = m._db
    orphan_id = "orphanjob000000000000000000000001"
    db.insert_job(orphan_id, "running", 1, 5.0,
                  '{"shipments":[],"vehicle_types":[{"id":"T","max_weight":10,'
                  '"max_volume":100,"cost":100,"available":1}],"conflicts":[],"seed":1,'
                  '"time_limit_seconds":5.0}', time.time())

    # 用同一数据文件启动新的管理器（模拟重启）。
    fresh_db = Database(db_path)
    mgr = JobManager(fresh_db)
    mgr.recover_on_startup()
    row = fresh_db.get_job(orphan_id)
    assert row["status"] == "interrupted"
    # 已完成的不受影响，方案仍在
    done = fresh_db.get_job(done_id)
    assert done["status"] == "completed"
    assert done["solution_json"] is not None


def test_healthz(client):
    c, _ = client
    assert c.get("/healthz").json() == {"status": "ok"}
