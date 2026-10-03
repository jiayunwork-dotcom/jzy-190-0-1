"""API 行为：作业生命周期、取消、超时、持久化与重启语义。"""
import random
import time

from fastapi.testclient import TestClient

from app.main import create_app
from app.solver import build_problem, verify_solution

from conftest import make_problem, submit, wait_done


def four_items_problem():
    return make_problem(
        items=[
            {"id": "a", "weight": 6, "volume": 0},
            {"id": "b", "weight": 5, "volume": 0},
            {"id": "c", "weight": 5, "volume": 0},
            {"id": "d", "weight": 4, "volume": 0},
        ],
        vehicle_types=[
            {"name": "t10", "max_weight": 10, "max_volume": 10**9, "cost": 100, "count": 10}
        ],
    )


def big_problem(n=300, seed=1):
    rng = random.Random(seed)
    return make_problem(
        items=[
            {"id": f"i{k}", "weight": rng.randint(1, 9), "volume": rng.randint(1, 9)}
            for k in range(n)
        ],
        vehicle_types=[
            {"name": "small", "max_weight": 20, "max_volume": 30, "cost": 60, "count": n},
            {"name": "large", "max_weight": 50, "max_volume": 80, "cost": 120, "count": n},
        ],
    )


def verify_via_api(problem, snap):
    p, _, _ = build_problem(problem)
    return verify_solution(p, snap)


def test_submit_poll_and_lower_bound(client):
    job = submit(client, four_items_problem(), seed=0, time_limit=10)
    assert job["status"] in ("running", "completed")  # 小实例可能提交即完成
    assert job["lower_bound"]["vehicles"] == 2
    assert job["lower_bound"]["cost"] == 200.0
    assert job["lower_bound"]["tightest"] == "weight"

    done = wait_done(client, job["job_id"])
    assert done["status"] == "completed"
    assert done["stop_reason"] == "optimal"
    assert done["proven_optimal"] is True
    best = done["best_solution"]
    assert best["vehicle_count"] == 2
    assert best["total_cost"] == 200.0
    assert done["gap"] == 0.0
    assert verify_via_api(four_items_problem(), best) == []


def test_gap_and_cost_never_below_bound(client):
    problem = big_problem(n=80, seed=3)
    job = submit(client, problem, seed=5, time_limit=10)
    done = wait_done(client, job["job_id"])
    assert done["status"] == "completed"
    assert done["best_cost"] >= done["lower_bound"]["cost"] - 1e-6
    assert done["gap"] is not None and done["gap"] >= 0
    assert verify_via_api(problem, done["best_solution"]) == []


def test_same_input_same_seed_same_result(client):
    problem = big_problem(n=60, seed=8)
    j1 = wait_done(client, submit(client, problem, seed=42, time_limit=20)["job_id"])
    j2 = wait_done(client, submit(client, problem, seed=42, time_limit=20)["job_id"])
    assert j1["best_solution"] == j2["best_solution"]
    assert j1["best_cost"] == j2["best_cost"]


def test_cancel_returns_complete_feasible_solution(client):
    problem = big_problem(n=1500, seed=11)
    job = submit(client, problem, seed=1, time_limit=600)
    # 等到出现当前最好方案再取消
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        cur = client.get(f"/jobs/{job['job_id']}").json()
        if cur["best_solution"] is not None:
            break
        time.sleep(0.05)
    r = client.post(f"/jobs/{job['job_id']}/cancel")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "cancelled"
    assert body["stop_reason"] == "cancelled"
    assert body["best_solution"] is not None
    assert verify_via_api(problem, body["best_solution"]) == []


def test_time_limit_stop(client):
    problem = big_problem(n=1500, seed=13)
    job = submit(client, problem, seed=2, time_limit=1)
    done = wait_done(client, job["job_id"], timeout=60)
    assert done["status"] == "completed"
    assert done["stop_reason"] == "time_limit"
    assert done["proven_optimal"] is False
    assert verify_via_api(problem, done["best_solution"]) == []


def test_unknown_job_404(client):
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/jobs/nope/cancel").status_code == 404


def test_persistence_and_restart(tmp_path):
    """重启后：已完成作业与方案保留；未完成的标为中断。"""
    db = str(tmp_path / "restart.db")
    small = four_items_problem()
    big = big_problem(n=1500, seed=17)

    app1 = create_app(db)
    with TestClient(app1) as c1:
        done = wait_done(c1, submit(c1, small, seed=0, time_limit=10)["job_id"])
        assert done["status"] == "completed"
        completed_id = done["job_id"]
        completed_solution = done["best_solution"]

        running = submit(c1, big, seed=1, time_limit=600)
        running_id = running["job_id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if c1.get(f"/jobs/{running_id}").json()["best_solution"] is not None:
                break
            time.sleep(0.05)
    # 离开 with：服务关停

    app2 = create_app(db)
    with TestClient(app2) as c2:
        kept = c2.get(f"/jobs/{completed_id}")
        assert kept.status_code == 200
        kept = kept.json()
        assert kept["status"] == "completed"
        assert kept["best_solution"] == completed_solution

        interrupted = c2.get(f"/jobs/{running_id}").json()
        assert interrupted["status"] == "interrupted"
        assert interrupted["stop_reason"] == "interrupted"
        # 中断前已改进出的可行方案仍然保留
        assert interrupted["best_solution"] is not None
        assert verify_via_api(big, interrupted["best_solution"]) == []

        jobs = c2.get("/jobs").json()["jobs"]
        assert {j["job_id"] for j in jobs} >= {completed_id, running_id}
