"""随机压力：大量随机实例上方案必须始终满足全部硬约束。"""
from __future__ import annotations

import random

import pytest

from app.bounds import compute_lower_bound
from app.solver import solve
from app.verify import verify
from conftest import make_problem


@pytest.mark.parametrize("trial", range(20))
def test_random_instances_always_feasible(trial):
    rng = random.Random(1000 + trial)
    n = rng.randrange(1, 80)
    cats = [None, None, None, "A", "B", "C"]
    ships = []
    for i in range(n):
        d = {
            "id": f"s{i}",
            "weight": round(rng.uniform(0.0, 9.0), 3),
            "volume": round(rng.uniform(0.0, 14.0), 3),
            "category": rng.choice(cats),
        }
        if rng.random() < 0.08:
            d["exclusive"] = True
        ships.append(d)
    k = rng.randrange(1, 4)
    types = []
    used_names = set()
    for j in range(k):
        cap_w = rng.choice([5, 8, 10, 15, 20])
        cap_v = rng.choice([8, 12, 20, 30, 40])
        cost = int(cap_w * 10 + cap_v * 2 + rng.randrange(0, 30))
        name = f"T{j}"
        used_names.add(name)
        types.append(
            {"id": name, "max_weight": cap_w, "max_volume": cap_v,
             "cost": cost, "available": rng.randrange(1, n + 5)}
        )
    # 保证有至少一种大车型，避免大量不可行；把一个车型放大。
    big = dict(types[0])
    big.update({"id": "BIG", "max_weight": 25, "max_volume": 45,
                "cost": 400, "available": n + 5})
    types.append(big)
    conflicts = []
    if rng.random() < 0.6:
        conflicts.append(["A", "B"])
    if rng.random() < 0.3:
        conflicts.append(["C", "C"])

    payload = {
        "shipments": ships,
        "vehicle_types": types,
        "conflicts": conflicts,
        "seed": trial * 7 + 3,
        "time_limit_seconds": 0.4,
    }
    problem = make_problem(payload)
    lb = compute_lower_bound(problem)
    # 业务层已保证单件可装与独占数<=总可用；容量总量不足时求解可能失败或仍可行，
    # 只要返回方案就必须可行。
    result = solve(problem, seed=payload["seed"], time_limit=0.4)
    if result.solution is not None:
        verify(problem, result.solution, expected_cost=result.cost)
        assert result.cost >= lb.cost - 1e-6


@pytest.mark.parametrize("trial", range(5))
def test_random_determinism(trial):
    rng = random.Random(2000 + trial)
    n = rng.randrange(10, 60)
    ships = [
        {"id": f"s{i}", "weight": round(rng.uniform(0.5, 9), 2),
         "volume": round(rng.uniform(0.5, 14), 2),
         "category": rng.choice([None, "A", "B"])}
        for i in range(n)
    ]
    payload = {
        "shipments": ships,
        "vehicle_types": [
            {"id": "S", "max_weight": 10, "max_volume": 16, "cost": 100, "available": n},
            {"id": "L", "max_weight": 20, "max_volume": 30, "cost": 170, "available": n},
        ],
        "conflicts": [["A", "B"]],
        "seed": 99,
        "time_limit_seconds": 0.5,
    }
    problem = make_problem(payload)

    def sig():
        r = solve(problem, seed=99, time_limit=0.5)
        s = tuple(
            sorted((v.type_id, tuple(sorted(v.shipment_ids))) for v in r.solution.vehicles)
        )
        return r.cost, s

    a = sig()
    b = sig()
    assert a == b
