"""求解器端到端性质测试：可行性、最优/差距、确定性、缩放、取消完整性。"""
from __future__ import annotations

import pytest

from app.bounds import compute_lower_bound
from app.solver import solve
from app.verify import InfeasibleSolution, verify
from conftest import make_problem, one_truck_type


def run(payload, seed=1, tl=2.0, cancel=None):
    p = make_problem(payload)
    return p, solve(p, seed=seed, time_limit=tl, is_cancelled=cancel)


def test_6554_two_trucks_and_proven():
    p, res = run(
        {
            "shipments": [{"id": f"s{i}", "weight": w} for i, w in enumerate([6, 5, 5, 4])],
            "vehicle_types": [one_truck_type(cap_weight=10, available=10)],
        }
    )
    assert res.status == "optimal"
    assert res.vehicle_count == 2
    assert res.cost == pytest.approx(200.0)
    verify(p, res.solution, expected_cost=res.cost)
    assert res.relative_gap == pytest.approx(0.0, abs=1e-9)


def test_all_fit_single_truck():
    p, res = run(
        {
            "shipments": [{"id": "a", "weight": 3}, {"id": "b", "weight": 2}],
            "vehicle_types": [one_truck_type(cap_weight=10, available=5)],
        }
    )
    assert res.vehicle_count == 1
    assert res.status == "optimal"
    verify(p, res.solution)


def test_empty_shipments_zero_vehicles():
    p, res = run({"shipments": [], "vehicle_types": [one_truck_type()]})
    assert res.status == "optimal"
    assert res.vehicle_count == 0
    assert res.cost == 0
    verify(p, res.solution)


def test_exclusive_shipment_is_alone():
    p, res = run(
        {
            "shipments": [
                {"id": "a", "weight": 3, "exclusive": True},
                {"id": "b", "weight": 2},
                {"id": "c", "weight": 1},
            ],
            "vehicle_types": [one_truck_type(cap_weight=10, available=5)],
        },
        tl=0.5,
    )
    verify(p, res.solution)
    # a 所在车只能有 a 一票
    for v in res.solution.vehicles:
        if "a" in v.shipment_ids:
            assert v.shipment_ids == ["a"]


def test_conflicting_categories_never_share_vehicle():
    p, res = run(
        {
            "shipments": [
                {"id": "a", "weight": 4, "category": "food"},
                {"id": "b", "weight": 4, "category": "chem"},
                {"id": "c", "weight": 4, "category": "food"},
                {"id": "d", "weight": 4, "category": "chem"},
            ],
            "vehicle_types": [one_truck_type(cap_weight=5, available=10)],
            "conflicts": [["food", "chem"]],
        },
        tl=1.0,
    )
    verify(p, res.solution)
    assert res.vehicle_count == 4
    # 4 车是该整票装箱的最优车辆数；分数下界为 3.2 辆，故状态可能是 timeout。
    assert res.cost == pytest.approx(400.0)


def test_conflict_proven_optimal_when_integer_gap_closes():
    """每票体积强制单车时，冲突下界=4 车=方案，可证明最优。"""
    p, res = run(
        {
            "shipments": [
                {"id": "a", "weight": 4, "volume": 6, "category": "food"},
                {"id": "b", "weight": 4, "volume": 6, "category": "chem"},
                {"id": "c", "weight": 4, "volume": 6, "category": "food"},
                {"id": "d", "weight": 4, "volume": 6, "category": "chem"},
            ],
            "vehicle_types": [one_truck_type(cap_weight=5, cap_volume=6, available=10)],
            "conflicts": [["food", "chem"]],
        },
        tl=1.0,
    )
    verify(p, res.solution)
    assert res.vehicle_count == 4
    assert res.relative_gap == pytest.approx(0.0, abs=1e-9)


def test_hetero_fleet_picks_two_big_trucks():
    p, res = run(
        {
            "shipments": [{"id": f"x{i}", "weight": 6} for i in range(6)],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 6},
                {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 150, "available": 6},
            ],
        },
        tl=1.0,
    )
    verify(p, res.solution)
    assert res.vehicle_count == 2
    assert res.cost == pytest.approx(300.0)
    assert {v.type_id for v in res.solution.vehicles} == {"L"}


def test_vehicle_type_availability_respected():
    p, res = run(
        {
            "shipments": [{"id": f"x{i}", "weight": 6} for i in range(6)],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 6},
                {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 150, "available": 1},
            ],
        },
        tl=1.0,
    )
    verify(p, res.solution)
    counts = res.solution.type_counts()
    assert counts["L"] <= 1
    assert counts["S"] <= 6


def test_cost_never_below_lower_bound():
    p, res = run(
        {
            "shipments": [{"id": f"x{i}", "weight": w} for i, w in enumerate([7, 7, 7, 3, 3])],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 10},
                {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 180, "available": 10},
            ],
        },
        tl=0.8,
    )
    verify(p, res.solution)
    assert res.cost >= res.lower_bound.cost - 1e-9


def test_determinism_same_seed_same_result():
    payload = {
        "shipments": [
            {"id": f"x{i}", "weight": w}
            for i, w in enumerate([6, 9, 4, 7, 3, 8, 2, 5, 6, 4, 7, 2])
        ],
        "vehicle_types": [
            {"id": "S", "max_weight": 12, "max_volume": 1e9, "cost": 100, "available": 20},
            {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 160, "available": 10},
        ],
    }
    _, r1 = run(payload, seed=123, tl=0.6)
    _, r2 = run(payload, seed=123, tl=0.6)
    assert r1.cost == r2.cost
    assert _signature(r1) == _signature(r2)


def test_different_seed_can_differ_but_both_feasible():
    payload = {
        "shipments": [{"id": f"x{i}", "weight": w} for i, w in enumerate([6, 9, 4, 7, 3, 8, 2, 5])],
        "vehicle_types": [
            {"id": "S", "max_weight": 12, "max_volume": 1e9, "cost": 100, "available": 20},
            {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 160, "available": 10},
        ],
    }
    p1, r1 = run(payload, seed=1, tl=0.4)
    p2, r2 = run(payload, seed=2, tl=0.4)
    verify(p1, r1.solution)
    verify(p2, r2.solution)


def test_scaling_weights_and_capacities_keeps_vehicle_count():
    k = 3.7
    weights = [6, 5, 5, 4]
    _, r1 = run(
        {
            "shipments": [{"id": f"s{i}", "weight": w} for i, w in enumerate(weights)],
            "vehicle_types": [one_truck_type(cap_weight=10, cost=100, available=10)],
        },
        seed=5,
        tl=0.5,
    )
    _, r2 = run(
        {
            "shipments": [{"id": f"s{i}", "weight": w * k} for i, w in enumerate(weights)],
            "vehicle_types": [one_truck_type(cap_weight=10 * k, cost=100, available=10)],
        },
        seed=5,
        tl=0.5,
    )
    assert r1.vehicle_count == r2.vehicle_count == 2


def test_cancel_returns_complete_feasible_solution():
    calls = {"n": 0}

    def cancelled():
        calls["n"] += 1
        return calls["n"] > 1  # 第一轮后即取消

    p, res = run(
        {
            "shipments": [{"id": f"x{i}", "weight": w} for i, w in enumerate([6, 9, 4, 7, 3, 8, 2, 5] * 2)],
            "vehicle_types": [one_truck_type(cap_weight=12, available=40)],
        },
        tl=30.0,
        cancel=cancelled,
    )
    assert res.status == "cancelled"
    assert res.solution is not None
    verify(p, res.solution)


def test_timeout_returns_best_feasible_solution():
    p, res = run(
        {
            "shipments": [{"id": f"x{i}", "weight": (i % 7) + 1} for i in range(60)],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 80},
                {"id": "L", "max_weight": 18, "max_volume": 1e9, "cost": 160, "available": 40},
            ],
        },
        tl=0.3,
    )
    assert res.status in {"timeout", "optimal"}
    verify(p, res.solution)


def test_volume_constraint_enforced():
    p, res = run(
        {
            "shipments": [{"id": f"p{i}", "weight": 0.5, "volume": 6} for i in range(4)],
            "vehicle_types": [one_truck_type(cap_weight=10, cap_volume=10, available=10)],
        },
        tl=0.5,
    )
    verify(p, res.solution)
    # 每车最多 1 票（6+6>10 方），但分数下界 3 辆与整票 4 辆不同，方案应为 4 辆
    assert res.vehicle_count == 4


def _signature(res):
    return sorted(
        (v.type_id, tuple(sorted(v.shipment_ids))) for v in res.solution.vehicles
    )
