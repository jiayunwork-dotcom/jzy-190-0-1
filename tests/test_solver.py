"""求解器性质：可行性、下界关系、确定性、缩放不变性。"""
import random

from app.solver import (
    build_problem,
    compute_lower_bound,
    snapshot_solution,
    solve,
    verify_solution,
)


def run_solver(raw, seed=1, time_limit=10.0):
    p, sw, sv = build_problem(raw)
    lb = compute_lower_bound(p)
    result = solve(p, lb.cost, seed, time_limit, lambda: False, lambda s: None)
    assert result.error is None
    snap = snapshot_solution(p, result.solution, sw, sv)
    return p, lb, result, snap


def assignment_of(snap):
    """item_id -> 车辆序号，用于比较两次求解的方案结构。"""
    out = {}
    for vi, v in enumerate(snap["vehicles"]):
        for iid in v["items"]:
            out[iid] = vi
    return out


def four_items_problem():
    return {
        "items": [
            {"id": "a", "weight": 6, "volume": 0},
            {"id": "b", "weight": 5, "volume": 0},
            {"id": "c", "weight": 5, "volume": 0},
            {"id": "d", "weight": 4, "volume": 0},
        ],
        "vehicle_types": [
            {"name": "t10", "max_weight": 10, "max_volume": 10**9, "cost": 100, "count": 10}
        ],
        "categories": [],
        "conflicts": [],
    }


def test_four_items_two_vehicles_optimal():
    """6/5/5/4 吨、10 吨车型：下界 2 辆，求解给出 2 辆且费用达到下界。"""
    _, lb, result, snap = run_solver(four_items_problem())
    assert lb.vehicles == 2
    assert snap["vehicle_count"] == 2
    assert snap["total_cost"] == 200.0
    assert result.stop_reason == "optimal"


def test_all_fit_one_vehicle():
    raw = {
        "items": [{"id": f"i{k}", "weight": 10, "volume": 1} for k in range(5)],
        "vehicle_types": [
            {"name": "big", "max_weight": 100, "max_volume": 100, "cost": 90, "count": 5}
        ],
        "categories": [],
        "conflicts": [],
    }
    _, _, _, snap = run_solver(raw)
    assert snap["vehicle_count"] == 1


def test_merge_into_cheaper_single_vehicle():
    """全部货装得进一辆大车且比两辆小车便宜时，应收敛到一辆车。"""
    raw = {
        "items": [{"id": f"i{k}", "weight": 10, "volume": 1} for k in range(10)],
        "vehicle_types": [
            {"name": "small", "max_weight": 10, "max_volume": 100, "cost": 50, "count": 20},
            {"name": "big", "max_weight": 100, "max_volume": 100, "cost": 90, "count": 20},
        ],
        "categories": [],
        "conflicts": [],
    }
    _, _, _, snap = run_solver(raw)
    assert snap["vehicle_count"] == 1
    assert snap["vehicles"][0]["vehicle_type"] == "big"
    assert snap["total_cost"] == 90.0


def test_conflicts_force_separate_vehicles():
    raw = {
        "items": [
            {"id": "food1", "weight": 1, "volume": 1, "category": "food"},
            {"id": "chem1", "weight": 1, "volume": 1, "category": "chemical"},
        ],
        "vehicle_types": [
            {"name": "t", "max_weight": 100, "max_volume": 100, "cost": 10, "count": 5}
        ],
        "categories": ["food", "chemical"],
        "conflicts": [["food", "chemical"]],
    }
    p, _, _, snap = run_solver(raw)
    assert snap["vehicle_count"] == 2
    assert verify_solution(p, snap) == []


def test_exclusive_item_rides_alone():
    raw = {
        "items": [
            {"id": "vip", "weight": 1, "volume": 1, "exclusive": True},
            {"id": "n1", "weight": 1, "volume": 1},
            {"id": "n2", "weight": 1, "volume": 1},
        ],
        "vehicle_types": [
            {"name": "t", "max_weight": 100, "max_volume": 100, "cost": 10, "count": 5}
        ],
        "categories": [],
        "conflicts": [],
    }
    p, _, _, snap = run_solver(raw)
    assert snap["vehicle_count"] == 2
    assert verify_solution(p, snap) == []
    for v in snap["vehicles"]:
        if "vip" in v["items"]:
            assert v["items"] == ["vip"]


def test_vehicle_count_limit_respected():
    raw = {
        "items": [{"id": f"i{k}", "weight": 10, "volume": 1} for k in range(2)],
        "vehicle_types": [
            {"name": "small", "max_weight": 10, "max_volume": 10, "cost": 5, "count": 1},
            {"name": "big", "max_weight": 100, "max_volume": 100, "cost": 9, "count": 5},
        ],
        "categories": [],
        "conflicts": [],
    }
    p, _, _, snap = run_solver(raw)
    assert verify_solution(p, snap) == []
    # 最优：一辆大车装全部（9），而不是小车+大车（14）
    assert snap["total_cost"] == 9.0


def test_deterministic_same_seed():
    raw = _random_problem(random.Random(99), n=60, n_types=3, conflicts=True, exclusive=True)
    _, _, _, snap1 = run_solver(raw, seed=7)
    _, _, _, snap2 = run_solver(raw, seed=7)
    assert snap1 == snap2


def test_weight_scaling_keeps_vehicle_count_and_assignment():
    """全部重量和载重乘同一个正数，车辆数与装载结构不变。"""
    rng = random.Random(5)
    raw = _random_problem(rng, n=40, n_types=2)
    _, _, _, snap1 = run_solver(raw, seed=3)
    for factor in (10, 0.5):
        scaled = {
            "items": [dict(it, weight=it["weight"] * factor) for it in raw["items"]],
            "vehicle_types": [
                dict(t, max_weight=t["max_weight"] * factor) for t in raw["vehicle_types"]
            ],
            "categories": raw["categories"],
            "conflicts": raw["conflicts"],
        }
        _, _, _, snap2 = run_solver(scaled, seed=3)
        assert snap2["vehicle_count"] == snap1["vehicle_count"]
        assert assignment_of(snap2) == assignment_of(snap1)


def _random_problem(rng, n, n_types=2, conflicts=False, exclusive=False):
    types = []
    for t in range(n_types):
        types.append(
            {
                "name": f"T{t}",
                "max_weight": rng.choice([10.0, 20.0, 40.0]),
                "max_volume": rng.choice([20.0, 50.0, 100.0]),
                "cost": rng.choice([50.0, 80.0, 120.0, 200.0]),
                "count": n,  # 数量给足，保证可行
            }
        )
    items = []
    for i in range(n):
        items.append(
            {
                "id": f"I{i}",
                "weight": rng.randint(1, 8),
                "volume": rng.randint(1, 10),
                "category": rng.choice(["a", "b"]) if conflicts else None,
                "exclusive": exclusive and rng.random() < 0.1,
            }
        )
    return {
        "items": items,
        "vehicle_types": types,
        "categories": ["a", "b"] if conflicts else [],
        "conflicts": [["a", "b"]] if conflicts else [],
    }


def test_fuzz_feasible_and_above_lower_bound():
    """随机实例：方案必须完整可行，费用不低于下界，车辆数不低于下界。"""
    rng = random.Random(2024)
    for case in range(12):
        raw = _random_problem(
            rng,
            n=rng.randint(5, 35),
            n_types=rng.randint(1, 3),
            conflicts=case % 2 == 0,
            exclusive=case % 3 == 0,
        )
        p, lb, result, snap = run_solver(raw, seed=case, time_limit=2.0)
        errs = verify_solution(p, snap)
        assert errs == [], f"case {case}: {errs}"
        assert snap["total_cost"] >= lb.cost - 1e-6
        assert snap["vehicle_count"] >= lb.vehicles
