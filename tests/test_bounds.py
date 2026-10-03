"""下界测试：需求中的判定性质全部覆盖。"""
from __future__ import annotations

import pytest

from app.bounds import compute_lower_bound
from conftest import make_problem, one_truck_type  # noqa: F401  (conftest 注入路径)


def problem(weights, **kw):
    return make_problem(
        {
            "shipments": [
                {"id": f"s{i}", "weight": w, "volume": kw.get("volume", 0.0)}
                for i, w in enumerate(weights)
            ],
            "vehicle_types": [
                one_truck_type(
                    cap_weight=kw.get("cap_w", 10),
                    cap_volume=kw.get("cap_v", 1e9),
                    cost=kw.get("cost", 100),
                    available=kw.get("available", 20),
                )
            ],
        }
    )


def test_lb_6554_is_two_trucks():
    """需求情形一：6/5/5/4 吨、10 吨车型，下界 2 辆。"""
    lb = compute_lower_bound(problem([6, 5, 5, 4]))
    assert lb.vehicles == 2
    assert lb.cost == pytest.approx(200.0)
    assert lb.binding == "weight"


def test_lb_all_fit_one_truck():
    lb = compute_lower_bound(problem([3, 2]))
    assert lb.vehicles == 1


def test_lb_adding_shipment_never_decreases():
    """多加一票货，下界（费用与车辆数）都不会降低。"""
    base = problem([6, 5, 5, 4])
    lb1 = compute_lower_bound(base)
    bigger = problem([6, 5, 5, 4, 1])
    lb2 = compute_lower_bound(bigger)
    assert lb2.cost >= lb1.cost - 1e-9
    assert lb2.vehicles >= lb1.vehicles


def test_lb_scaling_invariance():
    """重量与载重同乘正数，车辆数下界不变（分数费用下界等比缩放）。"""
    k = 2.5
    p1 = problem([6, 5, 5, 4], cap_w=10)
    p2 = problem([6 * k, 5 * k, 5 * k, 4 * k], cap_w=10 * k)
    lb1, lb2 = compute_lower_bound(p1), compute_lower_bound(p2)
    assert lb2.vehicles == lb1.vehicles == 2
    assert lb2.cost == pytest.approx(lb1.cost)  # 费用本身不变（车型费用没缩放）


def test_lb_volume_dimension():
    p = make_problem(
        {
            "shipments": [{"id": f"p{i}", "weight": 0.1, "volume": 6} for i in range(5)],
            "vehicle_types": [one_truck_type(cap_weight=10, cap_volume=10, available=10)],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.vehicles == 3
    assert lb.binding == "volume"
    assert lb.cost == pytest.approx(300.0)


def test_lb_exclusive_count_and_cost():
    p = make_problem(
        {
            "shipments": [
                {"id": "a", "weight": 3, "exclusive": True},
                {"id": "b", "weight": 2},
                {"id": "c", "weight": 1},
            ],
            "vehicle_types": [one_truck_type(cap_weight=10, available=5)],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.vehicles == 1  # 独占货至少 1 辆
    assert lb.dimensions["exclusive"].cost == pytest.approx(100.0)
    assert lb.exclusive_assignment.assignment == {"a": "T"}


def test_lb_conflict_pair_vehicles_and_cost():
    """食品/化工各两票 4 吨，车 5 吨。

    合法下界是分数装箱：每侧 8 吨 / 5 吨 = 1.6 辆，两侧相加 3.2 辆，
    分数下界不会进位（放松了整票约束）。整票 4 车是可行方案，
    与下界之间存在装箱问题固有的整数间隙。
    """
    p = make_problem(
        {
            "shipments": [
                {"id": "a", "weight": 4, "category": "food"},
                {"id": "b", "weight": 4, "category": "chem"},
                {"id": "c", "weight": 4, "category": "food"},
                {"id": "d", "weight": 4, "category": "chem"},
            ],
            "vehicle_types": [one_truck_type(cap_weight=5, available=10)],
            "conflicts": [["food", "chem"]],
        }
    )
    lb = compute_lower_bound(p)
    # 每侧分数下界 160，合计 320；分数辆数不进位。
    assert lb.cost == pytest.approx(320.0)
    assert lb.dimensions["conflict"].cost == pytest.approx(320.0)


def test_lb_conflict_pair_integer_when_items_fill_trucks():
    """当每票都必须单车且无法与同类拼车时，冲突下界能给出整数车辆数。

    每票 4 吨但车限 5 吨且总体积约束迫使每票一车（体积 6 方、车 6 方），
    此时分数辆数 = 票数，下界 = 4 辆。
    """
    p = make_problem(
        {
            "shipments": [
                {"id": "a", "weight": 4, "volume": 6, "category": "food"},
                {"id": "b", "weight": 4, "volume": 6, "category": "chem"},
                {"id": "c", "weight": 4, "volume": 6, "category": "food"},
                {"id": "d", "weight": 4, "volume": 6, "category": "chem"},
            ],
            "vehicle_types": [one_truck_type(cap_weight=5, cap_volume=6, available=10)],
            "conflicts": [["food", "chem"]],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.vehicles == 4
    assert lb.cost == pytest.approx(400.0)


def test_lb_self_conflict():
    p = make_problem(
        {
            "shipments": [
                {"id": "a", "weight": 1, "category": "X"},
                {"id": "b", "weight": 1, "category": "X"},
                {"id": "c", "weight": 1, "category": "X"},
            ],
            "vehicle_types": [one_truck_type(cap_weight=10, available=10)],
            "conflicts": [["X", "X"]],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.vehicles == 3


def test_lb_hetero_fractional_cost():
    """6x6 吨；小车 10/100 限 6 辆、大车 20/150 限 6 辆。

    分数装载：6 辆小车先装 36? 不——单位费用小车 10/吨、大车 7.5/吨，
    受可用量限制：6 辆大车 120 吨容量足够装 36 吨 -> 36*7.5 = 270。
    """
    p = make_problem(
        {
            "shipments": [{"id": f"x{i}", "weight": 6} for i in range(6)],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 6},
                {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 150, "available": 6},
            ],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.cost == pytest.approx(270.0)
    assert lb.vehicles == 2


def test_lb_total_capacity_shortage_is_infeasible():
    p = make_problem(
        {
            "shipments": [{"id": "a", "weight": 9}, {"id": "b", "weight": 9}],
            "vehicle_types": [one_truck_type(cap_weight=10, available=1)],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.feasible is False
    assert lb.reason == "total_weight_capacity"


def test_lb_exclusive_assignment_uses_cheapest_fit():
    p = make_problem(
        {
            "shipments": [
                {"id": "big", "weight": 18, "exclusive": True},
                {"id": "small", "weight": 2, "exclusive": True},
            ],
            "vehicle_types": [
                {"id": "S", "max_weight": 10, "max_volume": 1e9, "cost": 100, "available": 5},
                {"id": "L", "max_weight": 20, "max_volume": 1e9, "cost": 150, "available": 5},
            ],
        }
    )
    lb = compute_lower_bound(p)
    assert lb.feasible
    assert lb.cost == pytest.approx(250.0)
    assert lb.exclusive_assignment.assignment == {"big": "L", "small": "S"}
