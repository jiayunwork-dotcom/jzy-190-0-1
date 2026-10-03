"""下界的正确性与要求的性质。"""
from app.solver import build_problem, compute_lower_bound


def lb_of(raw):
    p, _, _ = build_problem(raw)
    return compute_lower_bound(p)


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


def test_weight_bound_four_items():
    """6/5/5/4 吨、10 吨车型、容积不起作用：下界为 2 辆，最紧的是重量角度。"""
    lb = lb_of(four_items_problem())
    assert lb.vehicles == 2
    assert lb.cost == 200.0
    assert lb.tightest == "weight"


def test_volume_bound_can_be_tightest():
    """大车型单位体积费用更低时，体积角度给出最紧的费用界。"""
    raw = {
        "items": [{"id": f"i{k}", "weight": 1, "volume": 60} for k in range(3)],
        "vehicle_types": [
            {"name": "s", "max_weight": 100, "max_volume": 10, "cost": 30, "count": 10},
            {"name": "b", "max_weight": 100, "max_volume": 100, "cost": 100, "count": 10},
        ],
        "categories": [],
        "conflicts": [],
    }
    lb = lb_of(raw)
    assert lb.vehicles == 2  # ceil(180/100)
    assert lb.tightest == "volume"  # 180 * (100/100) = 180 > 2 * 30
    assert lb.cost == 180.0


def test_exclusive_bound():
    raw = {
        "items": [
            {"id": f"e{k}", "weight": 1, "volume": 1, "exclusive": True} for k in range(3)
        ]
        + [{"id": "n", "weight": 1, "volume": 1}],
        "vehicle_types": [
            {"name": "t", "max_weight": 100, "max_volume": 100, "cost": 30, "count": 10}
        ],
        "categories": [],
        "conflicts": [],
    }
    lb = lb_of(raw)
    assert lb.vehicles == 3
    assert lb.tightest == "exclusive"
    assert lb.cost == 90.0


def test_lower_bound_monotonic_when_adding_item():
    base = four_items_problem()
    lb1 = lb_of(base)
    for extra in [
        {"id": "x", "weight": 1, "volume": 0},
        {"id": "x", "weight": 7, "volume": 0},
        {"id": "x", "weight": 0.5, "volume": 0, "exclusive": True},
    ]:
        raw2 = dict(base)
        raw2["items"] = base["items"] + [extra]
        lb2 = lb_of(raw2)
        assert lb2.cost >= lb1.cost
        assert lb2.vehicles >= lb1.vehicles


def test_lower_bound_scale_invariant_vehicle_count():
    """全部重量和载重乘同一个正数，车辆数下界不变。"""
    base = four_items_problem()
    lb1 = lb_of(base)
    for factor in (2, 10, 0.5, 1000):
        raw = {
            "items": [dict(it, weight=it["weight"] * factor) for it in base["items"]],
            "vehicle_types": [
                dict(t, max_weight=t["max_weight"] * factor) for t in base["vehicle_types"]
            ],
            "categories": [],
            "conflicts": [],
        }
        lb2 = lb_of(raw)
        assert lb2.vehicles == lb1.vehicles


def test_empty_problem():
    lb = lb_of({"items": [], "vehicle_types": [], "categories": [], "conflicts": []})
    assert lb.vehicles == 0
    assert lb.cost == 0.0
