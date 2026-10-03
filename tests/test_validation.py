"""输入校验：每种非法输入都要指出具体字段。"""
import json
import math

from conftest import make_problem

VALID_TYPES = [{"name": "t", "max_weight": 10, "max_volume": 10, "cost": 100, "count": 5}]
VALID_ITEMS = [{"id": "a", "weight": 1, "volume": 1}]


def submit_raw(client, problem):
    r = client.post(
        "/jobs",
        json={"problem": problem, "options": {"seed": 0, "time_limit_seconds": 5}},
    )
    assert r.status_code == 422, r.json()
    return r.json()["detail"]


def submit_raw_allow_nan(client, problem):
    """httpx 的 json= 拒绝 NaN/Inf，改用原始 JSON 文本发送（服务端仍应识别并报字段）。"""
    body = json.dumps(
        {"problem": problem, "options": {"seed": 0, "time_limit_seconds": 5}},
        allow_nan=True,
    )
    r = client.post("/jobs", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    return r.json()["detail"]


def fields_of(detail):
    return {e["field"] for e in detail}


def test_negative_weight(client):
    p = make_problem([{"id": "a", "weight": -1, "volume": 1}], VALID_TYPES)
    assert "items[0].weight" in fields_of(submit_raw(client, p))


def test_negative_volume(client):
    p = make_problem([{"id": "a", "weight": 1, "volume": -0.5}], VALID_TYPES)
    assert "items[0].volume" in fields_of(submit_raw(client, p))


def test_non_finite_weight(client):
    for bad in (math.nan, math.inf, -math.inf):
        p = make_problem([{"id": "a", "weight": bad, "volume": 1}], VALID_TYPES)
        detail = submit_raw_allow_nan(client, p)
        assert "items[0].weight" in fields_of(detail), bad


def test_non_finite_volume(client):
    p = make_problem([{"id": "a", "weight": 1, "volume": math.nan}], VALID_TYPES)
    assert "items[0].volume" in fields_of(submit_raw_allow_nan(client, p))


def test_item_bigger_than_any_vehicle_type(client):
    p = make_problem([{"id": "huge", "weight": 11, "volume": 1}], VALID_TYPES)
    detail = submit_raw(client, p)
    assert "items[0]" in fields_of(detail)
    assert any("does not fit" in e["message"] for e in detail)


def test_too_many_exclusive_items(client):
    items = [{"id": f"e{k}", "weight": 1, "volume": 1, "exclusive": True} for k in range(3)]
    types = [{"name": "t", "max_weight": 10, "max_volume": 10, "cost": 1, "count": 2}]
    detail = submit_raw(client, make_problem(items, types))
    assert any("exclusive" in e["message"] for e in detail)


def test_conflict_references_unknown_category(client):
    p = make_problem(
        [{"id": "a", "weight": 1, "volume": 1, "category": "food"}],
        VALID_TYPES,
        conflicts=[["food", "ghost"]],
    )
    detail = submit_raw(client, p)
    assert "conflicts[0]" in fields_of(detail)
    assert any("ghost" in e["message"] for e in detail)


def test_bad_vehicle_type_fields(client):
    p = make_problem(VALID_ITEMS, [{"name": "t", "max_weight": 0, "max_volume": 10, "cost": 1, "count": 1}])
    assert "vehicle_types[0].max_weight" in fields_of(submit_raw(client, p))
    p = make_problem(VALID_ITEMS, [{"name": "t", "max_weight": 10, "max_volume": 10, "cost": -1, "count": 1}])
    assert "vehicle_types[0].cost" in fields_of(submit_raw(client, p))
    p = make_problem(VALID_ITEMS, [{"name": "t", "max_weight": 10, "max_volume": 10, "cost": 1, "count": -1}])
    assert "vehicle_types[0].count" in fields_of(submit_raw(client, p))


def test_aggregate_capacity_infeasible(client):
    items = [{"id": f"i{k}", "weight": 9, "volume": 1} for k in range(3)]
    types = [{"name": "t", "max_weight": 10, "max_volume": 10, "cost": 1, "count": 2}]
    detail = submit_raw(client, make_problem(items, types))
    assert "vehicle_types" in fields_of(detail)


def test_duplicate_item_id(client):
    p = make_problem(
        [{"id": "a", "weight": 1, "volume": 1}, {"id": "a", "weight": 1, "volume": 1}],
        VALID_TYPES,
    )
    assert "items[1].id" in fields_of(submit_raw(client, p))
