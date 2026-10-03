"""输入校验测试：负/非有限数、超尺寸、独占货超总数、冲突引用不存在类别等，逐字段报错。"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.schemas import JobRequest, validate_problem


def errors_for(payload: dict):
    req = JobRequest.model_validate(payload)
    return validate_problem(req)


VALID_TYPE = {"id": "T", "max_weight": 10, "max_volume": 100, "cost": 100, "available": 5}


def test_negative_weight_rejected_with_field_name():
    with pytest.raises(ValidationError) as exc:
        JobRequest.model_validate(
            {"shipments": [{"id": "a", "weight": -3}], "vehicle_types": [VALID_TYPE]}
        )
    locs = [".".join(str(x) for x in e["loc"] if x != "body") for e in exc.value.errors()]
    assert any("shipments" in loc and "weight" in loc for loc in locs)


def test_nan_and_infinity_rejected():
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValidationError):
            JobRequest.model_validate(
                {"shipments": [{"id": "a", "weight": bad}], "vehicle_types": [VALID_TYPE]}
            )


def test_negative_vehicle_cost_rejected():
    bad_type = dict(VALID_TYPE, cost=-1)
    with pytest.raises(ValidationError):
        JobRequest.model_validate({"shipments": [], "vehicle_types": [bad_type]})


def test_shipment_larger_than_any_type_points_to_field():
    errs = errors_for(
        {
            "shipments": [{"id": "huge", "weight": 99, "volume": 0}],
            "vehicle_types": [VALID_TYPE],
        }
    )
    assert any(e.field == "shipments[0]" and "比任何可用车型都大" in e.message for e in errs)


def test_shipment_volume_too_large_points_to_field():
    errs = errors_for(
        {
            "shipments": [{"id": "bulky", "weight": 1, "volume": 999}],
            "vehicle_types": [VALID_TYPE],
        }
    )
    assert any("shipments[0]" in e.field for e in errs)


def test_too_many_exclusive_shipments():
    errs = errors_for(
        {
            "shipments": [
                {"id": f"e{i}", "weight": 1, "exclusive": True} for i in range(6)
            ],
            "vehicle_types": [VALID_TYPE],  # available=5
        }
    )
    assert any(e.field == "shipments.exclusive" and "独占货" in e.message for e in errs)


def test_conflict_unknown_category_reported():
    errs = errors_for(
        {
            "shipments": [{"id": "a", "weight": 1, "category": "food"}],
            "vehicle_types": [VALID_TYPE],
            "conflicts": [["food", "ghost_category"]],
        }
    )
    assert any("conflicts[0]" in e.field and "ghost_category" in e.message for e in errs)


def test_duplicate_shipment_id_reported():
    errs = errors_for(
        {
            "shipments": [{"id": "a", "weight": 1}, {"id": "a", "weight": 1}],
            "vehicle_types": [VALID_TYPE],
        }
    )
    assert any("id" in e.field and "重复" in e.message for e in errs)


def test_misspelled_field_rejected():
    with pytest.raises(ValidationError):
        JobRequest.model_validate(
            {"shipments": [{"id": "a", "weight": 1, "volum": 2}], "vehicle_types": [VALID_TYPE]}
        )


def test_conflict_pair_must_have_two_elements():
    with pytest.raises(ValidationError):
        JobRequest.model_validate(
            {
                "shipments": [{"id": "a", "weight": 1, "category": "A"}],
                "vehicle_types": [VALID_TYPE],
                "conflicts": [["A"]],
            }
        )


def test_valid_request_passes():
    errs = errors_for(
        {
            "shipments": [
                {"id": "a", "weight": 3, "category": "food"},
                {"id": "b", "weight": 2, "category": "chem", "exclusive": True},
            ],
            "vehicle_types": [VALID_TYPE],
            "conflicts": [["food", "chem"]],
            "seed": 7,
            "time_limit_seconds": 1.0,
        }
    )
    assert errs == []
