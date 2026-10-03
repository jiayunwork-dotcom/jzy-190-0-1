"""pytest 共享夹具与构造辅助。"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.schemas import JobRequest  # noqa: E402
from app.serialize import build_problem  # noqa: E402


def make_problem(payload: dict):
    return build_problem(JobRequest.model_validate(payload))


def one_truck_type(cost=100, cap_weight=10, cap_volume=1e9, available=10, type_id="T"):
    return {
        "id": type_id,
        "max_weight": cap_weight,
        "max_volume": cap_volume,
        "cost": cost,
        "available": available,
    }
