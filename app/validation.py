"""业务校验：所有错误都指出具体字段路径，便于调度系统定位。"""
from __future__ import annotations

import math

from .solver.types import build_problem


def validate_problem(raw: dict) -> list[dict]:
    """返回错误列表 [{"field": ..., "message": ...}]，空列表表示合法。"""
    errors: list[dict] = []

    def err(field: str, message: str) -> None:
        errors.append({"field": field, "message": message})

    types = raw["vehicle_types"]
    items = raw["items"]

    # 1. 数值本身合法（有限、符号正确）
    nums_ok = True
    for i, t in enumerate(types):
        for f in ("max_weight", "max_volume"):
            v = t[f]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                err(f"vehicle_types[{i}].{f}", "must be a finite number > 0")
                nums_ok = False
        v = t["cost"]
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            err(f"vehicle_types[{i}].cost", "must be a finite number >= 0")
            nums_ok = False
        if t["count"] < 0:
            err(f"vehicle_types[{i}].count", "must be >= 0")
            nums_ok = False

    seen: set[str] = set()
    for i, it in enumerate(items):
        if not it["id"]:
            err(f"items[{i}].id", "must be a non-empty string")
        elif it["id"] in seen:
            err(f"items[{i}].id", f"duplicate item id '{it['id']}'")
        seen.add(it["id"])
        for f in ("weight", "volume"):
            v = it[f]
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                err(f"items[{i}].{f}", "must be a finite number >= 0")
                nums_ok = False

    # 2. 冲突关系引用的类别必须存在（声明过的或货物实际用到的）
    universe = set(raw["categories"]) | {
        it["category"] for it in items if it["category"] is not None
    }
    for i, pair in enumerate(raw["conflicts"]):
        for c in pair:
            if c not in universe:
                err(f"conflicts[{i}]", f"unknown category '{c}'")

    if not nums_ok:
        return errors

    # 3. 以下检查在整数缩放后的精确表示上进行
    problem, _, _ = build_problem(raw)

    # 某票货比任何车型都大
    for i, it in enumerate(problem.items):
        if not any(t.capw >= it.w and t.capv >= it.v for t in problem.types):
            err(
                f"items[{i}]",
                f"item '{it.id}' (weight={items[i]['weight']}, volume={items[i]['volume']}) "
                "does not fit any vehicle type",
            )

    # 独占货多于所有车型的可用总数
    n_excl = sum(1 for it in problem.items if it.exclusive)
    total_vehicles = sum(t.count for t in problem.types)
    if n_excl > total_vehicles:
        err(
            "items",
            f"{n_excl} exclusive items require dedicated vehicles, "
            f"but only {total_vehicles} vehicles available in total",
        )

    # 总量可行性（精确整数比较）
    total_w = sum(it.w for it in problem.items)
    total_v = sum(it.v for it in problem.items)
    cap_w = sum(t.count * t.capw for t in problem.types)
    cap_v = sum(t.count * t.capv for t in problem.types)
    if cap_w < total_w:
        err(
            "vehicle_types",
            f"total weight capacity is less than total item weight: instance is infeasible",
        )
    if cap_v < total_v:
        err(
            "vehicle_types",
            f"total volume capacity is less than total item volume: instance is infeasible",
        )
    return errors
