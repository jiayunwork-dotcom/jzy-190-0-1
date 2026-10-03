"""在 API 输入、领域模型、可 JSON 序列化输出之间转换。"""
from __future__ import annotations

from typing import Any

from .bounds import LowerBound
from .models import Problem, Shipment, Solution, VehicleType
from .schemas import JobRequest

Status = str  # pending/running/completed/timeout/cancelled/interrupted/infeasible


def build_problem(req: JobRequest) -> Problem:
    shipments = [
        Shipment(
            id=s.id,
            weight=float(s.weight),
            volume=float(s.volume),
            category=s.category,
            exclusive=bool(s.exclusive),
            index=i,
        )
        for i, s in enumerate(req.shipments)
    ]
    types = [
        VehicleType(
            id=t.id,
            max_weight=float(t.max_weight),
            max_volume=float(t.max_volume),
            cost=float(t.cost),
            available=int(t.available),
        )
        for t in req.vehicle_types
    ]
    conflicts: set[tuple[str, str]] = set()
    adj: dict[str, set[str]] = {}
    for a, b in req.conflicts:
        pair = (a, b) if a <= b else (b, a)
        conflicts.add(pair)
    for a, b in conflicts:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return Problem(shipments=shipments, vehicle_types=types, conflicts=conflicts, conflict_adj=adj)


def solution_to_dict(problem: Problem, sol: Solution) -> dict[str, Any]:
    vehicles = []
    for i, v in enumerate(sol.vehicles):
        vt = problem.vehicle_type(v.type_id)
        vehicles.append(
            {
                "vehicle_index": i,
                "type_id": v.type_id,
                "cost": _num(vt.cost),
                "shipment_ids": list(v.shipment_ids),
                "weight": _num(v.weight),
                "volume": _num(v.volume),
                "exclusive": len(v.shipment_ids) == 1
                and problem.shipment(v.shipment_ids[0]).exclusive,
            }
        )
    counts = sol.type_counts()
    return {
        "vehicle_count": sol.vehicle_count(),
        "total_cost": _num(sol.cost()),
        "type_counts": {t.id: counts[t.id] for t in problem.vehicle_types},
        "vehicles": vehicles,
    }


def lower_bound_to_dict(lb: LowerBound) -> dict[str, Any]:
    return {
        "cost": _num(lb.cost),
        "vehicles": lb.vehicles,
        "binding": lb.binding,
        "dimensions": {
            name: {
                "cost": _num(d.cost),
                "vehicles": d.vehicles,
                "detail": d.detail,
            }
            for name, d in lb.dimensions.items()
        },
    }


def _num(x: float) -> float | int:
    r = round(float(x), 6)
    return int(r) if float(r).is_integer() else r
