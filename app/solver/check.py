"""方案可行性独立校验：测试与作业落库前都会用到。"""
from __future__ import annotations

from .types import Problem


def verify_solution(p: Problem, snap: dict) -> list[str]:
    """返回错误列表，空列表表示方案完整可行。"""
    errs: list[str] = []
    by_id = {it.id: it for it in p.items}
    type_by_name = {t.name: (i, t) for i, t in enumerate(p.types)}
    seen: set[str] = set()
    usage = [0] * len(p.types)
    total = 0.0

    vehicles = snap.get("vehicles", [])
    for vi, v in enumerate(vehicles):
        tref = type_by_name.get(v.get("vehicle_type"))
        if tref is None:
            errs.append(f"vehicles[{vi}]: unknown vehicle type '{v.get('vehicle_type')}'")
            continue
        tidx, t = tref
        usage[tidx] += 1
        total += t.cost
        w = 0
        vol = 0
        cats: dict[str, int] = {}
        n_excl = 0
        ids = v.get("items", [])
        for iid in ids:
            it = by_id.get(iid)
            if it is None:
                errs.append(f"vehicles[{vi}]: unknown item '{iid}'")
                continue
            if iid in seen:
                errs.append(f"vehicles[{vi}]: item '{iid}' assigned more than once")
            seen.add(iid)
            w += it.w
            vol += it.v
            if it.cat is not None:
                cats[it.cat] = cats.get(it.cat, 0) + 1
            if it.exclusive:
                n_excl += 1
        if w > t.capw:
            errs.append(f"vehicles[{vi}]: weight {w} exceeds capacity {t.capw}")
        if vol > t.capv:
            errs.append(f"vehicles[{vi}]: volume {vol} exceeds capacity {t.capv}")
        if n_excl and len(ids) > 1:
            errs.append(f"vehicles[{vi}]: exclusive item shares a vehicle")
        cl = list(cats)
        for a_i in range(len(cl)):
            for b_i in range(a_i, len(cl)):
                if frozenset((cl[a_i], cl[b_i])) in p.conflict_set:
                    errs.append(
                        f"vehicles[{vi}]: conflicting categories '{cl[a_i]}' and '{cl[b_i]}' share a vehicle"
                    )

    missing = [iid for iid in by_id if iid not in seen]
    if missing:
        errs.append(f"items not assigned: {missing}")
    for tidx, u in enumerate(usage):
        if u > p.types[tidx].count:
            errs.append(
                f"vehicle type '{p.types[tidx].name}' used {u} > available {p.types[tidx].count}"
            )
    if abs(total - snap.get("total_cost", -1.0)) > 1e-6:
        errs.append(
            f"total_cost mismatch: reported {snap.get('total_cost')} vs computed {total}"
        )
    if snap.get("vehicle_count") != len(vehicles):
        errs.append("vehicle_count mismatch")
    return errs
