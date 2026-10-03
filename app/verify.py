"""方案可行性独立校验。

求解器内部维护方案一致性，交付前再由本模块独立核验一遍：
恰好覆盖、不超载、不超容、冲突类别不同车、独占货独占、车型用量不超上限、费用一致。
"""
from __future__ import annotations

from .models import EPS, Problem, Solution


class InfeasibleSolution(Exception):
    """方案不满足约束，message 指出具体问题。"""


def verify(problem: Problem, solution: Solution, expected_cost: float | None = None) -> None:
    """校验通过返回 None，否则抛 InfeasibleSolution。"""
    shipments_by_id = {s.id: s for s in problem.shipments}
    type_by_id = {t.id: t for t in problem.vehicle_types}

    seen: dict[str, str] = {}
    for vi, v in enumerate(solution.vehicles):
        if v.type_id not in type_by_id:
            raise InfeasibleSolution(f"第 {vi} 辆车引用了不存在的车型 {v.type_id}")
        vt = type_by_id[v.type_id]

        w = vol = 0.0
        cat_counts: dict[str, int] = {}
        exclusive_ids: list[str] = []
        for sid in v.shipment_ids:
            if sid not in shipments_by_id:
                raise InfeasibleSolution(f"第 {vi} 辆车引用了不存在的货 {sid}")
            if sid in seen:
                raise InfeasibleSolution(f"货 {sid} 同时出现在第 {seen[sid]} 辆和第 {vi} 辆车上")
            seen[sid] = str(vi)
            ship = shipments_by_id[sid]
            if ship.exclusive:
                exclusive_ids.append(sid)
            w += ship.weight
            vol += ship.volume
            if ship.category is not None:
                cat_counts[ship.category] = cat_counts.get(ship.category, 0) + 1

        if w > vt.max_weight + EPS * max(1.0, vt.max_weight):
            raise InfeasibleSolution(f"第 {vi} 辆车（车型 {vt.id}）超载：{w} > {vt.max_weight}")
        if vol > vt.max_volume + EPS * max(1.0, vt.max_volume):
            raise InfeasibleSolution(f"第 {vi} 辆车（车型 {vt.id}）超容：{vol} > {vt.max_volume}")

        # 与内部统计交叉核对
        if abs(w - v.weight) > 1e-7 * max(1.0, w):
            raise InfeasibleSolution(f"第 {vi} 辆车内部重量统计不一致")
        if abs(vol - v.volume) > 1e-7 * max(1.0, vol):
            raise InfeasibleSolution(f"第 {vi} 辆车内部体积统计不一致")
        if set(cat_counts) != set(v.categories):
            raise InfeasibleSolution(f"第 {vi} 辆车内部类别统计不一致")

        # 独占货：一辆车只能有一票货，且该票货必须是独占货。
        if len(v.shipment_ids) >= 2 and exclusive_ids:
            raise InfeasibleSolution(
                f"第 {vi} 辆车上独占货 {exclusive_ids} 与其它货同车"
            )
        if len(v.shipment_ids) == 1:
            only = shipments_by_id[v.shipment_ids[0]]
            if not only.exclusive:
                pass  # 非独占货单独成车是允许的（浪费，但合法）
        # 冲突类别：同车上任意两个类别不得构成冲突对（自冲突要求同类至多一票）。
        for cat, n in cat_counts.items():
            for other in cat_counts:
                pair = (cat, other) if cat <= other else (other, cat)
                if pair in problem.conflicts:
                    if cat == other:
                        if n > 1:
                            raise InfeasibleSolution(
                                f"第 {vi} 辆车上自冲突类别 {cat} 出现 {n} 票"
                            )
                    else:
                        raise InfeasibleSolution(
                            f"第 {vi} 辆车上冲突类别 {cat} 与 {other} 同车"
                        )

    missing = set(shipments_by_id) - set(seen)
    if missing:
        raise InfeasibleSolution(f"以下货物未分配：{sorted(missing)}")

    counts = solution.type_counts()
    for t in problem.vehicle_types:
        if counts[t.id] > t.available:
            raise InfeasibleSolution(
                f"车型 {t.id} 使用 {counts[t.id]} 辆，超过可用上限 {t.available}"
            )

    if expected_cost is not None:
        actual = solution.cost()
        if abs(actual - expected_cost) > 1e-7 * max(1.0, actual):
            raise InfeasibleSolution(f"费用统计不一致：{actual} != {expected_cost}")


def is_feasible(problem: Problem, solution: Solution) -> bool:
    try:
        verify(problem, solution)
    except InfeasibleSolution:
        return False
    return True
