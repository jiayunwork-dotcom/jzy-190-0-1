"""下界推导。

从五个角度给出费用下界与车辆数下界，各自独立合法，取费用最大者作为最终下界：

1. 平凡下界：有货就至少一辆车，费用不低于最便宜车型。
2. 重量：总重量按"单位载重费用最便宜"的车型做分数装载（允许切开货），
   再按各车型可用辆数做带可用上限的分数装载（LP 松弛，更紧；容量不足即不可行证明）。
3. 体积：同理。
4. 独占货：每票独占货必须单独占一辆车，求货 -> 车型的最小费用二分匹配
   （最小费用流），同时得到一个可直接复用的独占货指派。
5. 互斥：对每对冲突类别 (A,B)，A、B 两类货彼此不能同车，故至少需要
   ceil(W_A/Q_A) + ceil(W_B/Q_B) 辆车（Q 为能装下该类最重货的车型最大载重，
   分数装载放松了货物不可分割约束）。自冲突 (A,A) 表示同类互斥，按单类计算。

这些放松只解除了"货物不可分割"等约束，没有解除任何一个真实约束，
因此结果必然不大于任何可行方案的费用，即合法下界。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .mincostflow import MinCostFlow
from .models import EPS, Problem, Shipment, VehicleType

# 平局时的固定优先级，保证同输入输出确定。
PRIORITY = ["weight", "volume", "exclusive", "conflict", "trivial"]


@dataclass
class DimBound:
    cost: float
    vehicles: int
    detail: str


@dataclass
class ExclusiveAssignment:
    """独占货 -> 车型的最小费用指派。"""

    cost: float
    assignment: dict[str, str]  # shipment_id -> type_id


@dataclass
class LowerBound:
    feasible: bool
    reason: str | None  # 不可行性证明来源
    cost: float
    vehicles: int
    binding: str  # weight / volume / exclusive / conflict / trivial
    dimensions: dict[str, DimBound] = field(default_factory=dict)
    exclusive_assignment: ExclusiveAssignment | None = None


def ceil_div(num: float, den: float) -> int:
    """带容差的 ceil(num/den)；num 近零时为 0。den 必须为正。"""
    if num <= EPS * max(1.0, abs(den)):
        return 0
    return int(math.ceil(num / den - EPS))


def _fractional_cost_unlimited(amount: float, cap_attr: str, types: list[VehicleType]) -> float:
    """忽略车型可用量：单位容量费用最便宜的车型可无限辆使用（分数装载）。"""
    if amount <= 0.0:
        return 0.0
    usable = [t for t in types if getattr(t, cap_attr) > 0.0]
    if not usable:
        return float("inf")
    cheapest = min(usable, key=lambda t: (t.cost / getattr(t, cap_attr), t.id))
    return amount * cheapest.cost / getattr(cheapest, cap_attr)


def _fractional_cost_with_availability(
    amount: float, cap_attr: str, types: list[VehicleType]
) -> float:
    """考虑车型可用辆数的分数装载 LP。容量不足返回 inf（不可行证明）。"""
    if amount <= 0.0:
        return 0.0
    usable = [t for t in types if getattr(t, cap_attr) > 0.0]
    if not usable:
        return float("inf")
    usable.sort(key=lambda t: (t.cost / getattr(t, cap_attr), t.id))
    remaining = amount
    total = 0.0
    for t in usable:
        cap = getattr(t, cap_attr) * t.available
        take = min(remaining, cap)
        total += take * (t.cost / getattr(t, cap_attr))
        remaining -= take
        if remaining <= EPS:
            return total
    return float("inf")


def _max_cap(cap_attr: str, types: list[VehicleType]) -> float:
    vals = [getattr(t, cap_attr) for t in types]
    return max(vals) if vals else 0.0


def _max_cap_for(
    shipments: list[Shipment], cap_attr: str, dim_attr: str, types: list[VehicleType]
) -> float:
    """能装下这些货（按单一维度）的车型的最大该维度容量；无货返回 0。"""
    if not shipments:
        return 0.0
    need = max(getattr(s, dim_attr) for s in shipments)
    vals = [
        getattr(t, cap_attr)
        for t in types
        if getattr(t, cap_attr) + EPS * max(1.0, getattr(t, cap_attr)) >= need
    ]
    return max(vals) if vals else 0.0


def solve_exclusive_assignment(
    problem: Problem,
) -> tuple[float | None, dict[str, str], str | None]:
    """最小费用流求独占货 -> 车型的最小费用匹配。

    返回 (费用, 指派 shipment_id->type_id, 不可行原因)。无独占货时费用为 0。
    """
    exclusive = [s for s in problem.shipments if s.exclusive]
    if not exclusive:
        return 0.0, {}, None

    types = problem.vehicle_types
    E, K = len(exclusive), len(types)
    # 节点：0=源，1..E=货，E+1..E+K=车型，E+K+1=汇
    ship_base = 1
    type_base = 1 + E
    sink = 1 + E + K
    mcf = MinCostFlow(sink + 1)

    for i in range(E):
        mcf.add_edge(0, ship_base + i, 1, 0.0)
    match_edges: list[tuple[int, str, str]] = []
    for i, ship in enumerate(exclusive):
        for j, vt in enumerate(types):
            if (
                ship.weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
                and ship.volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
            ):
                idx = mcf.add_edge(ship_base + i, type_base + j, 1, float(vt.cost))
                match_edges.append((idx, ship.id, vt.id))
    for j, vt in enumerate(types):
        mcf.add_edge(type_base + j, sink, vt.available, 0.0)

    result = mcf.solve(0, sink, E)
    if not result.feasible:
        return None, {}, "exclusive_capacity"

    assignment: dict[str, str] = {}
    for edge_idx, ship_id, type_id in match_edges:
        if result.flows[edge_idx] > 0:
            assignment[ship_id] = type_id
    return result.total_cost, assignment, None


def _bin_pack_dim_bound(
    ships: list[Shipment], types: list[VehicleType]
) -> tuple[float, int]:
    """一组（彼此可同车的）货按分数装箱的费用/辆数下界，取重量、体积两维最紧者。

    每辆车的两维容量独立放松：重量维允许只按重量装载（忽略体积），反之亦然，
    故两维各算一个合法下界，取最紧。车型可用量受限的分数 LP 一并参与取最紧。
    """
    if not ships:
        return 0.0, 0
    total_w = sum(s.weight for s in ships)
    total_v = sum(s.volume for s in ships)

    # 只保留能装下本组最大单件的车型（其余车型对本组完全不可用）。
    need_w = max(s.weight for s in ships)
    need_v = max(s.volume for s in ships)
    usable = [
        t for t in types
        if t.max_weight + EPS * max(1.0, t.max_weight) >= need_w
        and t.max_volume + EPS * max(1.0, t.max_volume) >= need_v
    ]
    if not usable:
        return float("inf"), 10 ** 9

    # 重量维：允许切开货，只用重量容量（体积放松为无穷）。
    w_cost = max(
        _fractional_cost_unlimited(total_w, "max_weight", usable),
        _fractional_cost_with_availability(total_w, "max_weight", usable),
    )
    w_count = ceil_div(total_w, max(t.max_weight for t in usable))
    # 体积维同理。
    v_cost = max(
        _fractional_cost_unlimited(total_v, "max_volume", usable),
        _fractional_cost_with_availability(total_v, "max_volume", usable),
    )
    v_count = ceil_div(total_v, max(t.max_volume for t in usable))

    if w_cost >= v_cost:
        return w_cost, w_count
    return v_cost, v_count


def _cheapest_fit_cost(ship: Shipment, types: list[VehicleType]) -> float:
    """能同时装下该票货重量与体积的最便宜车型费用。"""
    fits = [
        t for t in types
        if ship.weight <= t.max_weight + EPS * max(1.0, t.max_weight)
        and ship.volume <= t.max_volume + EPS * max(1.0, t.max_volume)
    ]
    return min(t.cost for t in fits)


def _fractional_fleet_value(
    total_w: float, total_v: float, types: list[VehicleType]
) -> float:
    """一个互斥组（与其它组不共享车辆）的分数装载费用，取重量/体积维最紧。

    注：这里的车型可用量是"全车队可用量"，多个组各自做带相同上限的分数 LP 后直接
    相加，会放松"组间共享同一批车配额"的约束，故仍是合法（偏松）下界。
    """
    return max(
        _fractional_cost_unlimited(total_w, "max_weight", types),
        _fractional_cost_with_availability(total_w, "max_weight", types),
        _fractional_cost_unlimited(total_v, "max_volume", types),
        _fractional_cost_with_availability(total_v, "max_volume", types),
    )


def _category_separated_bound(
    problem: Problem,
) -> tuple[float, int, str] | None:
    """按冲突类别隔离的分数装箱下界（比逐类别只算自己更紧）。

    每个出现在冲突关系里的类别各自成一组（其车不能与冲突对象共享），
    不属于任何冲突类别的货合成"中性组"。组间车辆不共享，分别做分数装载后费用相加。
    这相对真实的冲突图着色是放松（可能允许了实际不允许的共享），因此仍是合法下界；
    容量不足时返回 None。
    """
    conflict_cats = set(problem.conflict_adj)
    cat_ships: dict[str, list[Shipment]] = {c: [] for c in conflict_cats}
    neutral_w = neutral_v = 0.0
    for s in problem.shipments:
        if s.category in conflict_cats:
            cat_ships[s.category].append(s)
        else:
            neutral_w += s.weight
            neutral_v += s.volume

    total_cost = 0.0
    total_count = 0
    parts: list[str] = []
    for cat in sorted(cat_ships):
        ships = cat_ships[cat]
        if not ships:
            continue
        cw = sum(s.weight for s in ships)
        cv = sum(s.volume for s in ships)
        val = _fractional_fleet_value(cw, cv, problem.vehicle_types)
        if val == float("inf"):
            return None
        total_cost += val
        total_count += _bin_pack_dim_bound(ships, problem.vehicle_types)[1]
        parts.append(f"{cat}≈{_r(val)}")
    if neutral_w > EPS or neutral_v > EPS:
        val = _fractional_fleet_value(neutral_w, neutral_v, problem.vehicle_types)
        if val == float("inf"):
            return None
        total_cost += val
        parts.append(f"中性≈{_r(val)}")
    if not parts:
        return 0.0, 0, ""
    return total_cost, total_count, " + ".join(parts)


def _self_conflict_bound(ships: list[Shipment], types: list[VehicleType]) -> tuple[float, int]:
    """自冲突类别（同类货两两不能同车）：每票必须独占一辆可行车型。"""
    if not ships:
        return 0.0, 0
    total = sum(_cheapest_fit_cost(s, types) for s in ships)
    return total, len(ships)


def _conflict_dim(
    problem: Problem,
    by_category: dict[str, list[Shipment]],
) -> DimBound:
    """冲突类别给出的下界，对每个冲突对分别算后取最大。

    - 普通冲突对 (A,B)：A 与 B 的货不能同车，但各自内部可同车。
      两侧各算一个分数装箱下界（车辆不共享），费用/辆数相加。
    - 自冲突 (A,A)：同类两两不能同车，每票独占一车。
    """
    types = problem.vehicle_types
    best = DimBound(0.0, 0, "无冲突类别对")
    for a, b in sorted(problem.conflicts):
        if a == b:
            cost, count = _self_conflict_bound(by_category.get(a, []), types)
            detail = f"自冲突类别 {a}：{len(by_category.get(a, []))} 票货各占一车，至少 {count} 辆"
        else:
            ca_cost, ca = _bin_pack_dim_bound(by_category.get(a, []), types)
            cb_cost, cb = _bin_pack_dim_bound(by_category.get(b, []), types)
            cost = ca_cost + cb_cost
            count = ca + cb
            detail = f"冲突类别 {a}/{b} 各需 {ca}+{cb}={count} 辆（分数装箱下界）"
        if cost > best.cost + EPS or (abs(cost - best.cost) <= EPS and count > best.vehicles):
            best = DimBound(cost, count, detail)

    # 更紧的全局冲突界：所有冲突类别各自隔离 + 中性组，分组分数装载费用之和。
    sep = _category_separated_bound(problem)
    if sep is not None:
        cost, count, detail = sep
        if cost > best.cost + EPS or (abs(cost - best.cost) <= EPS and count > best.vehicles):
            best = DimBound(cost, count, f"冲突类别隔离分数装载：{detail}")
    return best


def compute_lower_bound(problem: Problem) -> LowerBound:
    shipments = problem.shipments
    types = problem.vehicle_types

    # ---- 单件可行性（防御性检查，正常输入已由校验层拦截）----
    for s in shipments:
        if not any(
            s.weight <= t.max_weight + EPS * max(1.0, t.max_weight)
            and s.volume <= t.max_volume + EPS * max(1.0, t.max_volume)
            for t in types
        ):
            return LowerBound(False, f"shipment_too_large:{s.id}", 0.0, 0, "weight")

    total_w = sum(s.weight for s in shipments)
    total_v = sum(s.volume for s in shipments)

    # ---- 重量 ----
    w_count = ceil_div(total_w, _max_cap("max_weight", types))
    w_cost = _fractional_cost_unlimited(total_w, "max_weight", types)
    w_cost_avail = _fractional_cost_with_availability(total_w, "max_weight", types)
    if w_cost_avail == float("inf"):
        return LowerBound(False, "total_weight_capacity", 0.0, 0, "weight")
    w_cost = max(w_cost, w_cost_avail)
    w_detail = f"总重 {_r(total_w)} 吨分数装载至少 {w_count} 辆"

    # ---- 体积 ----
    v_count = ceil_div(total_v, _max_cap("max_volume", types))
    v_cost = _fractional_cost_unlimited(total_v, "max_volume", types)
    v_cost_avail = _fractional_cost_with_availability(total_v, "max_volume", types)
    if v_cost_avail == float("inf"):
        return LowerBound(False, "total_volume_capacity", 0.0, 0, "volume")
    v_cost = max(v_cost, v_cost_avail)
    v_detail = f"总体积 {_r(total_v)} 方分数装载至少 {v_count} 辆"

    # ---- 独占货 ----
    ex_cost, ex_assign, ex_reason = solve_exclusive_assignment(problem)
    if ex_reason is not None:
        return LowerBound(False, ex_reason, 0.0, 0, "exclusive")
    ex_count = sum(1 for s in shipments if s.exclusive)
    ex_detail = f"{ex_count} 票独占货各占一车，最小匹配费用 {_r(ex_cost)}"

    # ---- 冲突类别 ----
    by_category: dict[str, list[Shipment]] = {}
    for s in shipments:
        if s.category is not None:
            by_category.setdefault(s.category, []).append(s)
    conflict_dim = _conflict_dim(problem, by_category)

    # ---- 平凡下界 ----
    if shipments:
        t_cost = min(t.cost for t in types)
        t_dim = DimBound(float(t_cost), 1, "有货至少需要一辆车")
    else:
        t_dim = DimBound(0.0, 0, "空任务，无需车辆")

    dims: dict[str, DimBound] = {
        "weight": DimBound(w_cost, w_count, w_detail),
        "volume": DimBound(v_cost, v_count, v_detail),
        "exclusive": DimBound(ex_cost, ex_count, ex_detail),
        "conflict": conflict_dim,
        "trivial": t_dim,
    }

    binding = max(PRIORITY, key=lambda name: (dims[name].cost, -PRIORITY.index(name)))
    vehicles = max(d.vehicles for d in dims.values())

    return LowerBound(
        feasible=True,
        reason=None,
        cost=dims[binding].cost,
        vehicles=vehicles,
        binding=binding,
        dimensions=dims,
        exclusive_assignment=ExclusiveAssignment(ex_cost, ex_assign),
    )


def _r(x: float) -> float:
    r = round(float(x), 6)
    return int(r) if float(r).is_integer() else r
