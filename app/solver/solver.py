"""求解器：构造式启发 + 局部搜索 + 迭代局部搜索（ILS）。

- 构造：独占货先各自开最便宜可用车型；其余货按重量降序 best-fit 装入，
  装不下则开新车（能装下它的最便宜且还有剩余数量的车型）。
- 局部搜索（都保持可行、只在严格降费时接受）：
  1) 换型：在数量限制内给每辆车重新分配最便宜的可用车型；
  2) 整车消除：尝试把某辆车的货全部并入其它车，空车即省下包车费；
  3) 迁移：把一票货移到另一辆车，源车与目标车同时最优换型后费用下降则接受
     （源车变空、或源车可降为更便宜车型时尤其有效）。
- ILS：对当前解做小幅扰动（随机搬动几票货）后重新局部搜索，
  更好则接受，长期无改进则回到历史最好解重新扰动。轮数预算只随票数变化，
  随机数全部来自 seed，因此搜索轨迹确定（时间上限截断除外）。

任何时刻内存中的解都是完整可行的：每个动作先算好结果再整体生效，
取消 / 超时只发生在动作之间，不会停在改到一半的状态。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .types import Problem

_EPS = 1e-9  # 费用改进的最小显著量
_COST_TIE = 1e-12  # 车型费用并列判定
_MAX_LOCAL_PASSES = 2000  # 单次局部搜索的保险上限
_ELIMINATE_MAX_ITEMS = 12  # 整车消除只尝试不超过该票数的车辆
_RESTART_AFTER = 25  # 连续无改进多少轮后回到最好解


def _tol(x: float) -> float:
    return 1e-9 * max(1.0, abs(x))


class InfeasibleError(Exception):
    """构造阶段无法完整排车（正常输入已被校验拦截，这里兜底）。"""


@dataclass(eq=False)
class Vehicle:
    type_idx: int
    items: list[int]  # 货物下标
    w: int
    v: int
    cats: dict[str, int]  # 车上各类别的票数
    sealed: bool  # 含独占货，禁止再装

    def clone(self) -> "Vehicle":
        return Vehicle(self.type_idx, list(self.items), self.w, self.v, dict(self.cats), self.sealed)


class Solution:
    def __init__(self, p: Problem):
        self.p = p
        self.vehicles: list[Vehicle] = []
        self.cost = 0.0

    def clone(self) -> "Solution":
        s = Solution(self.p)
        s.vehicles = [v.clone() for v in self.vehicles]
        s.cost = self.cost
        return s

    def usage(self) -> list[int]:
        u = [0] * len(self.p.types)
        for veh in self.vehicles:
            u[veh.type_idx] += 1
        return u


# ---------------------------------------------------------------- 基本操作

def _conflict(p: Problem, a: str, b: str) -> bool:
    return frozenset((a, b)) in p.conflict_set


def _add_item(p: Problem, veh: Vehicle, i: int) -> None:
    it = p.items[i]
    veh.items.append(i)
    veh.w += it.w
    veh.v += it.v
    if it.cat is not None:
        veh.cats[it.cat] = veh.cats.get(it.cat, 0) + 1


def _remove_item(p: Problem, veh: Vehicle, i: int) -> None:
    it = p.items[i]
    veh.items.remove(i)
    veh.w -= it.w
    veh.v -= it.v
    if it.cat is not None:
        left = veh.cats[it.cat] - 1
        if left:
            veh.cats[it.cat] = left
        else:
            del veh.cats[it.cat]


def _can_add(p: Problem, veh: Vehicle, i: int) -> bool:
    """按车辆当前车型判断能否再装第 i 票（不改车型）。"""
    it = p.items[i]
    if veh.sealed or it.exclusive:
        return False
    t = p.types[veh.type_idx]
    if veh.w + it.w > t.capw or veh.v + it.v > t.capv:
        return False
    if it.cat is not None:
        for c in veh.cats:
            if _conflict(p, it.cat, c):
                return False
    return True


def _cheapest_type(p: Problem, w: int, v: int, avail: list[int] | None) -> int:
    """能装 (w, v) 的最便宜车型下标；avail 为 None 表示不限数量。找不到返回 -1。"""
    best = -1
    for t, ty in enumerate(p.types):
        if avail is not None and avail[t] <= 0:
            continue
        if ty.capw >= w and ty.capv >= v:
            if best < 0 or ty.cost < p.types[best].cost - _COST_TIE:
                best = t
    return best


# ---------------------------------------------------------------- 构造

def _construct(p: Problem) -> Solution:
    sol = Solution(p)
    avail = [t.count for t in p.types]
    fit_count = [
        sum(1 for t in p.types if t.capw >= it.w and t.capv >= it.v) for it in p.items
    ]

    def open_vehicle(i: int) -> None:
        it = p.items[i]
        t = _cheapest_type(p, it.w, it.v, avail)
        if t < 0:
            raise InfeasibleError(f"no vehicle type available for item '{it.id}'")
        avail[t] -= 1
        veh = Vehicle(t, [], 0, 0, {}, False)
        _add_item(p, veh, i)
        veh.sealed = it.exclusive
        sol.vehicles.append(veh)
        sol.cost += p.types[t].cost

    # 独占货：可装车型最少的先排，同条件重者优先，编号定序保证确定
    excl = [i for i in range(len(p.items)) if p.items[i].exclusive]
    excl.sort(key=lambda i: (fit_count[i], -p.items[i].w, -p.items[i].v, p.items[i].id))
    for i in excl:
        open_vehicle(i)

    # 普通货：重量降序 best-fit（装后剩余载重最小者），装不下开新车
    rest = [i for i in range(len(p.items)) if not p.items[i].exclusive]
    rest.sort(key=lambda i: (-p.items[i].w, -p.items[i].v, p.items[i].id))
    for i in rest:
        best_vi, best_key = -1, None
        for vi, veh in enumerate(sol.vehicles):
            if _can_add(p, veh, i):
                t = p.types[veh.type_idx]
                key = (
                    t.capw - (veh.w + p.items[i].w),
                    t.capv - (veh.v + p.items[i].v),
                    vi,
                )
                if best_key is None or key < best_key:
                    best_key, best_vi = key, vi
        if best_vi < 0:
            open_vehicle(i)
        else:
            _add_item(p, sol.vehicles[best_vi], i)
    return sol


# ---------------------------------------------------------------- 局部搜索

def _retype_pass(sol: Solution) -> bool:
    """在数量限制内给所有车重新指派最便宜车型；总费用下降才生效。"""
    p = sol.p
    order = sorted(
        range(len(sol.vehicles)),
        key=lambda vi: (-sol.vehicles[vi].w, -sol.vehicles[vi].v, vi),
    )
    avail = [t.count for t in p.types]
    assign = [0] * len(sol.vehicles)
    for vi in order:
        veh = sol.vehicles[vi]
        t = _cheapest_type(p, veh.w, veh.v, avail)
        if t < 0:
            return False
        assign[vi] = t
        avail[t] -= 1
    new_cost = sum(p.types[assign[vi]].cost for vi in range(len(sol.vehicles)))
    if new_cost < sol.cost - _EPS:
        for vi in range(len(sol.vehicles)):
            sol.vehicles[vi].type_idx = assign[vi]
        sol.cost = new_cost
        return True
    return False


def _eliminate_pass(sol: Solution, stopped) -> bool:
    """尝试把某辆（较贵的、票数不多的）车的货全部并入其它车。"""
    p = sol.p
    order = sorted(
        range(len(sol.vehicles)),
        key=lambda vi: (-p.types[sol.vehicles[vi].type_idx].cost, vi),
    )
    for ai in order:
        if stopped():
            return False
        A = sol.vehicles[ai]
        if A.sealed or len(A.items) > _ELIMINATE_MAX_ITEMS:
            continue
        if p.types[A.type_idx].cost <= _EPS:
            continue  # 免费车空掉不省钱，也避免死循环
        sim_w: dict[int, int] = {}
        sim_v: dict[int, int] = {}
        sim_cats: dict[int, dict[str, int]] = {}
        plan: list[tuple[int, int]] = []
        ok = True
        for i in sorted(A.items, key=lambda i: (-p.items[i].w, -p.items[i].v, p.items[i].id)):
            it = p.items[i]
            best_bi, best_key = -1, None
            for bi, B in enumerate(sol.vehicles):
                if bi == ai or B.sealed:
                    continue
                bw = sim_w.get(bi, B.w)
                bv = sim_v.get(bi, B.v)
                t = p.types[B.type_idx]
                if bw + it.w > t.capw or bv + it.v > t.capv:
                    continue
                if it.cat is not None:
                    cats = sim_cats.get(bi, B.cats)
                    if any(_conflict(p, it.cat, c) for c in cats):
                        continue
                key = (t.capw - (bw + it.w), t.capv - (bv + it.v), bi)
                if best_key is None or key < best_key:
                    best_key, best_bi = key, bi
            if best_bi < 0:
                ok = False
                break
            plan.append((i, best_bi))
            B = sol.vehicles[best_bi]
            if best_bi not in sim_w:
                sim_w[best_bi] = B.w
                sim_v[best_bi] = B.v
                sim_cats[best_bi] = dict(B.cats)
            sim_w[best_bi] += it.w
            sim_v[best_bi] += it.v
            if it.cat is not None:
                sim_cats[best_bi][it.cat] = sim_cats[best_bi].get(it.cat, 0) + 1
        if not ok:
            continue
        for i, bi in plan:
            _remove_item(p, A, i)
            _add_item(p, sol.vehicles[bi], i)
        sol.cost -= p.types[A.type_idx].cost
        del sol.vehicles[ai]
        return True
    return False


def _best_pair_types(
    p: Problem, avail: list[int], aw: int, av: int, a_empty: bool, bw: int, bv: int
) -> tuple[float, int | None, int] | None:
    """迁移后源车(A')与目标车(B')的最优车型组合及费用；A' 为空则撤车。"""
    if a_empty:
        a_opts: list[int | None] = [None]
    else:
        a_opts = [
            t
            for t in range(len(p.types))
            if avail[t] > 0 and p.types[t].capw >= aw and p.types[t].capv >= av
        ]
        if not a_opts:
            return None
    best: tuple[float, int | None, int] | None = None
    for tA in a_opts:
        for tB in range(len(p.types)):
            tyB = p.types[tB]
            if tyB.capw < bw or tyB.capv < bv:
                continue
            if tA is None:
                if avail[tB] < 1:
                    continue
            elif tA == tB:
                if avail[tB] < 2:
                    continue
            elif avail[tA] < 1 or avail[tB] < 1:
                continue
            cost = (p.types[tA].cost if tA is not None else 0.0) + tyB.cost
            if best is None or cost < best[0] - _COST_TIE:
                best = (cost, tA, tB)
    return best


def _relocate_pass(sol: Solution, stopped) -> bool:
    """把一票货移到另一辆车（源/目标车同时最优换型），费用下降则接受。"""
    p = sol.p
    usage = sol.usage()
    remaining = [p.types[t].count - usage[t] for t in range(len(p.types))]
    for ai in range(len(sol.vehicles)):
        A = sol.vehicles[ai]
        if A.sealed:
            continue
        for i in list(A.items):
            if stopped():
                return False
            it = p.items[i]
            a_empty = len(A.items) == 1
            cost_a = p.types[A.type_idx].cost
            # 剪枝：源车一侧没有任何节省空间时，迁移不可能降费
            if a_empty:
                save_a = cost_a
            else:
                tA = _cheapest_type(p, A.w - it.w, A.v - it.v, None)
                if tA < 0:
                    continue
                save_a = cost_a - p.types[tA].cost
            if save_a <= _COST_TIE:
                continue
            for bi in range(len(sol.vehicles)):
                if bi == ai:
                    continue
                B = sol.vehicles[bi]
                if B.sealed:
                    continue
                if it.cat is not None and any(_conflict(p, it.cat, c) for c in B.cats):
                    continue
                avail = remaining[:]
                avail[A.type_idx] += 1
                avail[B.type_idx] += 1
                res = _best_pair_types(
                    p, avail, A.w - it.w, A.v - it.v, a_empty, B.w + it.w, B.v + it.v
                )
                if res is None:
                    continue
                new_pair, tA, tB = res
                delta = new_pair - (cost_a + p.types[B.type_idx].cost)
                if delta < -_EPS:
                    _remove_item(p, A, i)
                    _add_item(p, B, i)
                    B.type_idx = tB
                    if tA is None:
                        del sol.vehicles[ai]
                    else:
                        A.type_idx = tA
                    sol.cost += delta
                    return True
    return False


def _local_search(sol: Solution, stopped) -> None:
    for _ in range(_MAX_LOCAL_PASSES):
        if stopped():
            return
        if _retype_pass(sol):
            continue
        if _eliminate_pass(sol, stopped):
            continue
        if _relocate_pass(sol, stopped):
            continue
        return


# ---------------------------------------------------------------- 扰动

def _perturb(sol: Solution, rng: random.Random) -> bool:
    """随机搬动几票货（可能开新车）。无车可开时返回 False。"""
    p = sol.p
    movable = [
        (vi, i)
        for vi, veh in enumerate(sol.vehicles)
        if not veh.sealed
        for i in veh.items
    ]
    if len(movable) < 2:
        return False
    k = rng.randint(2, max(2, min(10, len(movable) // 2)))
    chosen = rng.sample(movable, k)
    removed: list[int] = []
    for vi, i in chosen:
        _remove_item(p, sol.vehicles[vi], i)
        removed.append(i)
    kept = []
    for veh in sol.vehicles:
        if veh.items:
            kept.append(veh)
        else:
            sol.cost -= p.types[veh.type_idx].cost
    sol.vehicles = kept
    usage = sol.usage()
    avail = [p.types[t].count - usage[t] for t in range(len(p.types))]
    rng.shuffle(removed)
    for i in removed:
        best_vi, best_key = -1, None
        for vi, veh in enumerate(sol.vehicles):
            if _can_add(p, veh, i):
                t = p.types[veh.type_idx]
                key = (
                    t.capw - (veh.w + p.items[i].w),
                    t.capv - (veh.v + p.items[i].v),
                    vi,
                )
                if best_key is None or key < best_key:
                    best_key, best_vi = key, vi
        if best_vi >= 0:
            _add_item(p, sol.vehicles[best_vi], i)
        else:
            it = p.items[i]
            t = _cheapest_type(p, it.w, it.v, avail)
            if t < 0:
                return False
            avail[t] -= 1
            veh = Vehicle(t, [], 0, 0, {}, False)
            _add_item(p, veh, i)
            sol.vehicles.append(veh)
            sol.cost += p.types[t].cost
    return True


# ---------------------------------------------------------------- 主流程

def _ils_budget(n: int) -> int:
    """ILS 轮数预算：只随票数变化，保证确定性。"""
    return max(40, min(400, 30 + n))


@dataclass
class SolveResult:
    solution: Solution | None
    stop_reason: str  # "optimal" | "exhausted" | "time_limit" | "cancelled"
    error: str | None = None


def solve(
    p: Problem,
    lb_cost: float,
    seed: int,
    time_limit: float,
    should_cancel,
    on_improvement,
) -> SolveResult:
    """求解。on_improvement 在每次刷新历史最好解时回调（解始终完整可行）。"""
    rng = random.Random(seed)
    deadline = time.monotonic() + max(0.0, time_limit)

    def stopped() -> bool:
        return should_cancel() or time.monotonic() >= deadline

    try:
        sol = _construct(p)
    except InfeasibleError as e:
        return SolveResult(None, "failed", str(e))
    _local_search(sol, stopped)
    best = sol
    on_improvement(best)

    stop: str | None = None
    if best.cost <= lb_cost + _tol(lb_cost):
        stop = "optimal"

    n = len(p.items)
    budget = _ils_budget(n)
    current = best.clone()
    since_improve = 0
    it_no = 0
    while stop is None and it_no < budget:
        if should_cancel():
            stop = "cancelled"
            break
        if time.monotonic() >= deadline:
            stop = "time_limit"
            break
        it_no += 1
        cand = (best if since_improve > _RESTART_AFTER else current).clone()
        if not _perturb(cand, rng):
            current = best.clone()
            since_improve += 1
            continue
        _local_search(cand, stopped)
        if cand.cost < best.cost - _EPS:
            best = cand
            on_improvement(best)
            since_improve = 0
            current = cand.clone()
            if best.cost <= lb_cost + _tol(lb_cost):
                stop = "optimal"
                break
        else:
            since_improve += 1
            current = cand if cand.cost <= current.cost + _EPS else best.clone()

    if stop is None:
        stop = "exhausted"
    if stop != "optimal" and should_cancel():
        stop = "cancelled"
    return SolveResult(best, stop)


def snapshot_solution(p: Problem, sol: Solution, scale_w: int, scale_v: int) -> dict:
    """导出为可 JSON 序列化的方案（还原原始单位）。"""
    vehicles = []
    for veh in sol.vehicles:
        t = p.types[veh.type_idx]
        vehicles.append(
            {
                "vehicle_type": t.name,
                "items": [p.items[i].id for i in veh.items],
                "load_weight": veh.w / scale_w,
                "load_volume": veh.v / scale_v,
            }
        )
    return {
        "vehicle_count": len(vehicles),
        "total_cost": round(sol.cost, 9),
        "vehicles": vehicles,
    }
