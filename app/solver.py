"""排车求解器：多策略构造 + 局部搜索 + 迭代局部搜索（ILS）。

方法选择（详见 docs/算法说明.md）：
  问题本质是异构车型的二维装箱（2D-VRP 的装载子问题）+ 类别互斥 + 独占约束，
  属 NP-hard。这里不做精确搜索，而采用：
    1) 独占货先用最小费用流做最优车型指派（可证明最优的子结构）；
    2) 非独占货用多种确定性构造策略（重量/体积/密度降序 × 最佳/最省适配）；
    3) 局部搜索：全局车型再优化（最小费用流，可证明该分组下最优）、
       整车合并（含交换助攻）、搬迁后降档；
    4) ILS 随机扰动后重复局部搜索，保留全局最好解。
  全程只在可行解之间移动；任何一轮结束时的最好解都完整可行。
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .bounds import LowerBound, compute_lower_bound
from .mincostflow import MinCostFlow
from .models import EPS, Problem, Shipment, Solution, Vehicle, VehicleType
from .verify import is_feasible, verify

ProgressCallback = Callable[[Solution, float, int], None]
CancelCheck = Callable[[], bool]


# ---------------------------------------------------------------------------
# 确定性的工作量标定（进程内一次性测量并缓存）
#
# 为了同时满足"按时间上限持续改进"与"同输入同种子逐位一致"，停止规则不能直接看
# 墙钟（机器负载会让停在第几轮随机）。改为：用规模估计每轮的"工作量单位"，
# 用进程级标定常数把秒换算成单位，确定地算出轮数。常数在进程生命周期内恒定，
# 故同进程的求解严格可复现；跨机器的墙钟会有差异，但结果仍由种子确定。
# ---------------------------------------------------------------------------

_SECONDS_PER_UNIT: Optional[float] = None
_CAL_LOCK = threading.Lock()


def _round_units(problem: Problem) -> int:
    """一个 ILS 轮（一次扰动 + 一轮局部搜索）的工作量估计。

    构造后的车辆数与票数同量级；合并/搬迁均为 O(票数^2 * 车型数) 的平方级邻域，
    故用 n^2 * K 的量级刻画，常数项保证小实例不为零。
    """
    n = max(1, len(problem.shipments))
    k = max(1, len(problem.vehicle_types))
    # 平方邻域（合并车对、搬迁的货*车*车扫描）随 n^2*k 增长；实测搬迁扫描再陡一些
    # （约 n^2.3），用 n^0.4 修正项使大实例的耗时估计不偏乐观。
    return int(4000 + 40 * n + (n ** 2) * k * (n ** 0.4))


def _calibrate_seconds_per_unit() -> float:
    """在合成小问题上实测单位工作量耗时（秒）；进程内只测一次。"""
    global _SECONDS_PER_UNIT
    if _SECONDS_PER_UNIT is not None:
        return _SECONDS_PER_UNIT
    with _CAL_LOCK:
        if _SECONDS_PER_UNIT is None:
            from .schemas import JobRequest as _JR  # 避免模块顶层循环依赖
            from .serialize import build_problem as _build

            payload = {
                "shipments": [
                    {"id": f"c{i}", "weight": 3 + (i % 5), "volume": 2 + (i % 4)}
                    for i in range(60)
                ],
                "vehicle_types": [
                    {"id": "S", "max_weight": 10, "max_volume": 12, "cost": 100, "available": 60},
                    {"id": "L", "max_weight": 20, "max_volume": 24, "cost": 170, "available": 30},
                ],
                "conflicts": [],
            }
            prob = _build(_JR.model_validate(payload))
            lb = compute_lower_bound(prob)
            starter = construct(
                prob, VARIANTS[0], lb.exclusive_assignment.assignment
            )
            t0 = time.monotonic()
            local_search(prob, starter, max_rounds=1)
            dt = max(1e-6, time.monotonic() - t0)
            units = _round_units(prob)
            _SECONDS_PER_UNIT = dt / units
    return _SECONDS_PER_UNIT


@dataclass
class SolveResult:
    status: str  # optimal / timeout / cancelled / interrupted / infeasible
    feasible: bool
    reason: Optional[str]
    cost: float
    vehicle_count: int
    solution: Optional[Solution]
    lower_bound: LowerBound
    iterations: int
    elapsed_seconds: float

    @property
    def relative_gap(self) -> Optional[float]:
        if self.solution is None:
            return None
        return relative_gap(self.cost, self.lower_bound.cost)


def relative_gap(cost: float, lb_cost: float) -> float:
    """当前方案相对下界的差距 (cost-lb)/max(lb, 1e-12)。"""
    if lb_cost <= 1e-12:
        return 0.0 if cost <= 1e-12 else 1.0
    return max(0.0, (cost - lb_cost) / lb_cost)


# ---------------------------------------------------------------------------
# 给定货物分组的全局最优车型选择（最小费用流）
# ---------------------------------------------------------------------------

def optimize_types(problem: Problem, solution: Solution) -> bool:
    """在不改变货物分组的前提下，为每辆车选费用最小的可行车型（含可用量约束）。

    原地修改；分组配不出可行车型时返回 False。当前分组可能处于"名义车型超载"的
    临时状态（合并已发生、尚未升级），故无条件采用求出来的可行指派。
    """
    vehicles = solution.vehicles
    if not vehicles:
        return False
    nominal_cost = solution.cost()
    new_cost = solve_assignment(problem, vehicles, apply=True)
    if new_cost is None:
        return False
    return solution.cost() < nominal_cost - EPS


def _type_counts_ok_to_open(problem: Problem, solution: Solution, type_id: str) -> bool:
    counts = solution.type_counts()
    vt = problem.vehicle_type(type_id)
    return counts[type_id] < vt.available


def _cheapest_type_for(
    problem: Problem, ship: Shipment, solution: Solution, forbidden: Optional[set[str]] = None
) -> Optional[str]:
    """能装下 ship 且还能再开一辆的最便宜车型 id。"""
    candidates = []
    for vt in problem.vehicle_types:
        if forbidden and vt.id in forbidden:
            continue
        if (
            ship.weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
            and ship.volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
            and _type_counts_ok_to_open(problem, solution, vt.id)
        ):
            candidates.append((vt.cost, vt.id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _is_exclusive_car(problem: Problem, v: Vehicle) -> bool:
    if len(v.shipment_ids) != 1:
        return False
    return problem.shipment(v.shipment_ids[0]).exclusive


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

@dataclass
class ConstructConfig:
    name: str
    order: str  # weight / volume / density / normdim
    placement: str  # bestfit / cheapfit
    randomness: float = 0.0  # >0 时启用 GRASP：在前若干个候选中按概率选


def _sort_nonexclusive(
    ships: list[Shipment], order: str
) -> list[Shipment]:
    def key(s: Shipment):
        vol = s.volume if s.volume > EPS else 1e-9
        if order == "weight":
            return (-s.weight, -s.volume, s.index)
        if order == "volume":
            return (-s.volume, -s.weight, s.index)
        # density：重泡比降序（泡货轻、泡货比重大时排序靠后）。
        return (-s.weight / vol, -s.weight, s.index)

    return sorted(ships, key=key)


def _normdim_order(problem: Problem, ships: list[Shipment]) -> list[Shipment]:
    """按"相对最小可行车型的两维占用比"降序。映射预计算，不修改 frozen 对象。"""
    norm: dict[str, float] = {}
    for s in ships:
        fits = [
            vt
            for vt in problem.vehicle_types
            if s.weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
            and s.volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
        ]
        if not fits:
            norm[s.id] = 1e9
            continue
        vt = min(fits, key=lambda t: (max(t.max_weight, t.max_volume), t.id))
        norm[s.id] = max(
            s.weight / max(vt.max_weight, 1e-9),
            s.volume / max(vt.max_volume, 1e-9),
        )
    return sorted(ships, key=lambda s: (-norm[s.id], s.index))


def _place_score(
    problem: Problem, v: Vehicle, vt: VehicleType, ship: Shipment, placement: str
) -> tuple:
    if placement == "bestfit":
        # 装完后剩余空间比例越小越好（重量、体积两个维度的最大剩余）。
        slack_w = (vt.max_weight - v.weight - ship.weight) / max(vt.max_weight, 1e-9)
        slack_v = (vt.max_volume - v.volume - ship.volume) / max(vt.max_volume, 1e-9)
        # 主排序：最紧的维度剩余最小；次排序：车型便宜
        return (max(slack_w, slack_v), vt.cost, v.type_id, len(v.shipment_ids))
    # cheapfit：已有的车边际费用都是 0，按填充紧度打破平局
    slack_w = (vt.max_weight - v.weight - ship.weight) / max(vt.max_weight, 1e-9)
    slack_v = (vt.max_volume - v.volume - ship.volume) / max(vt.max_volume, 1e-9)
    return (0.0, max(slack_w, slack_v), vt.cost, v.type_id)


def construct(
    problem: Problem,
    cfg: ConstructConfig,
    exclusive_assignment: dict[str, str],
    rng: Optional[random.Random] = None,
) -> Optional[Solution]:
    """按给定策略构造一个完整可行方案；失败返回 None。"""
    sol = Solution(vehicles=[], problem=problem)

    # 1) 独占货：各开一辆指定车型的车。
    for ship in problem.shipments:
        if not ship.exclusive:
            continue
        type_id = exclusive_assignment.get(ship.id)
        if type_id is None:
            type_id = _cheapest_type_for(problem, ship, sol)
        if type_id is None or not _type_counts_ok_to_open(problem, sol, type_id):
            return None
        vt = problem.vehicle_type(type_id)
        if (
            ship.weight > vt.max_weight + EPS * max(1.0, vt.max_weight)
            or ship.volume > vt.max_volume + EPS * max(1.0, vt.max_volume)
        ):
            return None
        v = Vehicle(type_id=type_id)
        sol.add_to(v, ship)
        sol.vehicles.append(v)

    # 2) 非独占货排序
    others = [s for s in problem.shipments if not s.exclusive]
    if cfg.order == "normdim":
        ordered = _normdim_order(problem, others)
    else:
        ordered = _sort_nonexclusive(others, cfg.order)

    # 3) 逐票放置
    for ship in ordered:
        candidates: list[tuple] = []
        for v in sol.vehicles:
            if _is_exclusive_car(problem, v):
                continue
            vt = problem.vehicle_type(v.type_id)
            if v.can_add(ship, problem, vt):
                candidates.append((_place_score(problem, v, vt, ship, cfg.placement), v))
        new_type = _cheapest_type_for(problem, ship, sol)

        chosen: Optional[Vehicle] = None
        if candidates:
            candidates.sort(key=lambda c: c[0])
            if rng is not None and cfg.randomness > 0.0:
                top_k = min(len(candidates), 3)
                pick = rng.randrange(top_k) if rng.random() < cfg.randomness else 0
                chosen = candidates[pick][1]
            else:
                chosen = candidates[0][1]

        if chosen is None:
            if new_type is None:
                return None
            chosen = Vehicle(type_id=new_type)
            sol.vehicles.append(chosen)
        sol.add_to(chosen, ship)

    return sol


VARIANTS = [
    ConstructConfig("重量降序-最佳适配", "weight", "bestfit"),
    ConstructConfig("体积降序-最佳适配", "volume", "bestfit"),
    ConstructConfig("密度降序-最佳适配", "density", "bestfit"),
    ConstructConfig("归一尺寸降序-最佳适配", "normdim", "bestfit"),
    ConstructConfig("重量降序-最省适配", "weight", "cheapfit"),
    ConstructConfig("体积降序-最省适配", "volume", "cheapfit"),
]


# ---------------------------------------------------------------------------
# 局部搜索
#
# 关键设计：货物分组与车型选择解耦。
#   - 分组层面（哪些货同一辆车）只检查类别互斥与独占——这些与车型无关；
#   - 容量是否装得下、用哪种车型最省、车型是否够用，全部交给最小费用流一次裁决，
#     即"给定分组的全局最优车型选择"。这样允许把两车货并入一辆后升级到大车型，
#     只要新分组的最优车型总费用更低就接受。
# ---------------------------------------------------------------------------

def _groups_compatible(problem: Problem, a: Vehicle, b: Vehicle) -> bool:
    """两组货能否同车（只看类别互斥与独占，与车型无关）。"""
    if _is_exclusive_car(problem, a) or _is_exclusive_car(problem, b):
        return False
    for cat in a._cat_counts:
        if b._cat_counts.keys() & problem.conflict_adj.get(cat, set()):
            return False
    return True


def _group_compatible_add(problem: Problem, v: Vehicle, ship: Shipment) -> bool:
    """把一票货并入 v 的分组是否类别相容（独占车/独占货另行禁止）。"""
    if _is_exclusive_car(problem, v) or ship.exclusive:
        return False
    if ship.category is None:
        return True
    return not (v._cat_counts.keys() & problem.conflict_adj.get(ship.category, set()))


def _category_ok_without(
    problem: Problem, ships: list[Shipment], remove: Shipment, add: Shipment
) -> bool:
    """ships 去掉 remove、加入 add 后，类别两两不冲突（自冲突也检查）。"""
    cats: dict[str, int] = {}
    for s in ships:
        if s is remove:
            continue
        if s.category is not None:
            cats[s.category] = cats.get(s.category, 0) + 1
    if add.category is not None:
        cats[add.category] = cats.get(add.category, 0) + 1
    for cat, n in cats.items():
        if n > 1 and cat in problem.conflict_adj.get(cat, set()):
            return False  # 自冲突
        for other, m in cats.items():
            if other >= cat:
                continue
            if other in problem.conflict_adj.get(cat, set()):
                return False
    return True


def _swap_pair_category_ok(
    problem: Problem,
    ship_b: Shipment,
    ship_a: Shipment,
    a_ships: list[Shipment],
    b_ships: list[Shipment],
) -> bool:
    return _category_ok_without(problem, a_ships, ship_a, ship_b) and _category_ok_without(
        problem, b_ships, ship_b, ship_a
    )


def _feasible_type_costs(
    problem: Problem, weight: float, volume: float
) -> list[tuple[float, str]]:
    """能装下该总重/总体积的 (费用, 车型) 列表，按费用、车型 id 排序。"""
    out = []
    for vt in problem.vehicle_types:
        if (
            weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
            and volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
        ):
            out.append((vt.cost, vt.id))
    out.sort()
    return out


def solve_assignment(
    problem: Problem, vehicles: list[Vehicle], apply: bool = False
) -> Optional[float]:
    """给定分组求全局最优车型总费用（车型可用量受限）。

    快路径：每辆车独立取最便宜可行车型；若该指派满足全部可用上限，
    它既是一个可行方案又等于逐组费用下界之和，故可证明最优，O(V*K)。
    若触碰可用上限（含并列费用需换型的情形），回退到最小费用流求精确解。
    apply=True 时把最优指派写回各车 type_id。
    """
    if not vehicles:
        return 0.0
    types = problem.vehicle_types
    options: list[list[tuple[float, str]]] = []
    counts = {vt.id: 0 for vt in types}
    total = 0.0
    for v in vehicles:
        opts = _feasible_type_costs(problem, v.weight, v.volume)
        if not opts:
            return None
        options.append(opts)
        total += opts[0][0]
        counts[opts[0][1]] += 1
    greedy_ok = all(counts[vt.id] <= vt.available for vt in types)
    if greedy_ok:
        if apply:
            for v, opts in zip(vehicles, options):
                v.type_id = opts[0][1]
        return total
    return _assignment_mcmf(problem, vehicles, apply=apply)


def _assignment_mcmf(
    problem: Problem, vehicles: list[Vehicle], apply: bool
) -> Optional[float]:
    types = problem.vehicle_types
    V, K = len(vehicles), len(types)
    type_base = 1 + V
    sink = 1 + V + K
    mcf = MinCostFlow(sink + 1)
    for i in range(V):
        mcf.add_edge(0, 1 + i, 1, 0.0)
    edges: list[tuple[int, int, str]] = []
    for i, v in enumerate(vehicles):
        for j, vt in enumerate(types):
            if (
                v.weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
                and v.volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
            ):
                idx = mcf.add_edge(1 + i, type_base + j, 1, float(vt.cost))
                edges.append((idx, i, vt.id))
    for j, vt in enumerate(types):
        mcf.add_edge(type_base + j, sink, vt.available, 0.0)
    res = mcf.solve(0, sink, V)
    if not res.feasible:
        return None
    if apply:
        for edge_idx, vi, type_id in edges:
            if res.flows[edge_idx] > 0:
                vehicles[vi].type_id = type_id
    return res.total_cost


# 兼容旧名称：只评估、不写回。
def assignment_cost(problem: Problem, vehicles: list[Vehicle]) -> Optional[float]:
    return solve_assignment(problem, vehicles, apply=False)


def _cheapest_group_cost(problem: Problem, weight: float, volume: float) -> Optional[float]:
    """忽略可用量时，一组给定总重/总体积的货能用的最便宜车型费用；装不下返回 None。"""
    best: Optional[float] = None
    for vt in problem.vehicle_types:
        if (
            weight <= vt.max_weight + EPS * max(1.0, vt.max_weight)
            and volume <= vt.max_volume + EPS * max(1.0, vt.max_volume)
        ):
            best = vt.cost if best is None else min(best, vt.cost)
    return best


def _merge_pass(
    problem: Problem,
    sol: Solution,
    should_abort: Optional[Callable[[], bool]] = None,
    budget: int = 10_000_000,
) -> bool:
    """单趟贪心整车合并（允许升级车型），O(V^2 * K)。

    维持当前分组的"每组最便宜可行车型 + 车型用量计数"。合并 a、b 时增量更新，
    若合并组能塞进某车型、且其在用量允许下可选的最便宜车型费用不超过两车原费用之和
    （等费用平台也接受，为后续三车并一车铺路），则接受合并；否则跳过。
    类别互斥与独占约束在分组层面保证，费用永不升高。过程不看墙钟，结果确定。
    """
    types = problem.vehicle_types
    capacity = {vt.id: vt.available for vt in types}
    cost_of = {vt.id: vt.cost for vt in types}

    # 初始指派沿用 optimize_types 落定的真实（满足可用上限的）车型。
    groups = [v for v in sol.vehicles]
    chosen: dict[int, str] = {}
    used: dict[str, int] = {vt.id: 0 for vt in types}
    for gi, v in enumerate(groups):
        opts = _feasible_type_costs(problem, v.weight, v.volume)
        if not opts:
            return False
        if v.type_id not in {tid for _, tid in opts}:
            # 当前名义车型其实装不下（临时状态），先取一个可行车型再整体再优化。
            v.type_id = opts[0][1]
        chosen[gi] = v.type_id
        used[v.type_id] += 1

    improved = False
    tried = 0
    i = 0
    while i < len(groups):
        if tried >= budget or (should_abort and should_abort()):
            break
        a = groups[i]
        j = i + 1
        while j < len(groups):
            if tried >= budget or (should_abort and should_abort()):
                break
            b = groups[j]
            tried += 1
            if _groups_compatible(problem, a, b):
                u_opts = _feasible_type_costs(
                    problem, a.weight + b.weight, a.volume + b.volume
                )
                if u_opts:
                    # 临时释放 a、b 的车型占用，给合并组选最便宜且不超额的车型。
                    used[chosen[i]] -= 1
                    used[chosen[j]] -= 1
                    picked: Optional[str] = None
                    for c, tid in u_opts:
                        if used[tid] < capacity[tid]:
                            picked = tid
                            break
                    if picked is not None:
                        new_c = cost_of[picked]
                        old_c = cost_of[chosen[i]] + cost_of[chosen[j]]
                        if new_c <= old_c + EPS:
                            # 接受合并：物理合并到 a，删除 b。
                            for sid in b.shipment_ids:
                                sol.add_to(a, problem.shipment(sid))
                            if b in sol.vehicles:
                                sol.vehicles.remove(b)
                            a.type_id = picked
                            used[picked] += 1
                            chosen[i] = picked
                            groups.pop(j)
                            improved = True
                            continue
                    used[chosen[i]] += 1
                    used[chosen[j]] += 1
            j += 1
        i += 1
    if improved:
        optimize_types(problem, sol)
    return improved


def _swap_assist_merge(problem: Problem, sol: Solution, a: Vehicle, b: Vehicle,
                       baseline: float) -> bool:
    """一票对换后再合并 b 到 a；类别相容即可，容量/费用交 MCMF。失败完整回退。"""
    if len(a.shipment_ids) * len(b.shipment_ids) > 225:
        return False
    a_ships = [problem.shipment(s) for s in a.shipment_ids]
    b_ships = [problem.shipment(s) for s in b.shipment_ids]
    # 用临时分组，避免在 sol 上来回搬易错：构造对换后的 a' 与 b' 再试合并。
    for ship_b in b_ships:
        for ship_a in a_ships:
            # 对换必须在各自组内类别相容（先快速判定，再构造临时分组）。
            if not _swap_pair_category_ok(problem, ship_b, ship_a, a_ships, b_ships):
                continue
            # 对换后 b' 上的货：b 去掉 ship_b 加入 ship_a；a' 反之。
            a2 = Vehicle(type_id=a.type_id)
            b2 = Vehicle(type_id=b.type_id)
            for s in a_ships:
                target = a2 if s is not ship_a else b2
                sol.add_to(target, s)
            for s in b_ships:
                target = b2 if s is not ship_b else a2
                sol.add_to(target, s)
            # 先尝试 b2 并进 a2
            merged = _groups_compatible(problem, a2, b2)
            trial: list[Vehicle]
            if merged:
                m = Vehicle(type_id=a2.type_id)
                for sid in a2.shipment_ids + b2.shipment_ids:
                    sol.add_to(m, problem.shipment(sid))
                trial = [m] + [v for v in sol.vehicles if v is not a and v is not b]
            else:
                trial = [a2, b2] + [v for v in sol.vehicles if v is not a and v is not b]
            new_cost = solve_assignment(problem, trial, apply=True)
            # 合并掉一辆车时接受等费用平台移动；单纯对换则必须严格降费，避免空转。
            accept = new_cost is not None and (
                new_cost + EPS < baseline or (merged and new_cost <= baseline + EPS)
            )
            if accept:
                # 接受：用已写回车型的 trial 分组替换 sol。
                sol.vehicles = trial
                return True
    return False


def _swap_merge_pass(
    problem: Problem,
    sol: Solution,
    should_abort: Optional[Callable[[], bool]] = None,
    budget: int = 200,
) -> bool:
    """交换助攻合并，带尝试预算。过程确定，不看墙钟。"""
    improved = False
    tried = 0
    non_excl = [v for v in sol.vehicles if not _is_exclusive_car(problem, v)]
    for i in range(len(non_excl)):
        if tried >= budget:
            break
        if should_abort and should_abort():
            break
        for j in range(i + 1, len(non_excl)):
            if tried >= budget:
                break
            if should_abort and should_abort():
                break
            a, b = non_excl[i], non_excl[j]
            if a not in sol.vehicles or b not in sol.vehicles:
                continue
            if _groups_compatible(problem, a, b):
                continue  # 直接合并不需要交换助攻
            tried += 1
            baseline = assignment_cost(problem, sol.vehicles)
            if baseline is None:
                continue
            if _swap_assist_merge(problem, sol, a, b, baseline):
                improved = True
                non_excl = [v for v in sol.vehicles if not _is_exclusive_car(problem, v)]
                break
    optimize_types(problem, sol)
    return improved


def _relocate_pass(
    problem: Problem,
    sol: Solution,
    should_abort: Optional[Callable[[], bool]] = None,
    budget: int = 200000,
) -> bool:
    """单趟增量搬迁，目标是把一辆车清空（少一辆车）或降低总费用。

    维持每组当前车型与车型用量计数，单次移动只增量重选受影响的两辆车，
    摊还复杂度近似 O(总票数 * 车数 * K)。源车清空时接受总费用不升的移动，
    其余情形要求严格降费。任何移动都保持类别互斥与容量可行。过程确定，不看墙钟。
    """
    types = problem.vehicle_types
    capacity = {vt.id: vt.available for vt in types}
    cost_of = {vt.id: vt.cost for vt in types}

    def snapshot():
        groups = [v for v in sol.vehicles]
        chosen: dict[int, str] = {}
        used: dict[str, int] = {vt.id: 0 for vt in types}
        for gi, v in enumerate(groups):
            opts = _feasible_type_costs(problem, v.weight, v.volume)
            if not opts:
                return None
            if v.type_id not in {tid for _, tid in opts}:
                v.type_id = opts[0][1]
            chosen[gi] = v.type_id
            used[v.type_id] += 1
        return groups, chosen, used

    snap = snapshot()
    if snap is None:
        return False
    improved = False
    tried = 0
    fi = 0
    while fi < len(snap[0]):
        groups, chosen, used = snap
        if tried >= budget or (should_abort and should_abort()):
            break
        v_from = groups[fi]
        if _is_exclusive_car(problem, v_from) or v_from not in sol.vehicles:
            fi += 1
            continue
        moved = False
        for sid in list(v_from.shipment_ids):
            ship = problem.shipment(sid)
            fw, fv = v_from.weight - ship.weight, v_from.volume - ship.volume
            empties = fw <= EPS and fv <= EPS
            f_opts = [] if empties else _feasible_type_costs(problem, fw, fv)
            if not empties and not f_opts:
                continue
            for tj in range(len(groups)):
                if tj == fi:
                    continue
                v_to = groups[tj]
                if not _group_compatible_add(problem, v_to, ship):
                    continue
                t_opts = _feasible_type_costs(
                    problem, v_to.weight + ship.weight, v_to.volume + ship.volume
                )
                if not t_opts:
                    continue
                tried += 1
                old_f, old_t = chosen[fi], chosen[tj]
                used[old_f] -= 1
                used[old_t] -= 1
                new_t = next((tid for _, tid in t_opts if used[tid] < capacity[tid]), None)
                new_f: Optional[str] = None
                if new_t is not None:
                    if empties:
                        new_f = ""
                    else:
                        used[new_t] += 1
                        new_f = next(
                            (tid for _, tid in f_opts if used[tid] < capacity[tid]), None
                        )
                        used[new_t] -= 1
                ok = False
                if new_t is not None and new_f is not None:
                    old_c = cost_of[old_f] + cost_of[old_t]
                    new_c = cost_of[new_t] + (0.0 if new_f == "" else cost_of[new_f])
                    ok = new_c + EPS < old_c or (empties and new_c <= old_c + EPS)
                if ok:
                    sol.remove_from(v_from, ship)
                    sol.add_to(v_to, ship)
                    used[new_t] += 1
                    v_to.type_id = new_t
                    if empties:
                        if v_from in sol.vehicles:
                            sol.vehicles.remove(v_from)
                    else:
                        used[new_f] += 1  # type: ignore[arg-type]
                        v_from.type_id = new_f  # type: ignore[assignment]
                    improved = moved = True
                    break
                used[old_f] += 1
                used[old_t] += 1
            if moved:
                break
        if moved:
            # 车辆集合或车型用量可能已变（源车消失），重建快照后从头再扫。
            snap = snapshot()
            if snap is None:
                break
            fi = 0
        else:
            fi += 1
    if improved:
        optimize_types(problem, sol)
    return improved




def local_search(
    problem: Problem,
    sol: Solution,
    should_abort: Optional[Callable[[], bool]] = None,
    max_rounds: int = 3,
) -> Solution:
    """确定性局部搜索：循环 车型再优化 -> 合并 -> 交换助攻 -> 搬迁，至多 max_rounds 轮。

    不看墙钟、不依赖机器速度——给定输入与初始方案，结果完全确定；
    墙钟由 solve() 在轮次之间使用（用首轮计时标定总轮数）。
    """
    optimize_types(problem, sol)
    for _ in range(max_rounds):
        if should_abort and should_abort():
            break
        improved = optimize_types(problem, sol)
        if _merge_pass(problem, sol, should_abort):
            improved = True
        if not (should_abort and should_abort()) and _swap_merge_pass(problem, sol, should_abort):
            improved = True
        if not (should_abort and should_abort()) and _relocate_pass(problem, sol, should_abort):
            improved = True
        if not improved:
            break
    optimize_types(problem, sol)
    return sol


# ---------------------------------------------------------------------------
# ILS 扰动
# ---------------------------------------------------------------------------

def perturb(problem: Problem, sol: Solution, rng: random.Random, strength: int) -> Solution:
    """随机把若干票非独占货搬到别的车（或新车），保持方案可行。"""
    cand = [v for v in sol.vehicles if not _is_exclusive_car(problem, v) and v.shipment_ids]
    if not cand:
        return sol
    work = sol.clone()
    work_vehicles = list(work.vehicles)
    movable = [
        (v, sid)
        for v in work_vehicles
        if not _is_exclusive_car(problem, v)
        for sid in v.shipment_ids
    ]
    rng.shuffle(movable)
    kicks = min(strength, len(movable))
    for v_from, sid in movable[:kicks]:
        ship = problem.shipment(sid)
        targets = [v for v in work.vehicles if v is not v_from and not _is_exclusive_car(problem, v)]
        rng.shuffle(targets)
        placed = False
        for v_to in targets:
            vt = problem.vehicle_type(v_to.type_id)
            if v_to.can_add(ship, problem, vt):
                work.remove_from(v_from, ship)
                work.add_to(v_to, ship)
                placed = True
                break
        if not placed:
            new_type = _cheapest_type_for(problem, ship, work)
            if new_type is not None:
                work.remove_from(v_from, ship)
                v_new = Vehicle(type_id=new_type)
                work.vehicles.append(v_new)
                work.add_to(v_new, ship)
        if not v_from.shipment_ids and v_from in work.vehicles:
            work.vehicles.remove(v_from)
    return work


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def solve(
    problem: Problem,
    seed: int,
    time_limit: float,
    progress_cb: Optional[ProgressCallback] = None,
    is_cancelled: Optional[CancelCheck] = None,
) -> SolveResult:
    """运行 anytime 求解。返回时方案保证完整可行（infeasible 除外）。"""
    start = time.monotonic()
    deadline = start + max(0.01, time_limit)
    iterations = 0
    rng = random.Random(seed)

    lb = compute_lower_bound(problem)
    if not lb.feasible:
        return SolveResult(
            status="infeasible",
            feasible=False,
            reason=lb.reason,
            cost=0.0,
            vehicle_count=0,
            solution=None,
            lower_bound=lb,
            iterations=0,
            elapsed_seconds=0.0,
        )

    if not problem.shipments:
        empty = Solution(vehicles=[], problem=problem)
        return SolveResult(
            status="optimal", feasible=True, reason=None, cost=0.0, vehicle_count=0,
            solution=empty, lower_bound=lb, iterations=0, elapsed_seconds=0.0,
        )

    assignment = lb.exclusive_assignment.assignment if lb.exclusive_assignment else {}

    # 1) 多策略构造
    best: Optional[Solution] = None
    best_cost = float("inf")
    for cfg in VARIANTS:
        cand = construct(problem, cfg, assignment)
        if cand is None:
            continue
        verify(problem, cand)
        c = cand.cost()
        if c < best_cost:
            best, best_cost = cand, c
    # GRASP 随机构造补足（最坏情况下帮助找到可行解）。
    grasp_tries = 0
    while best is None and grasp_tries < 200 and time.monotonic() < deadline:
        cfg = ConstructConfig(
            f"grasp-{grasp_tries}",
            rng.choice(["weight", "volume", "density", "normdim"]),
            rng.choice(["bestfit", "cheapfit"]),
            randomness=0.8,
        )
        cand = construct(problem, cfg, assignment, rng=rng)
        grasp_tries += 1
        if cand is None:
            continue
        verify(problem, cand)
        best, best_cost = cand, cand.cost()

    if best is None:
        return _infeasible_result(
            lb,
            reason="construction_failed",
            iterations=grasp_tries,
            elapsed=time.monotonic() - start,
        )

    # 2) 初始局部搜索（确定性，固定 1 轮，得到良好可行解即可；深入改进交给 ILS）。
    local_search(problem, best, is_cancelled, max_rounds=1)
    best_cost = best.cost()
    iterations = 0
    if progress_cb:
        progress_cb(best, best_cost, iterations)

    def proven() -> bool:
        return best_cost <= lb.cost + EPS

    def run_ils_round(cur_best: Solution, idx: int) -> Solution:
        strength = 3 + (idx % 6)
        cand = perturb(problem, cur_best, rng, strength)
        local_search(problem, cand, is_cancelled, max_rounds=1)
        if is_feasible(problem, cand) and cand.cost() < cur_best.cost() - EPS:
            return cand
        return cur_best

    # 3) ILS：停止规则必须确定以保证"同输入同种子逐位一致"，因此不用墙钟决定停在哪。
    # 轮数 = 目标时限 / 单轮估计耗时；单轮耗时 = 每轮尝试预算(_ROUND_UNITS) *
    # 进程内标定的单位时间。标定常数在进程生命周期内固定，故同进程结果严格可复现。
    # 墙钟只作为硬上限兜底（防止标定严重偏小时拖太久），不作为正常停止条件。
    # 轮数 = 目标时限 / 单轮估计耗时；单轮耗时 = 每轮工作量(_round_units) *
    # 进程内标定的单位时间。标定常数在进程生命周期内固定，故同进程结果严格可复现。
    # 循环内不看墙钟（看墙钟会让停在第几轮受机器负载影响），只接受取消信号。
    # 系数 0.6 为保守余量：宁可略早停（仍随 time_limit 线性增加工作量），也不破坏可复现性。
    units = _round_units(problem)
    seconds_per_unit = _calibrate_seconds_per_unit()
    est_round_seconds = max(1e-4, units * seconds_per_unit)
    total_rounds = max(1, int((max(0.01, time_limit) / est_round_seconds) * 0.2))
    total_rounds = min(20000, total_rounds)

    idx = 0
    while not proven() and idx < total_rounds:
        if is_cancelled and is_cancelled():
            break
        idx += 1
        iterations = idx
        new_best = run_ils_round(best, idx)
        if new_best is not best:
            best, best_cost = new_best, new_best.cost()
            if progress_cb:
                progress_cb(best, best_cost, iterations)
        elif idx % 5 == 0 and progress_cb:
            progress_cb(best, best_cost, iterations)

    elapsed = time.monotonic() - start
    verify(problem, best)

    if is_cancelled and is_cancelled():
        status = "cancelled"
    elif proven():
        status = "optimal"
    else:
        status = "timeout"

    return SolveResult(
        status=status,
        feasible=True,
        reason=None,
        cost=best_cost,
        vehicle_count=best.vehicle_count(),
        solution=best,
        lower_bound=lb,
        iterations=iterations,
        elapsed_seconds=elapsed,
    )


def _infeasible_result(lb: LowerBound, reason: str, iterations: int, elapsed: float) -> SolveResult:
    return SolveResult(
        status="infeasible",
        feasible=False,
        reason=reason,
        cost=0.0,
        vehicle_count=0,
        solution=None,
        lower_bound=lb,
        iterations=iterations,
        elapsed_seconds=elapsed,
    )
