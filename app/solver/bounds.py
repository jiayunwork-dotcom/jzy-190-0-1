"""下界：从重量、体积、独占货三个角度推导，另加一个由车辆数派生的费用界，取最紧。

- 重量角度：每辆车最多装 max_capw，故车辆数 >= ceil(W / max_capw)；
  每吨运费至少 min_t(cost_t / capw_t)，故总费用 >= W * min_t(cost_t / capw_t)。
- 体积角度：同理。
- 独占角度：每票独占货必须单独一车，车辆数 >= 独占票数；
  每票独占货至少花"能装下它的最便宜车型"的费用（忽略车型数量限制，仍是有效下界）。
- 派生费用界：车辆数下界 * 最低单车费用。

全部用 Fraction 精确计算，保证：
- 下界对同一实例单调（多加一票货，W/V/独占数都不减，各界不减，max 不减）；
- 重量与载重同步缩放时 ceil 结果不变（整数精确运算）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .types import Problem


@dataclass
class LowerBound:
    cost: float  # 总费用下界
    vehicles: int  # 车辆数下界
    tightest: str  # 起作用的界: "weight" | "volume" | "exclusive" | "vehicle_count" | "trivial"
    details: dict = field(default_factory=dict)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute_lower_bound(p: Problem) -> LowerBound:
    if not p.items:
        return LowerBound(
            cost=0.0,
            vehicles=0,
            tightest="trivial",
            details={"note": "empty problem"},
        )

    total_w = sum(it.w for it in p.items)
    total_v = sum(it.v for it in p.items)
    n_excl = sum(1 for it in p.items if it.exclusive)

    max_capw = max(t.capw for t in p.types)
    max_capv = max(t.capv for t in p.types)
    cnt_w = _ceil_div(total_w, max_capw)
    cnt_v = _ceil_div(total_v, max_capv)
    vehicles = max(cnt_w, cnt_v, n_excl)

    # 费用界（Fraction 精确）
    min_ratio_w = min(Fraction(str(t.cost)) / t.capw for t in p.types)
    min_ratio_v = min(Fraction(str(t.cost)) / t.capv for t in p.types)
    cost_w = total_w * min_ratio_w
    cost_v = total_v * min_ratio_v
    cost_excl = Fraction(0)
    for it in p.items:
        if it.exclusive:
            cost_excl += min(
                Fraction(str(t.cost)) for t in p.types if t.capw >= it.w and t.capv >= it.v
            )
    min_cost = min(Fraction(str(t.cost)) for t in p.types)
    cost_count = vehicles * min_cost

    # 取最紧；并列时按固定顺序取先者，保证确定性
    candidates = [
        ("weight", cost_w),
        ("volume", cost_v),
        ("exclusive", cost_excl),
        ("vehicle_count", cost_count),
    ]
    tightest, best = candidates[0]
    for name, val in candidates[1:]:
        if val > best:
            tightest, best = name, val

    details = {
        "total_weight": total_w,
        "total_volume": total_v,
        "exclusive_items": n_excl,
        "count_by_weight": cnt_w,
        "count_by_volume": cnt_v,
        "count_by_exclusive": n_excl,
        "cost_by_weight": float(cost_w),
        "cost_by_volume": float(cost_v),
        "cost_by_exclusive": float(cost_excl),
        "cost_by_vehicle_count": float(cost_count),
    }
    return LowerBound(cost=float(best), vehicles=vehicles, tightest=tightest, details=details)
