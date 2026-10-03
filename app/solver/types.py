"""内部问题表示。

所有重量 / 体积在载入时被放大为整数（10 的幂倍），求解全程使用整数运算：
- 容量比较、求和完全精确，不受浮点舍入影响；
- 全部重量和载重同时乘同一个正数（JSON 能表示的数都是有限小数）时，
  整数表示同步缩放，所有比较结果不变，因此车辆数不变；
- 同样的输入得到同样的内部表示，是确定性的基础。
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Item:
    id: str
    w: int  # 重量，已按 scale_w 放大为整数
    v: int  # 体积，已按 scale_v 放大为整数
    cat: str | None  # 互斥类别，None 表示不参与互斥
    exclusive: bool  # 是否必须独占一车


@dataclass(frozen=True)
class VType:
    name: str
    capw: int
    capv: int
    cost: float
    count: int  # 可用数量上限


@dataclass
class Problem:
    items: list[Item]
    types: list[VType]
    conflict_set: set[frozenset]  # 互相冲突的类别对；单元集表示同类自斥


def _decimals(x: float) -> int:
    """x 的十进制小数位数（JSON 数都是有限小数，str 往返精确）。"""
    exp = Decimal(str(x)).as_tuple().exponent
    return -exp if isinstance(exp, int) and exp < 0 else 0


def _scale_factor(values: list[float]) -> int:
    p = 0
    for x in values:
        p = max(p, _decimals(x))
    return 10**p


def build_problem(raw: dict) -> tuple[Problem, int, int]:
    """把校验过的输入 dict 转成内部 Problem，返回 (problem, scale_w, scale_v)。"""
    items_in = raw["items"]
    types_in = raw["vehicle_types"]
    sw = _scale_factor([it["weight"] for it in items_in] + [t["max_weight"] for t in types_in])
    sv = _scale_factor(
        [it.get("volume", 0.0) for it in items_in] + [t["max_volume"] for t in types_in]
    )

    def to_int(x: float, scale: int) -> int:
        return int(Decimal(str(x)) * scale)

    items = [
        Item(
            id=it["id"],
            w=to_int(it["weight"], sw),
            v=to_int(it.get("volume", 0.0), sv),
            cat=it.get("category"),
            exclusive=bool(it.get("exclusive", False)),
        )
        for it in items_in
    ]
    types = [
        VType(
            name=t["name"],
            capw=to_int(t["max_weight"], sw),
            capv=to_int(t["max_volume"], sv),
            cost=float(t["cost"]),
            count=int(t["count"]),
        )
        for t in types_in
    ]
    conflicts = {frozenset(pair) for pair in raw.get("conflicts", [])}
    return Problem(items=items, types=types, conflict_set=conflicts), sw, sv
