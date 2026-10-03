"""领域模型：问题、车辆、方案。

单位约定：重量 weight 为吨、体积 volume 为方，仅用于展示；算法只依赖其数值大小。
所有可行性判定统一使用 EPS 相对容差，避免浮点误差把边界解误判为超载。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# 相对容差：判定超载/超容时允许 1e-9 的相对误差。
EPS = 1e-9


def capacity_epsilon(cap: float) -> float:
    return EPS * max(1.0, abs(cap))


@dataclass(frozen=True)
class Shipment:
    """一票货。category 为 None 表示不属于任何互斥类别。"""

    id: str
    weight: float
    volume: float
    category: Optional[str]
    exclusive: bool
    index: int  # 原始输入顺序（0 起），用于确定性排序


@dataclass(frozen=True)
class VehicleType:
    """一种车型。"""

    id: str
    max_weight: float
    max_volume: float
    cost: float
    available: int


@dataclass
class Problem:
    shipments: list[Shipment]
    vehicle_types: list[VehicleType]
    # 冲突类别对（无序，去重后）。
    conflicts: set[tuple[str, str]]
    # 类别 -> 与之冲突的类别集合（含自身当且仅当给了自冲突对）。
    conflict_adj: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._type_by_id = {vt.id: vt for vt in self.vehicle_types}
        self._ship_by_id = {s.id: s for s in self.shipments}

    def vehicle_type(self, type_id: str) -> VehicleType:
        return self._type_by_id[type_id]

    def shipment(self, ship_id: str) -> Shipment:
        return self._ship_by_id[ship_id]


@dataclass
class Vehicle:
    """方案中的一辆车。"""

    type_id: str
    shipment_ids: list[str] = field(default_factory=list)
    weight: float = 0.0
    volume: float = 0.0
    # category -> 本车上该类别的票数（离开/进入时据此维护集合）。
    _cat_counts: dict[str, int] = field(default_factory=dict)

    @property
    def categories(self) -> set[str]:
        return set(self._cat_counts)

    def _exceeds(self, ship: Shipment, vtype: VehicleType) -> Optional[str]:
        if self.weight + ship.weight > vtype.max_weight + capacity_epsilon(vtype.max_weight):
            return "weight"
        if self.volume + ship.volume > vtype.max_volume + capacity_epsilon(vtype.max_volume):
            return "volume"
        return None

    def can_add(
        self,
        ship: Shipment,
        problem: Problem,
        vtype: Optional[VehicleType] = None,
    ) -> bool:
        """能否把 ship 加入本车（载重、容积、类别互斥）。不修改状态。"""
        if vtype is None:
            vtype = problem.vehicle_type(self.type_id)
        if self._exceeds(ship, vtype) is not None:
            return False
        if ship.category is not None:
            blocked = self._cat_counts.keys() & problem.conflict_adj.get(ship.category, set())
            if blocked:
                return False
        return True

    def fits_capacity(self, ship: Shipment, vtype: VehicleType) -> bool:
        return self._exceeds(ship, vtype) is None


@dataclass
class Solution:
    """一个完整配车方案。"""

    vehicles: list[Vehicle]
    problem: Problem

    def cost(self) -> float:
        return sum(self.problem.vehicle_type(v.type_id).cost for v in self.vehicles)

    def vehicle_count(self) -> int:
        return len(self.vehicles)

    def type_counts(self) -> dict[str, int]:
        counts = {vt.id: 0 for vt in self.problem.vehicle_types}
        for v in self.vehicles:
            counts[v.type_id] += 1
        return counts

    # ---- 货物增删（唯一的状态修改入口，保证类别计数一致）----

    def add_to(self, vehicle: Vehicle, ship: Shipment) -> None:
        vehicle.shipment_ids.append(ship.id)
        vehicle.weight += ship.weight
        vehicle.volume += ship.volume
        if ship.category is not None:
            vehicle._cat_counts[ship.category] = vehicle._cat_counts.get(ship.category, 0) + 1

    def remove_from(self, vehicle: Vehicle, ship: Shipment) -> None:
        vehicle.shipment_ids.remove(ship.id)
        vehicle.weight -= ship.weight
        vehicle.volume -= ship.volume
        if ship.category is not None:
            n = vehicle._cat_counts[ship.category] - 1
            if n == 0:
                del vehicle._cat_counts[ship.category]
            else:
                vehicle._cat_counts[ship.category] = n

    def location_of(self) -> dict[str, Vehicle]:
        loc: dict[str, Vehicle] = {}
        for v in self.vehicles:
            for sid in v.shipment_ids:
                loc[sid] = v
        return loc

    def clone(self) -> "Solution":
        return Solution(
            vehicles=[
                Vehicle(
                    type_id=v.type_id,
                    shipment_ids=list(v.shipment_ids),
                    weight=v.weight,
                    volume=v.volume,
                    _cat_counts=dict(v._cat_counts),
                )
                for v in self.vehicles
            ],
            problem=self.problem,
        )
