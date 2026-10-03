from __future__ import annotations

from pydantic import BaseModel


class ItemIn(BaseModel):
    id: str
    weight: float
    volume: float = 0.0
    category: str | None = None
    exclusive: bool = False


class VehicleTypeIn(BaseModel):
    name: str
    max_weight: float
    max_volume: float
    cost: float
    count: int


class ProblemIn(BaseModel):
    items: list[ItemIn]
    vehicle_types: list[VehicleTypeIn]
    categories: list[str] = []  # 可选：显式声明类别全集
    conflicts: list[tuple[str, str]] = []  # 互相冲突的类别对


class OptionsIn(BaseModel):
    seed: int = 0
    time_limit_seconds: float = 30.0


class SubmitRequest(BaseModel):
    problem: ProblemIn
    options: OptionsIn = OptionsIn()
