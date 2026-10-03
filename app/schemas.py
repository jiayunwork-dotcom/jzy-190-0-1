"""HTTP 入参的 pydantic 模式与业务校验。

校验分两层：
  1) pydantic：类型、非有限数（NaN/Inf）、缺失字段、字段名错误 -> 422，逐字段给出位置；
  2) validate_problem：跨字段的业务规则（超尺寸、独占货超总数、冲突类别不存在等）
     -> 422，错误体含 field 与 message。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _finite(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} 必须是有限数，不能为 NaN 或无穷")
    return float(value)


class VehicleTypeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    max_weight: float
    max_volume: float
    cost: float
    available: int = Field(ge=0)

    @field_validator("max_weight", "max_volume", "cost")
    @classmethod
    def _check_finite(cls, v, info):
        return _finite(info.field_name, v)

    @field_validator("max_weight", "max_volume", "cost")
    @classmethod
    def _check_nonneg(cls, v, info):
        if v < 0:
            raise ValueError(f"{info.field_name} 不能为负数")
        return v


class ShipmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    weight: float
    volume: float = 0.0
    category: Optional[str] = None
    exclusive: bool = False

    @field_validator("weight", "volume")
    @classmethod
    def _check_weight(cls, v, info):
        v = _finite(info.field_name, v)
        if v < 0:
            raise ValueError(f"{info.field_name} 不能为负数")
        return v

    @field_validator("category")
    @classmethod
    def _check_category(cls, v):
        if v is not None and not v:
            raise ValueError("category 为空串时请使用 null（无类别）")
        return v


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shipments: list[ShipmentIn] = Field(default_factory=list)
    vehicle_types: list[VehicleTypeIn] = Field(min_length=1)
    # 冲突类别对：无序对，["A", "B"] 表示 A、B 互斥；["A", "A"] 表示 A 类自冲突。
    conflicts: list[list[str]] = Field(default_factory=list)
    seed: int = 0
    time_limit_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("time_limit_seconds")
    @classmethod
    def _check_tl(cls, v):
        return _finite("time_limit_seconds", v)

    @field_validator("conflicts")
    @classmethod
    def _check_conflict_shape(cls, v):
        for i, pair in enumerate(v):
            if len(pair) != 2 or not all(isinstance(c, str) and c for c in pair):
                raise ValueError(f"conflicts[{i}] 必须是两个非空类别名组成的数组")
        return v


@dataclass
class FieldError:
    field: str
    message: str


def validate_problem(req: JobRequest) -> list[FieldError]:
    """跨字段业务校验，返回错误列表（空列表表示通过）。"""
    errors: list[FieldError] = []

    # 货 id 唯一
    seen_ids: set[str] = set()
    for i, s in enumerate(req.shipments):
        if s.id in seen_ids:
            errors.append(FieldError(f"shipments[{i}].id", f"货号 {s.id} 重复"))
        seen_ids.add(s.id)

    # 车型 id 唯一
    seen_types: set[str] = set()
    for i, t in enumerate(req.vehicle_types):
        if t.id in seen_types:
            errors.append(FieldError(f"vehicle_types[{i}].id", f"车型 {t.id} 重复"))
        seen_types.add(t.id)

    if not req.vehicle_types:
        errors.append(FieldError("vehicle_types", "至少需要一种车型"))
        return errors

    # 每票货必须装得进至少一种车型（重量、体积两个维度）。
    for i, s in enumerate(req.shipments):
        fits = [
            t for t in req.vehicle_types
            if s.weight <= t.max_weight and s.volume <= t.max_volume
        ]
        if not fits:
            errors.append(
                FieldError(
                    f"shipments[{i}]",
                    (
                        f"货 {s.id}（重 {s.weight} 吨 / 体积 {s.volume} 方）"
                        "比任何可用车型都大，无法装载"
                    ),
                )
            )

    # 独占货数量不得超过所有车型可用总数。
    exclusive_count = sum(1 for s in req.shipments if s.exclusive)
    total_available = sum(t.available for t in req.vehicle_types)
    if exclusive_count > total_available:
        errors.append(
            FieldError(
                "shipments.exclusive",
                f"独占货有 {exclusive_count} 票，但所有车型可用总数仅 {total_available} 辆",
            )
        )
    # 逐车型维度也要能装下：存在独占货装不进任何车型时，单件检查已覆盖。

    # 冲突对引用的类别必须在货中出现过。
    used_categories = {s.category for s in req.shipments if s.category is not None}
    for i, pair in enumerate(req.conflicts):
        for c in pair:
            if c not in used_categories:
                errors.append(
                    FieldError(
                        f"conflicts[{i}]",
                        f"冲突关系引用了不存在的类别 {c!r}（没有任何货属于该类别）",
                    )
                )

    return errors
