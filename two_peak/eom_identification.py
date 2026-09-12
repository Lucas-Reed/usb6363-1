"""EOM/AOM 频率标定与候选边带标注。

该模块只处理索引和频率之间的标定关系，不访问采集卡，也不驱动锁定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_BREAKPOINTS = (2500, 7500)


@dataclass(frozen=True)
class CalibrationModel:
    """由用户在波形查看器中手动确认的 EOM/AOM 标定点。"""

    carrier_index: int
    aom_zero_index: int
    frequency_difference_mhz: float
    fsr_mhz: float
    breakpoints: tuple[int, ...] = DEFAULT_BREAKPOINTS
    version: int = 1
    updated_at: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self, sample_count: int | None = None) -> None:
        """校验标定值，尽早拒绝无法产生有意义标注的输入。"""

        if int(self.carrier_index) != self.carrier_index or int(self.aom_zero_index) != self.aom_zero_index:
            raise ValueError("carrier_index and aom_zero_index must be integers")
        if self.carrier_index < 0 or self.aom_zero_index < 0:
            raise ValueError("calibration indices must be non-negative")
        if self.carrier_index == self.aom_zero_index:
            raise ValueError("carrier_index and aom_zero_index must be different")
        if not math.isfinite(float(self.frequency_difference_mhz)) or self.frequency_difference_mhz == 0:
            raise ValueError("frequency_difference_mhz must be finite and non-zero")
        if not math.isfinite(float(self.fsr_mhz)) or self.fsr_mhz <= 0:
            raise ValueError("fsr_mhz must be finite and positive")
        points = tuple(int(point) for point in self.breakpoints)
        if points != tuple(sorted(set(points))) or any(point < 0 for point in points):
            raise ValueError("breakpoints must be sorted, unique, non-negative integers")
        if sample_count is not None and sample_count > 0:
            if self.carrier_index >= sample_count or self.aom_zero_index >= sample_count:
                raise ValueError("calibration index is outside the current frame")

    @property
    def index_per_mhz(self) -> float:
        """由两个手动点得到的局部索引/MHz 比例。"""

        return (self.aom_zero_index - self.carrier_index) / self.frequency_difference_mhz

    def frequency_for_index(self, index: float) -> float:
        """返回相对于 carrier 的频率偏移，单位 MHz。"""

        return (float(index) - self.carrier_index) / self.index_per_mhz

    def index_for_frequency(self, frequency_offset_mhz: float) -> float:
        """返回给定相对频率对应的波形索引。"""

        return self.carrier_index + float(frequency_offset_mhz) * self.index_per_mhz

    def candidate_sidebands(
        self,
        sample_count: int | None = None,
        orders: Iterable[int] = (-2, -1, 1, 2),
    ) -> list[dict[str, Any]]:
        """根据 FSR 生成候选边带，仅返回标注数据。"""

        self.validate(sample_count=sample_count)
        candidates: list[dict[str, Any]] = []
        for order in orders:
            order_int = int(order)
            if order_int == 0:
                continue
            offset = order_int * self.fsr_mhz
            index_float = self.index_for_frequency(offset)
            index = int(round(index_float))
            in_range = sample_count is None or 0 <= index < sample_count
            segment = self.segment_for_index(index)
            candidates.append(
                {
                    "kind": "sideband",
                    "label": f"EOM {order_int:+d} FSR",
                    "order": order_int,
                    "index": index,
                    "index_float": index_float,
                    "frequency_offset_mhz": offset,
                    "segment": segment,
                    "in_range": in_range,
                }
            )
        return candidates

    def segment_for_index(self, index: int) -> int:
        """返回索引所在的分段，便于前端解释断点附近的候选。"""

        for segment, breakpoint in enumerate(self.breakpoints):
            if index < breakpoint:
                return segment
        return len(self.breakpoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "carrier_index": self.carrier_index,
            "aom_zero_index": self.aom_zero_index,
            "frequency_difference_mhz": self.frequency_difference_mhz,
            "fsr_mhz": self.fsr_mhz,
            "breakpoints": list(self.breakpoints),
            "updated_at": self.updated_at,
            "index_per_mhz": self.index_per_mhz,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationModel":
        if not isinstance(payload, dict):
            raise ValueError("calibration model must be an object")
        raw_breakpoints = payload.get("breakpoints", DEFAULT_BREAKPOINTS)
        if isinstance(raw_breakpoints, str):
            raw_breakpoints = [item.strip() for item in raw_breakpoints.split(",") if item.strip()]
        if not isinstance(raw_breakpoints, (list, tuple)):
            raise ValueError("breakpoints must be a list")
        raw_updated_at = payload.get("updated_at")
        updated_at = None if raw_updated_at in (None, "") else float(raw_updated_at)
        return cls(
            carrier_index=int(payload["carrier_index"]),
            aom_zero_index=int(payload["aom_zero_index"]),
            frequency_difference_mhz=float(payload["frequency_difference_mhz"]),
            fsr_mhz=float(payload["fsr_mhz"]),
            breakpoints=tuple(int(item) for item in raw_breakpoints),
            version=int(payload.get("version", 1)),
            updated_at=updated_at,
        )
