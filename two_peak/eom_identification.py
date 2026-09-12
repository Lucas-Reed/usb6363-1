"""EOM/AOM 频率标定与候选边带标注。

该模块只处理索引和频率之间的标定关系，不访问采集卡，也不驱动锁定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


DEFAULT_BREAKPOINTS = (2500, 7500)


def folded_scan_coordinate(
    indices: np.ndarray | Iterable[float],
    breakpoints: tuple[int, int] = DEFAULT_BREAKPOINTS,
) -> np.ndarray:
    """Map the three sawtooth branches onto one monotonic scan coordinate.

    With the default turning points this maps 0..2500 to 2500..5000,
    2500..7500 to 5000..0, and 7500..10000 to 0..2500.
    """

    first, second = (int(value) for value in breakpoints)
    if first <= 0 or second <= first:
        raise ValueError("breakpoints must satisfy 0 < first < second")
    span = float(second - first)
    half = span / 2.0
    x = np.asarray(indices, dtype=float)
    return np.where(x < first, half + x, np.where(x < second, second - x, x - second))


def scan_segment(index: int, breakpoints: tuple[int, int]) -> int:
    first, second = breakpoints
    return 0 if index < first else (1 if index < second else 2)


def find_physical_peaks(
    signal: np.ndarray | Iterable[float],
    *,
    min_separation: int = 45,
    min_width: int = 8,
    max_width: int = 70,
    local_radius: int = 120,
) -> list[dict[str, Any]]:
    """Find independently broadened FP resonances and reject broad shoulders."""

    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 5 or not np.all(np.isfinite(values)):
        raise ValueError("signal must contain at least five finite samples")
    base = float(np.percentile(values, 10))
    median = float(np.median(values))
    noise = 1.4826 * float(np.median(np.abs(values - median)))
    threshold = base + max(3.0 * noise, 0.02 * float(np.ptp(values)))
    local = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= threshold)
    ) + 1
    kept: list[int] = []
    for index in sorted(local.tolist(), key=lambda item: values[item], reverse=True):
        if all(abs(index - other) >= min_separation for other in kept):
            kept.append(index)

    peaks: list[dict[str, Any]] = []
    for index in sorted(kept):
        left_bound = max(0, index - local_radius)
        right_bound = min(values.size, index + local_radius + 1)
        local_base = float(np.percentile(values[left_bound:right_bound], 10))
        prominence = float(values[index] - local_base)
        half_level = local_base + prominence / 2.0
        left = index
        right = index
        while left > left_bound and values[left] > half_level:
            left -= 1
        while right < right_bound - 1 and values[right] > half_level:
            right += 1
        width = right - left - 1
        independent = (
            min_width <= width <= max_width
            and index - left >= 2
            and right - index >= 2
            and left > left_bound
            and right < right_bound - 1
        )
        if independent:
            peaks.append(
                {
                    "index": int(index),
                    "value": float(values[index]),
                    "prominence": prominence,
                    "width_samples": int(width),
                }
            )
    return peaks


def _frequency_match(
    frequency: float,
    *,
    eom_frequency_mhz: float,
    aom_frequency_mhz: float,
    fsr_mhz: float,
    max_eom_order: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for eom_order in range(-max_eom_order, max_eom_order + 1):
        for aom_order in (0, 1):
            optical = eom_order * eom_frequency_mhz + aom_order * aom_frequency_mhz
            cavity_order = int(round((frequency - optical) / fsr_mhz))
            residual = frequency - optical - cavity_order * fsr_mhz
            candidate = {
                "eom_order": eom_order,
                "aom_order": aom_order,
                "cavity_order": cavity_order,
                "frequency_offset_mhz": optical,
                "residual_mhz": float(residual),
            }
            if best is None or abs(residual) < abs(float(best["residual_mhz"])):
                best = candidate
    assert best is not None
    return best


def identify_eom_aom_spectrum(
    signal: np.ndarray | Iterable[float],
    *,
    spacing_samples: float,
    spacing_mhz: float,
    spacing_tolerance_samples: float = 30.0,
    eom_frequency_mhz: float = 6800.0,
    fsr_mhz: float = 2500.0,
    breakpoints: tuple[int, int] = DEFAULT_BREAKPOINTS,
    max_eom_order: int = 4,
    residual_tolerance_mhz: float = 35.0,
) -> dict[str, Any]:
    """Identify the carrier/AOM pair, fit the folded scan, and label peaks.

    The supplied spacing calibration fixes the linear scan scale. Each
    same-branch peak pair near ``spacing_samples`` is tried as carrier and AOM
    first order. The pair anchors the folded quadratic frequency model used by
    the reference analysis, and the remaining peaks independently validate it
    against ``n*EOM + a*AOM + m*FSR``.
    """

    if spacing_samples <= 0 or spacing_mhz <= 0:
        raise ValueError("spacing_samples and spacing_mhz must be > 0")
    if spacing_tolerance_samples <= 0 or eom_frequency_mhz <= 0 or fsr_mhz <= 0:
        raise ValueError("tolerances and RF/FSR values must be > 0")
    points = tuple(int(value) for value in breakpoints)
    if len(points) != 2:
        raise ValueError("exactly two scan breakpoints are required")
    values = np.asarray(signal, dtype=float).reshape(-1)
    if points[0] <= 0 or points[1] <= points[0] or points[1] >= values.size:
        raise ValueError("scan breakpoints must lie inside the current waveform")
    peaks = find_physical_peaks(values)
    if len(peaks) < 2:
        raise ValueError("fewer than two independently broadened peaks were found")

    alpha = float(spacing_mhz / spacing_samples)
    span = float(points[1] - points[0])
    scan_amplitude = alpha * span
    indices = np.asarray([peak["index"] for peak in peaks], dtype=float)
    coords = folded_scan_coordinate(indices, points)
    shape = coords * (coords - span) / span
    pair_fits: list[dict[str, Any]] = []
    for left_index in range(len(peaks) - 1):
        for right_index in range(left_index + 1, len(peaks)):
            first_peak = peaks[left_index]
            second_peak = peaks[right_index]
            if scan_segment(first_peak["index"], points) != scan_segment(second_peak["index"], points):
                continue
            raw_spacing = abs(second_peak["index"] - first_peak["index"])
            spacing_error = abs(raw_spacing - spacing_samples)
            if spacing_error > spacing_tolerance_samples:
                continue
            pair = sorted((left_index, right_index), key=lambda item: coords[item])
            carrier_i, aom_i = pair[0], pair[1]
            delta_shape = shape[aom_i] - shape[carrier_i]
            if abs(delta_shape) < 1e-12:
                curvature = 0.0
            else:
                curvature = (
                    spacing_mhz - alpha * (coords[aom_i] - coords[carrier_i])
                ) / delta_shape
            # End-point slopes alpha-curvature and alpha+curvature must keep
            # the folded frequency coordinate monotonic.
            if abs(curvature) >= alpha:
                continue
            offset = (-alpha * coords[carrier_i] - curvature * shape[carrier_i]) % fsr_mhz
            fitted = offset + alpha * coords + curvature * shape
            matches = [
                _frequency_match(
                    float(frequency),
                    eom_frequency_mhz=eom_frequency_mhz,
                    aom_frequency_mhz=spacing_mhz,
                    fsr_mhz=fsr_mhz,
                    max_eom_order=max_eom_order,
                )
                for frequency in fitted
            ]
            carrier_cavity = int(round(float(fitted[carrier_i]) / fsr_mhz))
            matches[carrier_i] = {
                "eom_order": 0,
                "aom_order": 0,
                "cavity_order": carrier_cavity,
                "frequency_offset_mhz": 0.0,
                "residual_mhz": float(fitted[carrier_i] - carrier_cavity * fsr_mhz),
            }
            matches[aom_i] = {
                "eom_order": 0,
                "aom_order": 1,
                "cavity_order": carrier_cavity,
                "frequency_offset_mhz": float(spacing_mhz),
                "residual_mhz": float(
                    fitted[aom_i] - spacing_mhz - carrier_cavity * fsr_mhz
                ),
            }
            residuals = np.asarray([match["residual_mhz"] for match in matches])
            accepted = np.abs(residuals) <= residual_tolerance_mhz
            validation_mask = np.ones(len(peaks), dtype=bool)
            validation_mask[[carrier_i, aom_i]] = False
            validation_accepted = validation_mask & accepted
            validation_count = int(np.count_nonzero(validation_mask))
            independent_matches = int(np.count_nonzero(validation_accepted))
            validation_rms = (
                float(np.sqrt(np.mean(np.square(residuals[validation_accepted]))))
                if independent_matches
                else None
            )
            collisions: list[list[int]] = []
            assigned: dict[tuple[int, int, int, int], int] = {}
            for peak_index, peak, match, is_accepted in zip(
                range(len(peaks)), peaks, matches, accepted
            ):
                if not bool(is_accepted):
                    continue
                key = (
                    scan_segment(int(peak["index"]), points),
                    int(match["eom_order"]),
                    int(match["aom_order"]),
                    int(match["cavity_order"]),
                )
                previous = assigned.get(key)
                if previous is not None:
                    collisions.append([int(peaks[previous]["index"]), int(peak["index"])])
                else:
                    assigned[key] = peak_index
            strength = float(first_peak["prominence"] + second_peak["prominence"])
            pair_fits.append(
                {
                    "spacing_error_samples": float(spacing_error),
                    "carrier_i": carrier_i,
                    "aom_i": aom_i,
                    "offset_mhz": float(offset),
                    "curvature_mhz_per_sample": float(curvature),
                    "frequencies": fitted,
                    "matches": matches,
                    "accepted": accepted,
                    "validation_rms_mhz": validation_rms,
                    "validation_peak_count": validation_count,
                    "independent_match_count": independent_matches,
                    "coverage": (
                        independent_matches / validation_count if validation_count else 0.0
                    ),
                    "collisions": collisions,
                    "strength": strength,
                }
            )
    if not pair_fits:
        raise ValueError("no same-branch peak pair matches the calibrated spacing")
    max_strength = max(float(item["strength"]) for item in pair_fits)
    for item in pair_fits:
        residual_cost = (
            1.5
            if item["validation_rms_mhz"] is None
            else min(3.0, float(item["validation_rms_mhz"]) / residual_tolerance_mhz)
        )
        strength_quality = float(item["strength"]) / max(max_strength, 1e-12)
        item["strength_quality"] = strength_quality
        item["score"] = (
            float(item["spacing_error_samples"]) / spacing_tolerance_samples
            + 0.8 * residual_cost
            + 0.5 * (1.0 - float(item["coverage"]))
            + 0.75 * len(item["collisions"])
            + 0.25 * (1.0 - strength_quality)
        )
    pair_fits.sort(key=lambda item: float(item["score"]))
    best = pair_fits[0]
    labeled: list[dict[str, Any]] = []
    for peak, coordinate, frequency, match, accepted in zip(
        peaks, coords, best["frequencies"], best["matches"], best["accepted"]
    ):
        item = dict(peak)
        item.update(match)
        item.update(
            {
                "kind": "fitted_peak",
                "folded_coordinate": float(coordinate),
                "fitted_frequency_mhz": float(frequency),
                "segment": scan_segment(int(peak["index"]), points),
                "accepted": bool(accepted),
                "label": (
                    f"EOM {int(match['eom_order']):+d}, AOM {int(match['aom_order'])}"
                    if accepted
                    else "unassigned"
                ),
            }
        )
        labeled.append(item)
    spacing_quality = max(0.0, 1.0 - best["spacing_error_samples"] / spacing_tolerance_samples)
    validation_rms = best["validation_rms_mhz"]
    residual_quality = (
        0.0
        if validation_rms is None
        else max(0.0, 1.0 - float(validation_rms) / residual_tolerance_mhz)
    )
    coverage_quality = float(best["coverage"])
    if len(pair_fits) == 1:
        uniqueness_quality = 1.0
    else:
        score_gap = float(pair_fits[1]["score"]) - float(best["score"])
        uniqueness_quality = max(0.0, min(1.0, score_gap / 0.75))
    evidence_factor = 0.35 + 0.65 * min(
        1.0, int(best["independent_match_count"]) / 3.0
    )
    confidence = float(
        evidence_factor
        * (
            0.30 * spacing_quality
            + 0.25 * residual_quality
            + 0.20 * coverage_quality
            + 0.15 * float(best["strength_quality"])
            + 0.10 * uniqueness_quality
        )
    )
    carrier = labeled[int(best["carrier_i"])]
    aom_first = labeled[int(best["aom_i"])]
    carrier["label"] = "EOM carrier"
    carrier["kind"] = "carrier"
    carrier["anchor"] = "carrier"
    aom_first["label"] = "AOM first order"
    aom_first["kind"] = "aom_first"
    aom_first["anchor"] = "aom_first"
    return {
        "display_only": True,
        "carrier": carrier,
        "aom_first": aom_first,
        "peaks": labeled,
        "fit": {
            "breakpoints": list(points),
            "linear_mhz_per_sample": alpha,
            "scan_amplitude_mhz": scan_amplitude,
            "offset_mhz": best["offset_mhz"],
            "curvature_mhz_per_sample": best["curvature_mhz_per_sample"],
            "slope_start_mhz_per_sample": alpha - best["curvature_mhz_per_sample"],
            "slope_end_mhz_per_sample": alpha + best["curvature_mhz_per_sample"],
            "residual_rms_mhz": validation_rms,
            "matched_peak_count": int(np.count_nonzero(best["accepted"])),
            "detected_peak_count": len(peaks),
            "independent_validation_count": int(best["independent_match_count"]),
            "candidate_pair_count": len(pair_fits),
            "same_branch_collisions": best["collisions"],
        },
        "calibration": {
            "spacing_samples": float(spacing_samples),
            "spacing_mhz": float(spacing_mhz),
            "eom_frequency_mhz": float(eom_frequency_mhz),
            "fsr_mhz": float(fsr_mhz),
        },
        "confidence": confidence,
        "ambiguous": bool(
            confidence < 0.7
            or int(best["independent_match_count"]) < 2
            or bool(best["collisions"])
        ),
    }


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
