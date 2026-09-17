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
    x = np.asarray(indices, dtype=float)
    return np.where(x < first, span - first + x, np.where(x < second, second - x, x - second))


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
            and index - left >= (1 if min_width < 4 else 2)
            and right - index >= (1 if min_width < 4 else 2)
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


def identify_eom_aom_spectrum(
    signal: np.ndarray | Iterable[float],
    *,
    spacing_samples: float,
    spacing_mhz: float,
    spacing_tolerance_samples: float = 3.0,
    eom_frequency_mhz: float = 6800.0,
    fsr_mhz: float = 2500.0,
    breakpoints: tuple[int, int] | None = None,
    max_eom_order: int = 4,
    match_tolerance_samples: float = 3.0,
    min_peak_width: int | None = None,
    max_peak_width: int | None = None,
) -> dict[str, Any]:
    """Compare both carrier hypotheses using a folded, calibrated spectrum.

    The EOM comb and the separate AOM first-order line share the FP modulo-FSR
    frequency axis. Assignment tolerance is measured in original sample units.
    """
    values = np.asarray(signal, dtype=float).reshape(-1)
    scalars = (spacing_samples, spacing_mhz, spacing_tolerance_samples,
               eom_frequency_mhz, fsr_mhz, match_tolerance_samples)
    if not all(math.isfinite(v) and v > 0 for v in scalars):
        raise ValueError("calibration, frequencies and sample tolerances must be positive")
    if not 1 <= max_eom_order <= 10:
        raise ValueError("max_eom_order must be between 1 and 10")
    points = tuple(breakpoints) if breakpoints is not None else (values.size // 4, 3 * values.size // 4)
    if len(points) != 2 or not 0 < points[0] < points[1] < values.size:
        raise ValueError("scan breakpoints must lie inside the current waveform")
    span = float(points[1] - points[0])
    scale = values.size / 10000.0
    width_min = max(1, round(8 * scale)) if min_peak_width is None else min_peak_width
    width_max = max(4, round(70 * scale)) if max_peak_width is None else max_peak_width
    if not 1 <= width_min <= width_max:
        raise ValueError("peak widths must satisfy 1 <= minimum <= maximum")
    peaks = find_physical_peaks(values, min_separation=max(2, round(45 * scale)),
                                min_width=width_min, max_width=width_max,
                                local_radius=max(8, round(120 * scale)))
    if len(peaks) < 2:
        raise ValueError("fewer than two width-qualified peaks; check sampling and peak widths")
    indices = np.array([p["index"] for p in peaks], dtype=float)
    coords = folded_scan_coordinate(indices, points)
    shape = coords * (coords - span) / span
    alpha = spacing_mhz / spacing_samples
    segments = np.array([scan_segment(int(i), points) for i in indices])
    # One AOM beam is shifted from the carrier; EOM harmonics are not each
    # assumed to have an additional shifted copy.
    orders = np.arange(-max_eom_order, max_eom_order + 1)
    lattice = np.r_[orders * eom_frequency_mhz, spacing_mhz]
    nlabels = np.r_[orders, 0]
    alabels = np.r_[np.zeros(len(orders), dtype=int), 1]
    tolerance = match_tolerance_samples
    candidates = []
    for left in range(len(peaks) - 1):
        for right in range(left + 1, len(peaks)):
            # The calibration pair is defined on the descending branch.
            # Other branches are still used by the full-spectrum fit after
            # this pair has established the local frequency scale.
            if segments[left] != 1 or segments[right] != 1:
                continue
            spacing_error = abs(abs(coords[right] - coords[left]) - spacing_samples)
            if spacing_error > spacing_tolerance_samples:
                continue
            for carrier, partner in ((left, right), (right, left)):
                # Anchor Q(carrier)=0 and search the nonlinear correction
                # against all the other peaks, not just this pair.
                direction = float(np.sign(coords[partner] - coords[carrier]))
                delta_s = direction * (coords - coords[carrier])
                pair_slope = (shape[partner] - shape[carrier]) / (coords[partner] - coords[carrier])
                # Preserve the measured pair's secant calibration even when
                # fitting curvature. Either optical frequency direction is possible.
                delta_shape = direction * (shape - shape[carrier]) - pair_slope * delta_s
                curve_limit = 0.85 * alpha / (1 + abs(pair_slope))
                curves = np.linspace(-curve_limit, curve_limit, 171)
                qs = alpha * delta_s[None, :] + curves[:, None] * delta_shape[None, :]
                residual = (qs[:, :, None] - lattice + fsr_mhz / 2) % fsr_mhz - fsr_mhz / 2
                nearest = np.min(np.abs(residual), axis=2)
                slopes = alpha + curves[:, None] * (2 * coords / span - 1 - pair_slope)
                point_errors = nearest / slopes
                mask = np.arange(len(peaks)) != carrier
                losses = np.mean(np.minimum((point_errors[:, mask] / tolerance) ** 2, 9), axis=1)
                losses += 1e-9 * (curves / alpha) ** 2
                curvature = float(curves[int(np.argmin(losses))])
                # Refine curvature with inlier assignments while keeping the
                # user calibration and the selected carrier fixed.
                for _ in range(4):
                    q = alpha * delta_s + curvature * delta_shape
                    res = (q[:, None] - lattice + fsr_mhz / 2) % fsr_mhz - fsr_mhz / 2
                    labels = np.argmin(np.abs(res), axis=1)
                    r = res[np.arange(len(peaks)), labels]
                    slope = alpha + curvature * (2 * coords / span - 1 - pair_slope)
                    inliers = (np.abs(r / slope) <= tolerance) & mask
                    denom = float(np.sum(delta_shape[inliers] ** 2))
                    if denom < 1e-12:
                        break
                    proposal = curvature - float(np.sum(delta_shape[inliers] * r[inliers])) / denom
                    if abs(proposal) > curve_limit:
                        break
                    curvature = proposal
                q = alpha * delta_s + curvature * delta_shape
                res = (q[:, None] - lattice + fsr_mhz / 2) % fsr_mhz - fsr_mhz / 2
                labels = np.argmin(np.abs(res), axis=1)
                r = res[np.arange(len(peaks)), labels]
                targets = q - r
                linear = direction * (alpha - curvature * pair_slope)
                quadratic = direction * curvature
                offset = -linear * coords[carrier] - quadratic * shape[carrier]
                # Invert the monotonic quadratic to obtain exact predicted
                # sample positions. Linearized MHz residuals are insufficient
                # near a turning point or with significant curvature.
                lo = np.zeros(len(peaks))
                hi = np.full(len(peaks), span)
                for _ in range(45):
                    mid = (lo + hi) / 2
                    freq = offset + linear * mid + quadratic * mid * (mid - span) / span
                    below = direction * freq < direction * targets
                    lo = np.where(below, mid, lo)
                    hi = np.where(below, hi, mid)
                predicted_s = (lo + hi) / 2
                scan_limits = sorted((offset, offset + linear * span))
                in_scan = (targets >= scan_limits[0]) & (targets <= scan_limits[1])
                predicted = np.where(segments == 0, predicted_s - span + points[0],
                                    np.where(segments == 1, points[1] - predicted_s,
                                             points[1] + predicted_s))
                branch_valid = ((segments == 0) & (predicted >= 0) & (predicted < points[0])) | (
                    (segments == 1) & (predicted >= points[0]) & (predicted < points[1])) | (
                    (segments == 2) & (predicted >= points[1]) & (predicted < values.size))
                errors = indices - predicted
                accepted = in_scan & branch_valid & (np.abs(errors) <= tolerance)
                # Distinct peaks on one branch cannot represent the same line.
                cavity = np.rint((targets - lattice[labels]) / fsr_mhz).astype(int)
                collisions = 0
                assigned = {}
                for k in np.argsort(np.abs(errors)):
                    if not accepted[k]:
                        continue
                    key = (int(segments[k]), int(labels[k]), int(cavity[k]))
                    if key in assigned:
                        accepted[k] = False
                        collisions += 1
                    else:
                        assigned[key] = int(k)
                independent = accepted & (np.arange(len(peaks)) != carrier) & (np.arange(len(peaks)) != partner)
                validation_rms = float(np.sqrt(np.mean(errors[independent] ** 2))) if np.any(independent) else None
                loss = float(np.mean(np.minimum((errors[mask] / tolerance) ** 2, 9)))
                loss += float(np.count_nonzero(~accepted)) / len(peaks) + collisions
                pair_is_aom = bool(accepted[partner] and alabels[labels[partner]] == 1)
                candidates.append(dict(carrier=carrier, partner=partner, curvature=quadratic, linear=linear,
                    offset=offset, q=q, labels=labels, cavity=cavity, residual=r,
                    predicted=predicted, errors=errors, accepted=accepted,
                    independent=int(np.count_nonzero(independent)), rms=validation_rms,
                    score=loss, pair_is_aom=pair_is_aom, collisions=collisions))
    if not candidates:
        raise ValueError("no peak pair matches the 190 MHz calibration; check the current samples/MHz scale")
    candidates.sort(key=lambda c: (c["score"], not c["pair_is_aom"]))
    best = candidates[0]
    # Equivalent carriers observed on opposite scan branches are one hypothesis.
    alternatives = [c for c in candidates[1:]
                    if abs(coords[c["carrier"]] - coords[best["carrier"]]) > tolerance]
    margin = alternatives[0]["score"] - best["score"] if alternatives else None
    labeled = []
    for k, peak in enumerate(peaks):
        label = int(best["labels"][k])
        n, a = int(nlabels[label]), int(alabels[label])
        accepted = bool(best["accepted"][k])
        labeled.append(dict(peak, eom_order=n, aom_order=a,
            cavity_order=int(best["cavity"][k]), frequency_offset_mhz=float(lattice[label]),
            residual_mhz=float(best["residual"][k]), residual_samples=float(best["errors"][k]),
            predicted_index=float(best["predicted"][k]), folded_coordinate=float(coords[k]),
            segment=int(segments[k]), accepted=accepted, kind="fitted_peak",
            label=("AOM first order" if a else ("EOM carrier" if n == 0 else f"EOM {n:+d}")) if accepted else "unassigned"))
    carrier_peak = labeled[best["carrier"]]
    carrier_peak.update(label="EOM carrier", kind="carrier", anchor="carrier")
    aom_peaks = [p for p in labeled if p["accepted"] and p["aom_order"] == 1]
    aom = labeled[best["partner"]] if best["pair_is_aom"] else (aom_peaks[0] if aom_peaks else None)
    if aom:
        aom["kind"] = "aom_first"
    eom_first = [p for p in labeled if p["accepted"] and p["aom_order"] == 0 and abs(p["eom_order"]) == 1]
    central_pairs = [(e, a) for e in eom_first for a in aom_peaks if e["segment"] == a["segment"] == 1]
    central = min(central_pairs, key=lambda pair: (
        abs(pair[0]["index"] - pair[1]["index"]),
        abs((pair[0]["index"] + pair[1]["index"]) / 2 - sum(points) / 2))) if central_pairs else None
    coverage = float(np.mean(best["accepted"]))
    confidence = min(1.0, best["independent"] / 3) * coverage / (1 + best["score"])
    ambiguous = best["independent"] < 2 or coverage < 0.6 or (margin is not None and margin < 0.15) or not aom
    return dict(display_only=True, carrier=carrier_peak, aom_first=aom, eom_first=eom_first,
        centering_peaks=list(central) if central else [], peaks=labeled, confidence=confidence,
        ambiguous=bool(ambiguous), hypotheses=[
            dict(carrier_index=peaks[c["carrier"]]["index"], partner_index=peaks[c["partner"]]["index"],
                 score=c["score"], residual_rms_samples=c["rms"], matched_peak_count=int(np.count_nonzero(c["accepted"])),
                 pair_is_aom=c["pair_is_aom"]) for c in candidates],
        fit=dict(breakpoints=list(points), sample_count=values.size, linear_mhz_per_sample=best["linear"],
            scan_amplitude_mhz=abs(best["linear"] * span), offset_mhz=best["offset"],
            curvature_mhz_per_sample=best["curvature"],
            slope_start_mhz_per_sample=best["linear"] - best["curvature"],
            slope_end_mhz_per_sample=best["linear"] + best["curvature"],
            residual_rms_mhz=(float(np.sqrt(np.mean(best["residual"][best["accepted"]] ** 2)))
                              if best["independent"] else None),
            residual_rms_samples=best["rms"], match_tolerance_samples=tolerance,
            independent_validation_count=best["independent"], candidate_pair_count=len(candidates) // 2,
            matched_peak_count=int(np.count_nonzero(best["accepted"])), detected_peak_count=len(peaks)),
        calibration=dict(spacing_samples=spacing_samples, spacing_mhz=spacing_mhz,
                         eom_frequency_mhz=eom_frequency_mhz, fsr_mhz=fsr_mhz))


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
