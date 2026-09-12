"""纯逻辑的 PFI1 触发后续 AI 窗口测量。

本模块只消费已有连续 AI 流的读取回调，不创建或管理 NI-DAQmx task。
调用方在收到 PFI1 事件（以 AI sample index 表示）后调用
``Pfi1FollowupWindow.process_event``，即可得到窗口均值和 EMA。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


ReadSamples = Callable[[int, int], Any]


@dataclass(frozen=True)
class Pfi1WindowResult:
    """一次 PFI1 后续窗口的测量结果。"""

    event_sample: int
    window_start: int
    window_samples: int
    value: float
    pulse_value: float | None
    pulse_start: int | None
    pulse_end: int | None
    average_start: int
    average_end: int
    pulse_detected: bool
    ema: float | None
    samples: tuple[float, ...]
    threshold: float
    polarity: str

    @property
    def mean(self) -> float:
        """测量均值别名，便于调用方按统计语义读取。"""

        return self.value

    @property
    def pulse_mean(self) -> float | None:
        return self.pulse_value

    def as_dict(self) -> dict[str, Any]:
        """返回适合 API/日志传输的普通字典。"""

        return {
            "event_sample": self.event_sample,
            "window_start": self.window_start,
            "window_samples": self.window_samples,
            "value": self.value,
            "mean": self.value,
            "pulse_value": self.pulse_value,
            "pulse_mean": self.pulse_value,
            "pulse_start": self.pulse_start,
            "pulse_end": self.pulse_end,
            "average_start": self.average_start,
            "average_end": self.average_end,
            "pulse_detected": self.pulse_detected,
            "ema": self.ema,
            "samples": list(self.samples),
            "threshold": self.threshold,
            "polarity": self.polarity,
        }


class Pfi1FollowupWindow:
    """从连续 AI 读取回调中提取 PFI1 事件后的一个窗口。

    ``read_samples`` 的签名为 ``(start_sample, count) -> 1-D samples``。
    ``sample_rate_hz`` 用于把 ``delay_ms`` 换算成采样点；读取回调仍由调用方
    连接到已有连续 AI 缓冲，因此本类不会启动第二个采集任务。
    """

    def __init__(
        self,
        read_samples: ReadSamples,
        sample_rate_hz: float,
        delay_ms: float,
        window_samples: int,
        threshold: float,
        polarity: str = "positive",
        ema_alpha: float = 0.02,
    ) -> None:
        if not callable(read_samples):
            raise TypeError("read_samples must be callable")
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if not np.isfinite(delay_ms) or delay_ms < 0:
            raise ValueError("delay_ms must be >= 0")
        if int(window_samples) < 1:
            raise ValueError("window_samples must be >= 1")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if ema_alpha < 0 or ema_alpha > 1:
            raise ValueError("ema_alpha must be between 0 and 1")
        normalized_polarity = _normalize_polarity(polarity)
        self.read_samples = read_samples
        self.sample_rate_hz = float(sample_rate_hz)
        self.delay_ms = float(delay_ms)
        self.window_samples = int(window_samples)
        self.threshold = float(threshold)
        self.polarity = normalized_polarity
        self.ema_alpha = float(ema_alpha)
        self._ema: float | None = None

    @property
    def ema(self) -> float | None:
        return self._ema

    def reset(self) -> None:
        """清除 EMA，下一次事件将从窗口均值重新开始。"""

        self._ema = None

    def process_event(self, event_sample: int | dict[str, Any]) -> Pfi1WindowResult:
        """读取并测量一个 PFI1 事件后的窗口。"""

        event_index = _event_sample_index(event_sample)
        delay_samples = int(round(self.delay_ms * self.sample_rate_hz / 1000.0))
        window_start = event_index + delay_samples
        raw = self.read_samples(window_start, self.window_samples)
        values = np.asarray(raw, dtype=float).reshape(-1)
        if values.size != self.window_samples:
            raise ValueError(
                f"read_samples returned {values.size} samples; "
                f"expected {self.window_samples}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("read_samples returned non-finite samples")

        pulse = _find_complete_pulse(values, self.threshold, self.polarity)
        if pulse is None:
            pulse_value = None
            pulse_start = None
            pulse_end = None
            measured = float(np.mean(values))
            detected = False
            average_start = window_start
            average_end = window_start + self.window_samples - 1
        else:
            pulse_start, pulse_end = pulse
            pulse_value = float(np.mean(values[pulse_start : pulse_end + 1]))
            measured = pulse_value
            detected = True
            average_start = window_start + pulse_start
            average_end = window_start + pulse_end

        self._ema = _update_ema(self._ema, measured, self.ema_alpha)
        return Pfi1WindowResult(
            event_sample=event_index,
            window_start=window_start,
            window_samples=self.window_samples,
            value=measured,
            pulse_value=pulse_value,
            pulse_start=(None if pulse_start is None else window_start + pulse_start),
            pulse_end=(None if pulse_end is None else window_start + pulse_end),
            average_start=average_start,
            average_end=average_end,
            pulse_detected=detected,
            ema=self._ema,
            samples=tuple(float(value) for value in values),
            threshold=self.threshold,
            polarity=self.polarity,
        )

    # Short alias for callers that prefer treating the object as a processor.
    measure = process_event
    capture = process_event


def measure_pfi1_followup_window(
    read_samples: ReadSamples,
    event_sample: int | dict[str, Any],
    delay_ms: float,
    window_samples: int,
    threshold: float,
    polarity: str = "positive",
    *,
    sample_rate_hz: float,
    ema_previous: float | None = None,
    ema_alpha: float = 0.02,
) -> Pfi1WindowResult:
    """一次性测量接口，适合 core/server 直接调用。"""

    processor = Pfi1FollowupWindow(
        read_samples=read_samples,
        sample_rate_hz=sample_rate_hz,
        delay_ms=delay_ms,
        window_samples=window_samples,
        threshold=threshold,
        polarity=polarity,
        ema_alpha=ema_alpha,
    )
    processor._ema = ema_previous
    return processor.process_event(event_sample)


def _event_sample_index(event_sample: int | dict[str, Any]) -> int:
    if isinstance(event_sample, dict):
        for key in ("sample_index", "event_sample", "sample"):
            if key in event_sample:
                event_sample = event_sample[key]
                break
        else:
            raise ValueError("event_sample dictionary needs sample_index")
    index = int(event_sample)
    if index < 0:
        raise ValueError("event_sample must be >= 0")
    return index


def _normalize_polarity(polarity: str) -> str:
    value = str(polarity).strip().lower()
    if value in {"positive", "high", "rising", "+", "above"}:
        return "positive"
    if value in {"negative", "low", "falling", "-", "below"}:
        return "negative"
    raise ValueError("polarity must be positive/high/rising or negative/low/falling")


def _find_complete_pulse(
    values: np.ndarray, threshold: float, polarity: str
) -> tuple[int, int] | None:
    active = values >= threshold if polarity == "positive" else values <= threshold
    # 窗口起点本身没有可观测的上升沿；要求前一个采样点在窗口内且为非脉冲。
    rising = np.flatnonzero(active & ~np.concatenate(([True], active[:-1])))
    if rising.size == 0:
        return None
    start = int(rising[0])
    falling = np.flatnonzero(~active[start:])
    if falling.size == 0:
        return None
    end = start + int(falling[0]) - 1
    if end < start:
        return None
    return start, end


def _update_ema(previous: float | None, value: float, alpha: float) -> float | None:
    if alpha <= 0:
        return previous
    if previous is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * previous)


__all__ = [
    "PFI1WindowCapture",
    "Pfi1FollowupWindow",
    "Pfi1WindowResult",
    "measure_pfi1_followup_window",
]


# Public name used by the server-facing integration layer.  Keep the mixed-case
# implementation name above for normal Python style and backwards readability.
PFI1WindowCapture = Pfi1FollowupWindow
