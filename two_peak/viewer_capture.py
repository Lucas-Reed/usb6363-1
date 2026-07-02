"""双峰波形查看器的采集、测峰和样本保存逻辑。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from two_peak.signal import locate_and_measure_two_peaks
from two_peak.viewer_state import ViewerState


def parse_channels(value: Any) -> list[str]:
    """把 WebUI 传来的通道参数整理成列表。

    支持两种写法：
        ["ai0", "ai1"]
        "ai0, ai1"
    """

    if isinstance(value, list):
        channels = [str(item).strip() for item in value]
    elif isinstance(value, str):
        channels = [item.strip() for item in value.split(",")]
    else:
        raise ValueError("channels must be a list or comma-separated string")

    channels = [channel for channel in channels if channel]
    if not channels:
        raise ValueError("channels must not be empty")
    return channels


def bool_value(value: Any) -> bool:
    """把 WebUI/JSON 里的布尔值安全转换成 Python bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def capture_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """调用底层 API 同步采集一帧。"""

    channels = parse_channels(body.get("channels", ["ai0", "ai1"]))
    samples = int(body.get("samples", 5000))
    rate = float(body.get("rate", 50_000.0))
    terminal_config = str(body.get("terminal_config", "DIFF"))
    min_val = float(body.get("min_val", -5.0))
    max_val = float(body.get("max_val", 5.0))
    timeout = float(body.get("timeout", 10.0))
    trigger_enabled = bool_value(body.get("trigger_enabled", False))
    trigger_source = str(body.get("trigger_source", "PFI0"))
    trigger_edge = str(body.get("trigger_edge", "RISING"))

    frame = state.daq.capture_ai_frame(
        channels=channels,
        samples=samples,
        rate=rate,
        terminal_config=terminal_config,
        min_val=min_val,
        max_val=max_val,
        timeout=timeout,
        trigger_enabled=trigger_enabled,
        trigger_source=trigger_source,
        trigger_edge=trigger_edge,
    )
    frame["captured_by"] = "two_peak_viewer"
    frame["viewer_received_at"] = time.time()
    return frame


def measure_latest_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """测量最近一帧里的 P1/P2。

    当前版本默认在第一路 AI 波形上测峰，因为旧双峰程序里 AI0 是 FP 透射信号。
    """

    frame = state.latest_frame
    if frame is None:
        raise RuntimeError("No frame has been captured yet")

    values = np.asarray(frame["values"], dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise RuntimeError("latest frame has no AI waveform")

    peak_indices = body.get("peak_indices")
    if not isinstance(peak_indices, list) or len(peak_indices) != 2:
        raise ValueError("peak_indices must be a two-item list")

    smooth_window = int(body.get("smooth_window", 25))
    search_window_half = int(body.get("search_window_half", 20))
    measure_half = int(body.get("measure_half", 5))
    peak_mode = str(body.get("peak_mode", "height"))
    analysis_channel_index = int(body.get("analysis_channel_index", 0))
    if analysis_channel_index < 0 or analysis_channel_index >= values.shape[0]:
        raise ValueError("analysis_channel_index is out of range")

    _, measurements = locate_and_measure_two_peaks(
        ai0=values[analysis_channel_index],
        peak_indices=[int(peak_indices[0]), int(peak_indices[1])],
        smooth_window=smooth_window,
        search_window_half=search_window_half,
        measure_half=measure_half,
        mode=peak_mode,
    )

    return {
        "analysis_channel_index": analysis_channel_index,
        "analysis_channel": frame["channels"][analysis_channel_index],
        "peak_indices_input": [int(peak_indices[0]), int(peak_indices[1])],
        "smooth_window": smooth_window,
        "search_window_half": search_window_half,
        "measure_half": measure_half,
        "peak_mode": peak_mode,
        "measurements": [asdict(item) for item in measurements],
    }


def save_latest_frame(state: ViewerState, body: dict[str, Any]) -> dict[str, Any]:
    """保存最近一帧原始数据和元数据。"""

    frame = state.latest_frame
    if frame is None:
        raise RuntimeError("No frame has been captured yet")

    sample_dir = state.sample_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = str(body.get("label", "")).strip()
    safe_label = "".join(ch for ch in label if ch.isalnum() or ch in ("-", "_"))
    stem = f"{timestamp}_{safe_label}" if safe_label else timestamp

    npy_path = sample_dir / f"{stem}.npy"
    json_path = sample_dir / f"{stem}.json"

    values = np.asarray(frame["values"], dtype=np.float64)
    np.save(npy_path, values)

    metadata = {
        "saved_at": time.time(),
        "npy_file": str(npy_path.resolve()),
        "metadata_file": str(json_path.resolve()),
        "shape": list(values.shape),
        "frame": frame_summary(frame),
        "measurement": state.latest_measurement,
        "web_parameters": body.get("parameters", {}),
        "factory_defaults": state.settings.to_web_parameters(),
        "format": "numpy .npy, shape=(channel_count, samples_per_channel)",
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata


def frame_summary(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    """返回不包含巨大 values 的帧摘要。"""

    if frame is None:
        return None
    return {
        "device": frame.get("device"),
        "channels": frame.get("channels"),
        "channel_count": frame.get("channel_count"),
        "samples_per_channel": frame.get("samples_per_channel"),
        "rate_per_channel": frame.get("rate_per_channel"),
        "aggregate_rate": frame.get("aggregate_rate"),
        "terminal_config": frame.get("terminal_config"),
        "min_val": frame.get("min_val"),
        "max_val": frame.get("max_val"),
        "duration_seconds": frame.get("duration_seconds"),
        "started_at": frame.get("started_at"),
        "finished_at": frame.get("finished_at"),
        "trigger_enabled": frame.get("trigger_enabled"),
        "trigger_source": frame.get("trigger_source"),
        "trigger_edge": frame.get("trigger_edge"),
    }
