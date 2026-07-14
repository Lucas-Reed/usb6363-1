"""双窗口面积与峰高慢漂记录器。

这个模块只负责一件事：在后端长期记录动态窗口内面积和峰高的慢漂趋势。
它不直接访问 nidaqmx，而是通过 Usb6363Client 读取统一 AI 流的最新帧。

当前统计原则：
- A、B 两个窗口独立跟随各自的最高点，窗口宽度保持不变。
- 面积使用窗口内原始采样值求和，峰高使用同一窗口内原始采样值的最大值。
- 用最近 N 帧计算滑动均值、标准差、相对标准差和 shot-to-shot 抖动。
- 面积与峰高使用同一个 EMA alpha，便于直接比较两种测量量。
- 按指定记录频率写 CSV，适合一整晚观察慢漂。
"""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from two_peak.signal import measure_manual_area, track_and_measure_manual_area
from usb6363_client import Usb6363Client


@dataclass
class AreaTrendSample:
    """一帧动态窗口测量数据。

    面积和峰高来自同一帧、同一个窗口。峰高使用移动完成后的原始波形最大值，
    因此不会因为跟随定位使用了平滑波形而改变峰高本身的物理含义。
    """

    frame_id: int
    timestamp: float
    area: float
    peak_height: float
    area2: float | None = None
    peak2_height: float | None = None
    area_left: int | None = None
    area_right: int | None = None
    area_peak_index: int | None = None
    peak_height_index: int | None = None
    area2_left: int | None = None
    area2_right: int | None = None
    area2_peak_index: int | None = None
    peak2_height_index: int | None = None


class AreaTrendLogger:
    """后端慢漂 CSV 记录器。

    WebUI 可以关闭或刷新，只要 viewer 后端还在运行，本 logger 就能继续记录。
    """

    def __init__(self, daq: Usb6363Client, output_dir: Path) -> None:
        self._daq = daq
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._error: str | None = None
        self._csv_path: Path | None = None
        self._settings: dict[str, Any] = {}
        self._frames_seen = 0
        self._records_written = 0
        self._last_frame_id = 0
        self._latest_stats: dict[str, Any] | None = None
        self._recent_stats: deque[dict[str, Any]] = deque(maxlen=1000)

    def start(
        self,
        analysis_channel_index: int,
        area_left: int,
        area_right: int,
        area2_left: int | None = None,
        area2_right: int | None = None,
        window_frames: int = 200,
        record_hz: float = 1.0,
        ema_alpha: float = 0.02,
        auto_track_enabled: bool = False,
        auto_track_smooth_window: int = 9,
        auto_track_max_shift: int = 5,
        poll_interval: float = 0.05,
        stream_source: str = "unified_stream",
        channels: list[str] | None = None,
    ) -> dict[str, Any]:
        """启动长期记录。

        参数说明：
        - analysis_channel_index：分析哪一路 AI。
        - area_left/area_right：手动面积窗口左右边界。
        - area2_left/area2_right：可选的第二个面积窗口，用于同时记录另一个峰。
        - window_frames：每个 CSV 点使用最近多少帧做滑动统计。
        - record_hz：每秒写几行 CSV，例如 1 Hz 表示每秒记录一次。
        - poll_interval：后端检查新帧的间隔，一般保持默认即可。
        """

        if window_frames < 2:
            raise ValueError("window_frames must be >= 2")
        if record_hz <= 0:
            raise ValueError("record_hz must be > 0")
        if ema_alpha < 0 or ema_alpha > 1:
            raise ValueError("ema_alpha must be between 0 and 1")
        if auto_track_smooth_window < 1:
            raise ValueError("auto_track_smooth_window must be >= 1")
        if auto_track_max_shift < 0:
            raise ValueError("auto_track_max_shift must be >= 0")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        with self._lock:
            if self._running:
                raise RuntimeError("Area trend logger is already running")

            self._output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self._output_dir / f"area_trend_{timestamp}.csv"

            settings = {
                "analysis_channel_index": int(analysis_channel_index),
                "area_left": int(area_left),
                "area_right": int(area_right),
                "area2_left": None if area2_left is None else int(area2_left),
                "area2_right": None if area2_right is None else int(area2_right),
                "window_frames": int(window_frames),
                "record_hz": float(record_hz),
                "ema_alpha": float(ema_alpha),
                "auto_track_enabled": bool(auto_track_enabled),
                "auto_track_smooth_window": int(auto_track_smooth_window),
                "auto_track_max_shift": int(auto_track_max_shift),
                "poll_interval": float(poll_interval),
                "stream_source": str(stream_source),
                "channels": list(channels or []),
            }

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._worker,
                args=(settings, csv_path, stop_event),
                daemon=True,
                name="two-peak-area-trend-logger",
            )
            self._stop_event = stop_event
            self._thread = thread
            self._running = True
            self._error = None
            self._csv_path = csv_path
            self._settings = dict(settings)
            self._frames_seen = 0
            self._records_written = 0
            self._last_frame_id = 0
            self._latest_stats = None
            self._recent_stats.clear()
            thread.start()

        return self.status()

    def stop(self) -> dict[str, Any]:
        """停止长期记录。"""

        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            if stop_event is not None:
                stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._stop_event = None
                self._running = False

        return self.status()

    def status(self) -> dict[str, Any]:
        """返回当前记录状态。"""

        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "csv_file": str(self._csv_path.resolve()) if self._csv_path else None,
                "settings": dict(self._settings),
                "frames_seen": self._frames_seen,
                "records_written": self._records_written,
                "last_frame_id": self._last_frame_id,
                "latest_stats": dict(self._latest_stats or {}),
                "recent_stats": [dict(row) for row in self._recent_stats],
            }

    def _worker(
        self,
        settings: dict[str, Any],
        csv_path: Path,
        stop_event: threading.Event,
    ) -> None:
        """记录线程主体。"""

        samples: deque[AreaTrendSample] = deque(maxlen=max(10_000, int(settings["window_frames"]) * 5))
        last_seen_frame_id = 0
        last_recorded_frame_id = 0
        next_record_time = time.time()
        record_period = 1.0 / float(settings["record_hz"])
        ema_alpha = float(settings.get("ema_alpha", 0.02))
        area_ema: float | None = None
        area2_ema: float | None = None
        area_sum_ema: float | None = None
        peak_height_ema: float | None = None
        peak2_height_ema: float | None = None

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=_csv_fieldnames())
                writer.writeheader()

                while not stop_event.is_set():
                    try:
                        status = self._get_stream_status(settings)
                        frame_id = int(status.get("frame_id", 0))
                        if status.get("running") is not True and not status.get("has_frame"):
                            self._set_error("AI stream is not running and has no frame")
                            time.sleep(float(settings["poll_interval"]))
                            continue

                        if frame_id > last_seen_frame_id:
                            frame = self._get_stream_latest(settings)
                            sample = self._measure_frame(frame, settings)
                            samples.append(sample)
                            last_seen_frame_id = sample.frame_id
                            with self._lock:
                                self._frames_seen += 1
                                self._last_frame_id = sample.frame_id
                                self._error = None

                        now = time.time()
                        # 只有真的看到新帧之后才写 CSV。
                        # 如果连续采集意外停住，这里不会把同一批旧数据反复写很多行。
                        if now >= next_record_time and samples and last_seen_frame_id > last_recorded_frame_id:
                            row = self._build_csv_row(samples, settings)
                            area_ema = _update_ema(area_ema, row.get("area_mean"), ema_alpha)
                            area2_ema = _update_ema(area2_ema, row.get("area2_mean"), ema_alpha)
                            area_sum_ema = _update_ema(area_sum_ema, row.get("area_sum_mean"), ema_alpha)
                            peak_height_ema = _update_ema(
                                peak_height_ema,
                                row.get("peak_height_mean"),
                                ema_alpha,
                            )
                            peak2_height_ema = _update_ema(
                                peak2_height_ema,
                                row.get("peak2_height_mean"),
                                ema_alpha,
                            )
                            row["area_ema_alpha"] = ema_alpha
                            row["area_ema"] = area_ema
                            row["area2_ema_alpha"] = ema_alpha
                            row["area2_ema"] = area2_ema
                            row["area_sum_ema_alpha"] = ema_alpha
                            row["area_sum_ema"] = area_sum_ema
                            row["peak_height_ema_alpha"] = ema_alpha
                            row["peak_height_ema"] = peak_height_ema
                            row["peak2_height_ema_alpha"] = ema_alpha
                            row["peak2_height_ema"] = peak2_height_ema
                            writer.writerow(row)
                            file.flush()
                            last_recorded_frame_id = last_seen_frame_id
                            with self._lock:
                                self._records_written += 1
                                self._latest_stats = dict(row)
                                self._recent_stats.append(dict(row))
                            next_record_time = now + record_period

                    except Exception as exc:
                        self._set_error(str(exc))

                    time.sleep(float(settings["poll_interval"]))

        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._running = False

    def _get_stream_status(self, settings: dict[str, Any]) -> dict[str, Any]:
        """读取当前面积慢漂记录所依赖的数据流状态。"""

        if settings.get("stream_source") == "unified_stream":
            return self._daq.get_unified_ai_stream_status()
        return self._daq.get_ai_frame_stream_status()

    def _get_stream_latest(self, settings: dict[str, Any]) -> dict[str, Any]:
        """读取当前数据流的最新帧，并按需要过滤到 WebUI 选择的通道。"""

        if settings.get("stream_source") == "unified_stream":
            frame = self._daq.get_unified_ai_stream_latest_frame()
        else:
            frame = self._daq.get_ai_frame_stream_latest()

        channels = settings.get("channels") or []
        if channels:
            frame = _filter_frame_channels(frame, [str(channel) for channel in channels])
        return frame

    def _measure_frame(self, frame: dict[str, Any], settings: dict[str, Any]) -> AreaTrendSample:
        """从一帧波形里计算两个动态窗口的面积和峰高。"""

        values = np.asarray(frame["values"], dtype=float)
        channel_index = int(settings["analysis_channel_index"])
        if values.ndim != 2:
            raise RuntimeError("latest frame values must be a 2D array")
        if channel_index < 0 or channel_index >= values.shape[0]:
            raise ValueError("analysis_channel_index is out of range")

        signal = values[channel_index]
        if settings.get("auto_track_enabled"):
            measurement = track_and_measure_manual_area(
                signal,
                left_index=int(settings["area_left"]),
                right_index=int(settings["area_right"]),
                smooth_window=int(settings.get("auto_track_smooth_window", 9)),
                max_shift_per_frame=int(settings.get("auto_track_max_shift", 5)),
            )
            # 记录线程内部保存最新窗口位置，下一帧从这个位置继续跟随。
            settings["area_left"] = measurement.left_index
            settings["area_right"] = measurement.right_index
        else:
            measurement = measure_manual_area(
                signal,
                left_index=int(settings["area_left"]),
                right_index=int(settings["area_right"]),
            )
        measurement2 = None
        if settings.get("area2_left") is not None and settings.get("area2_right") is not None:
            # 第二个面积窗口和第一个完全独立，仍然使用同一帧、同一分析通道。
            if settings.get("auto_track_enabled"):
                measurement2 = track_and_measure_manual_area(
                    signal,
                    left_index=int(settings["area2_left"]),
                    right_index=int(settings["area2_right"]),
                    smooth_window=int(settings.get("auto_track_smooth_window", 9)),
                    max_shift_per_frame=int(settings.get("auto_track_max_shift", 5)),
                )
                settings["area2_left"] = measurement2.left_index
                settings["area2_right"] = measurement2.right_index
            else:
                measurement2 = measure_manual_area(
                    signal,
                    left_index=int(settings["area2_left"]),
                    right_index=int(settings["area2_right"]),
                )

        # 跟随算法只负责决定窗口位置；峰高始终在移动完成后的原始数据窗口内寻找。
        # 这样面积和峰高严格使用相同的左右边界，WebUI 和 CSV 的统计口径也一致。
        peak_height, peak_height_index = _measure_window_peak_height(
            signal,
            measurement.left_index,
            measurement.right_index,
        )
        peak2_height = None
        peak2_height_index = None
        if measurement2 is not None:
            peak2_height, peak2_height_index = _measure_window_peak_height(
                signal,
                measurement2.left_index,
                measurement2.right_index,
            )

        return AreaTrendSample(
            frame_id=int(frame.get("frame_id", 0)),
            timestamp=float(frame.get("finished_at", time.time())),
            area=float(measurement.value),
            peak_height=peak_height,
            area2=None if measurement2 is None else float(measurement2.value),
            peak2_height=peak2_height,
            area_left=int(measurement.left_index),
            area_right=int(measurement.right_index),
            area_peak_index=measurement.peak_index,
            peak_height_index=peak_height_index,
            area2_left=None if measurement2 is None else int(measurement2.left_index),
            area2_right=None if measurement2 is None else int(measurement2.right_index),
            area2_peak_index=None if measurement2 is None else measurement2.peak_index,
            peak2_height_index=peak2_height_index,
        )

    def _build_csv_row(
        self,
        samples: deque[AreaTrendSample],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """根据最近 N 帧面积和峰高构造一行 CSV。"""

        window_frames = int(settings["window_frames"])
        recent = list(samples)[-window_frames:]
        areas = np.asarray([sample.area for sample in recent], dtype=float)
        area2_values = [sample.area2 for sample in recent if sample.area2 is not None]
        area_sums = np.asarray(
            [sample.area + (0.0 if sample.area2 is None else sample.area2) for sample in recent],
            dtype=float,
        )
        peak_heights = np.asarray([sample.peak_height for sample in recent], dtype=float)
        peak2_heights = [
            sample.peak2_height for sample in recent if sample.peak2_height is not None
        ]
        frame_ids = [sample.frame_id for sample in recent]

        area_stats = _area_stats(areas)
        area2_stats = _area_stats(np.asarray(area2_values, dtype=float)) if area2_values else {}
        area_sum_stats = _area_stats(area_sums)
        peak_height_stats = _area_stats(peak_heights)
        peak2_height_stats = (
            _area_stats(np.asarray(peak2_heights, dtype=float)) if peak2_heights else {}
        )

        latest = recent[-1]
        latest_area_sum = latest.area + (0.0 if latest.area2 is None else latest.area2)
        return {
            "iso_time": datetime.fromtimestamp(latest.timestamp).isoformat(timespec="seconds"),
            "unix_time": latest.timestamp,
            "frame_id": latest.frame_id,
            "frame_id_start": frame_ids[0],
            "frame_id_end": frame_ids[-1],
            "analysis_channel_index": int(settings["analysis_channel_index"]),
            "auto_track_enabled": bool(settings.get("auto_track_enabled", False)),
            "auto_track_smooth_window": int(settings.get("auto_track_smooth_window", 9)),
            "auto_track_max_shift": int(settings.get("auto_track_max_shift", 5)),
            "area_left": latest.area_left,
            "area_right": latest.area_right,
            "area_peak_index": latest.area_peak_index,
            "peak_height_index": latest.peak_height_index,
            "area2_left": latest.area2_left,
            "area2_right": latest.area2_right,
            "area2_peak_index": latest.area2_peak_index,
            "peak2_height_index": latest.peak2_height_index,
            "window_frames": window_frames,
            "sample_count": int(areas.size),
            "area_current": float(areas[-1]),
            "area_mean": area_stats["mean"],
            "area_std": area_stats["std"],
            "area_rel_std_percent": area_stats["rel_std_percent"],
            "area_ema_alpha": float(settings.get("ema_alpha", 0.02)),
            "area_ema": None,
            "area2_current": latest.area2,
            "area2_mean": area2_stats.get("mean"),
            "area2_std": area2_stats.get("std"),
            "area2_rel_std_percent": area2_stats.get("rel_std_percent"),
            "area2_ema_alpha": float(settings.get("ema_alpha", 0.02)),
            "area2_ema": None,
            "area_sum_current": latest_area_sum,
            "area_sum_mean": area_sum_stats["mean"],
            "area_sum_std": area_sum_stats["std"],
            "area_sum_rel_std_percent": area_sum_stats["rel_std_percent"],
            "area_sum_ema_alpha": float(settings.get("ema_alpha", 0.02)),
            "area_sum_ema": None,
            "peak_height_current": latest.peak_height,
            "peak_height_mean": peak_height_stats["mean"],
            "peak_height_std": peak_height_stats["std"],
            "peak_height_rel_std_percent": peak_height_stats["rel_std_percent"],
            "peak_height_ema_alpha": float(settings.get("ema_alpha", 0.02)),
            "peak_height_ema": None,
            "peak2_height_current": latest.peak2_height,
            "peak2_height_mean": peak2_height_stats.get("mean"),
            "peak2_height_std": peak2_height_stats.get("std"),
            "peak2_height_rel_std_percent": peak2_height_stats.get("rel_std_percent"),
            "peak2_height_ema_alpha": float(settings.get("ema_alpha", 0.02)),
            "peak2_height_ema": None,
            "shot2shot_last_delta": area_stats["last_delta"],
            "shot2shot_last_delta_rel_percent": area_stats["last_delta_rel_percent"],
            "shot2shot_std": area_stats["shot2shot_std"],
            "shot2shot_rel_std_percent": area_stats["shot2shot_rel_std_percent"],
            "area_sum_shot2shot_last_delta": area_sum_stats["last_delta"],
            "area_sum_shot2shot_last_delta_rel_percent": area_sum_stats["last_delta_rel_percent"],
            "area_sum_shot2shot_std": area_sum_stats["shot2shot_std"],
            "area_sum_shot2shot_rel_std_percent": area_sum_stats["shot2shot_rel_std_percent"],
            "area2_shot2shot_last_delta": area2_stats.get("last_delta"),
            "area2_shot2shot_last_delta_rel_percent": area2_stats.get("last_delta_rel_percent"),
            "area2_shot2shot_std": area2_stats.get("shot2shot_std"),
            "area2_shot2shot_rel_std_percent": area2_stats.get("shot2shot_rel_std_percent"),
            "peak_height_shot2shot_last_delta": peak_height_stats["last_delta"],
            "peak_height_shot2shot_last_delta_rel_percent": peak_height_stats[
                "last_delta_rel_percent"
            ],
            "peak_height_shot2shot_std": peak_height_stats["shot2shot_std"],
            "peak_height_shot2shot_rel_std_percent": peak_height_stats[
                "shot2shot_rel_std_percent"
            ],
            "peak2_height_shot2shot_last_delta": peak2_height_stats.get("last_delta"),
            "peak2_height_shot2shot_last_delta_rel_percent": peak2_height_stats.get(
                "last_delta_rel_percent"
            ),
            "peak2_height_shot2shot_std": peak2_height_stats.get("shot2shot_std"),
            "peak2_height_shot2shot_rel_std_percent": peak2_height_stats.get(
                "shot2shot_rel_std_percent"
            ),
            "record_hz": float(settings["record_hz"]),
        }

    def _set_error(self, message: str) -> None:
        """记录线程错误，但不立刻退出，方便等待采集流恢复。"""

        with self._lock:
            self._error = message


def _relative_percent(std_or_delta: float, mean: float) -> float | None:
    """把绝对标准差或差值转换成相对百分比。"""

    if abs(mean) < 1e-30:
        return None
    return float(std_or_delta / abs(mean) * 100.0)


def _measure_window_peak_height(
    signal: np.ndarray,
    left_index: int,
    right_index: int,
) -> tuple[float, int]:
    """返回指定闭区间内原始波形的最高值及其全局索引。"""

    data = np.asarray(signal, dtype=float)
    left = max(0, min(int(left_index), data.size - 1))
    right = max(0, min(int(right_index), data.size - 1))
    if right < left:
        left, right = right, left
    window = data[left : right + 1]
    if window.size == 0:
        raise ValueError("peak height window must not be empty")
    local_index = int(np.argmax(window))
    return float(window[local_index]), left + local_index


def _update_ema(previous: float | None, value: Any, alpha: float) -> float | None:
    """更新指数滑动平均。

    alpha=0 表示关闭 EMA，此时返回 None。
    value=None 常见于没有设置 B 窗口的情况，此时不更新。
    """

    if alpha <= 0 or value is None:
        return previous if alpha > 0 else None
    raw_value = float(value)
    if previous is None:
        return raw_value
    return float(alpha * raw_value + (1.0 - alpha) * previous)


def _area_stats(values: np.ndarray) -> dict[str, Any]:
    """计算一个面积序列的滑动统计量。

    这里被 A/B 两个面积窗口共用，避免两套公式以后不小心改得不一致。
    """

    if values.size < 1:
        return {
            "mean": None,
            "std": None,
            "rel_std_percent": None,
            "last_delta": None,
            "last_delta_rel_percent": None,
            "shot2shot_std": None,
            "shot2shot_rel_std_percent": None,
        }

    mean = float(np.mean(values))
    std = float(np.std(values))
    rel_std_percent = _relative_percent(std, mean)

    if values.size >= 2:
        diffs = np.diff(values)
        shot2shot_std = float(np.std(diffs))
        shot2shot_rel_std_percent = _relative_percent(shot2shot_std, mean)
        last_delta = float(values[-1] - values[-2])
        last_delta_rel_percent = _relative_percent(last_delta, mean)
    else:
        shot2shot_std = 0.0
        shot2shot_rel_std_percent = None
        last_delta = 0.0
        last_delta_rel_percent = None

    return {
        "mean": mean,
        "std": std,
        "rel_std_percent": rel_std_percent,
        "last_delta": last_delta,
        "last_delta_rel_percent": last_delta_rel_percent,
        "shot2shot_std": shot2shot_std,
        "shot2shot_rel_std_percent": shot2shot_rel_std_percent,
    }


def _channel_short_name(channel: str) -> str:
    """把 Dev2/ai1、ai1 这两种写法统一成 ai1。"""

    return str(channel).strip().split("/")[-1].lower()


def _filter_frame_channels(frame: dict[str, Any], requested_channels: list[str]) -> dict[str, Any]:
    """把统一流的大帧裁成双峰查看器当前选择的通道。"""

    frame_channels = [str(channel) for channel in frame.get("channels", [])]
    frame_values = frame.get("values", [])
    if len(frame_channels) != len(frame_values):
        raise RuntimeError("latest frame channels and values length do not match")

    selected_channels: list[str] = []
    selected_values: list[Any] = []
    used_indices: set[int] = set()
    for requested in requested_channels:
        requested_short = _channel_short_name(requested)
        match_index = None
        for index, channel in enumerate(frame_channels):
            if index in used_indices:
                continue
            if _channel_short_name(channel) == requested_short:
                match_index = index
                break
        if match_index is None:
            raise RuntimeError(
                f"latest frame does not contain requested channel {requested}; "
                f"available channels are {frame_channels}"
            )
        used_indices.add(match_index)
        selected_channels.append(frame_channels[match_index])
        selected_values.append(frame_values[match_index])

    filtered = dict(frame)
    filtered["source_channels"] = frame_channels
    filtered["channels"] = selected_channels
    filtered["values"] = selected_values
    filtered["channel_count"] = len(selected_channels)
    return filtered


def _csv_fieldnames() -> list[str]:
    """CSV 列名集中放在这里，避免写入行时顺序混乱。"""

    return [
        "iso_time",
        "unix_time",
        "frame_id",
        "frame_id_start",
        "frame_id_end",
        "analysis_channel_index",
        "auto_track_enabled",
        "auto_track_smooth_window",
        "auto_track_max_shift",
        "area_left",
        "area_right",
        "area_peak_index",
        "peak_height_index",
        "area2_left",
        "area2_right",
        "area2_peak_index",
        "peak2_height_index",
        "window_frames",
        "sample_count",
        "area_current",
        "area_mean",
        "area_std",
        "area_rel_std_percent",
        "area_ema_alpha",
        "area_ema",
        "area2_current",
        "area2_mean",
        "area2_std",
        "area2_rel_std_percent",
        "area2_ema_alpha",
        "area2_ema",
        "area_sum_current",
        "area_sum_mean",
        "area_sum_std",
        "area_sum_rel_std_percent",
        "area_sum_ema_alpha",
        "area_sum_ema",
        "peak_height_current",
        "peak_height_mean",
        "peak_height_std",
        "peak_height_rel_std_percent",
        "peak_height_ema_alpha",
        "peak_height_ema",
        "peak2_height_current",
        "peak2_height_mean",
        "peak2_height_std",
        "peak2_height_rel_std_percent",
        "peak2_height_ema_alpha",
        "peak2_height_ema",
        "shot2shot_last_delta",
        "shot2shot_last_delta_rel_percent",
        "shot2shot_std",
        "shot2shot_rel_std_percent",
        "area_sum_shot2shot_last_delta",
        "area_sum_shot2shot_last_delta_rel_percent",
        "area_sum_shot2shot_std",
        "area_sum_shot2shot_rel_std_percent",
        "area2_shot2shot_last_delta",
        "area2_shot2shot_last_delta_rel_percent",
        "area2_shot2shot_std",
        "area2_shot2shot_rel_std_percent",
        "peak_height_shot2shot_last_delta",
        "peak_height_shot2shot_last_delta_rel_percent",
        "peak_height_shot2shot_std",
        "peak_height_shot2shot_rel_std_percent",
        "peak2_height_shot2shot_last_delta",
        "peak2_height_shot2shot_last_delta_rel_percent",
        "peak2_height_shot2shot_std",
        "peak2_height_shot2shot_rel_std_percent",
        "record_hz",
    ]
