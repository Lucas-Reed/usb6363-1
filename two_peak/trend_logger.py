"""手动面积慢漂记录器。

这个模块只负责一件事：在后端长期记录手动面积的慢漂趋势。
它不直接访问 nidaqmx，而是通过 Usb6363Client 读取底层 frame_stream 的最新帧。

第一版刻意保持简单：
- 只记录一个手动面积窗口。
- 不做滤波、不扣本底。
- 用最近 N 帧计算滑动均值、标准差、相对标准差和 shot-to-shot 抖动。
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

from two_peak.signal import measure_manual_area
from usb6363_client import Usb6363Client


@dataclass
class AreaTrendSample:
    """一帧手动面积数据。"""

    frame_id: int
    timestamp: float
    area: float


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

    def start(
        self,
        analysis_channel_index: int,
        area_left: int,
        area_right: int,
        window_frames: int = 200,
        record_hz: float = 1.0,
        poll_interval: float = 0.05,
    ) -> dict[str, Any]:
        """启动长期记录。

        参数说明：
        - analysis_channel_index：分析哪一路 AI。
        - area_left/area_right：手动面积窗口左右边界。
        - window_frames：每个 CSV 点使用最近多少帧做滑动统计。
        - record_hz：每秒写几行 CSV，例如 1 Hz 表示每秒记录一次。
        - poll_interval：后端检查新帧的间隔，一般保持默认即可。
        """

        if window_frames < 2:
            raise ValueError("window_frames must be >= 2")
        if record_hz <= 0:
            raise ValueError("record_hz must be > 0")
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
                "window_frames": int(window_frames),
                "record_hz": float(record_hz),
                "poll_interval": float(poll_interval),
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

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=_csv_fieldnames())
                writer.writeheader()

                while not stop_event.is_set():
                    try:
                        status = self._daq.get_ai_frame_stream_status()
                        frame_id = int(status.get("frame_id", 0))
                        if status.get("running") is not True and not status.get("has_frame"):
                            self._set_error("frame stream is not running and has no frame")
                            time.sleep(float(settings["poll_interval"]))
                            continue

                        if frame_id > last_seen_frame_id:
                            frame = self._daq.get_ai_frame_stream_latest()
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
                            writer.writerow(row)
                            file.flush()
                            last_recorded_frame_id = last_seen_frame_id
                            with self._lock:
                                self._records_written += 1
                                self._latest_stats = dict(row)
                            next_record_time = now + record_period

                    except Exception as exc:
                        self._set_error(str(exc))

                    time.sleep(float(settings["poll_interval"]))

        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._running = False

    def _measure_frame(self, frame: dict[str, Any], settings: dict[str, Any]) -> AreaTrendSample:
        """从一帧波形里计算手动面积。"""

        values = np.asarray(frame["values"], dtype=float)
        channel_index = int(settings["analysis_channel_index"])
        if values.ndim != 2:
            raise RuntimeError("latest frame values must be a 2D array")
        if channel_index < 0 or channel_index >= values.shape[0]:
            raise ValueError("analysis_channel_index is out of range")

        measurement = measure_manual_area(
            values[channel_index],
            left_index=int(settings["area_left"]),
            right_index=int(settings["area_right"]),
        )
        return AreaTrendSample(
            frame_id=int(frame.get("frame_id", 0)),
            timestamp=float(frame.get("finished_at", time.time())),
            area=float(measurement.value),
        )

    def _build_csv_row(
        self,
        samples: deque[AreaTrendSample],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        """根据最近 N 帧面积构造一行 CSV。"""

        window_frames = int(settings["window_frames"])
        recent = list(samples)[-window_frames:]
        areas = np.asarray([sample.area for sample in recent], dtype=float)
        frame_ids = [sample.frame_id for sample in recent]

        mean = float(np.mean(areas))
        std = float(np.std(areas))
        rel_std_percent = _relative_percent(std, mean)

        if areas.size >= 2:
            diffs = np.diff(areas)
            shot2shot_std = float(np.std(diffs))
            shot2shot_rel_std_percent = _relative_percent(shot2shot_std, mean)
            last_delta = float(areas[-1] - areas[-2])
            last_delta_rel_percent = _relative_percent(last_delta, mean)
        else:
            shot2shot_std = 0.0
            shot2shot_rel_std_percent = None
            last_delta = 0.0
            last_delta_rel_percent = None

        latest = recent[-1]
        return {
            "iso_time": datetime.fromtimestamp(latest.timestamp).isoformat(timespec="seconds"),
            "unix_time": latest.timestamp,
            "frame_id": latest.frame_id,
            "frame_id_start": frame_ids[0],
            "frame_id_end": frame_ids[-1],
            "analysis_channel_index": int(settings["analysis_channel_index"]),
            "area_left": int(settings["area_left"]),
            "area_right": int(settings["area_right"]),
            "window_frames": window_frames,
            "sample_count": int(areas.size),
            "area_current": float(areas[-1]),
            "area_mean": mean,
            "area_std": std,
            "area_rel_std_percent": rel_std_percent,
            "shot2shot_last_delta": last_delta,
            "shot2shot_last_delta_rel_percent": last_delta_rel_percent,
            "shot2shot_std": shot2shot_std,
            "shot2shot_rel_std_percent": shot2shot_rel_std_percent,
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


def _csv_fieldnames() -> list[str]:
    """CSV 列名集中放在这里，避免写入行时顺序混乱。"""

    return [
        "iso_time",
        "unix_time",
        "frame_id",
        "frame_id_start",
        "frame_id_end",
        "analysis_channel_index",
        "area_left",
        "area_right",
        "window_frames",
        "sample_count",
        "area_current",
        "area_mean",
        "area_std",
        "area_rel_std_percent",
        "shot2shot_last_delta",
        "shot2shot_last_delta_rel_percent",
        "shot2shot_std",
        "shot2shot_rel_std_percent",
        "record_hz",
    ]
