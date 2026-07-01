"""AI 模拟输入采样管理。"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from usb6363 import nidaqmx_driver
from usb6363.device import DeviceContext


# USB-6363 AI 采样率策略。
AI_SINGLE_CHANNEL_MAX_RATE = 2_000_000.0
AI_MULTI_CHANNEL_AGGREGATE_RATE = 1_000_000.0
AI_DEFAULT_BUFFER_SIZE = 100_000
AI_READ_CHUNK_SECONDS = 0.01
AI_CAPTURE_OUTPUT_DIR = "data"


class AiManager:
    """负责 AI 单点读取、后台连续采样、统计和写文件。"""

    def __init__(self, device: DeviceContext) -> None:
        self.device = device

        # 当前被订阅、需要后台连续采样的 AI 通道。
        self._active_channels: list[str] = []
        self._rate = 0.0
        self._running = False
        self._error: str | None = None
        self._last_update = 0.0
        self._sample_counts: dict[str, int] = {}
        self._latest: dict[str, float] = {}
        self._buffers: dict[str, deque[float]] = {}
        self._buffer_size = AI_DEFAULT_BUFFER_SIZE

        # 文件记录状态。record_to_file 会打开它，后台采样线程负责填数据。
        self._recording: dict[str, Any] | None = None

        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def subscribe_channel(self, channel: str) -> dict[str, Any]:
        """订阅一个 AI 通道，让后台开始或继续连续采样。"""

        physical_channel = self.device.normalize_ai_channel(channel)

        with self._lock:
            if physical_channel not in self._active_channels:
                self._active_channels.append(physical_channel)

        self._restart_sampling()
        return self.status()

    def unsubscribe_channel(self, channel: str) -> dict[str, Any]:
        """取消订阅一个 AI 通道。"""

        physical_channel = self.device.normalize_ai_channel(channel)

        with self._lock:
            if physical_channel in self._active_channels:
                self._active_channels.remove(physical_channel)

        self._restart_sampling()
        return self.status()

    def set_channels(self, channels: list[str]) -> dict[str, Any]:
        """一次性设置当前需要连续采样的 AI 通道列表。"""

        normalized_channels: list[str] = []
        for channel in channels:
            physical_channel = self.device.normalize_ai_channel(channel)
            if physical_channel not in normalized_channels:
                normalized_channels.append(physical_channel)

        with self._lock:
            self._active_channels = normalized_channels

        self._restart_sampling()
        return self.status()

    def clear_channels(self) -> dict[str, Any]:
        """取消所有 AI 通道订阅，并停止后台连续采样。"""

        with self._lock:
            self._active_channels = []

        self._restart_sampling()
        return self.status()

    def status(self) -> dict[str, Any]:
        """查询后台 AI 连续采样状态。"""

        with self._lock:
            channels = list(self._active_channels)
            return {
                "running": self._running,
                "channels": channels,
                "channel_count": len(channels),
                "rate_per_channel": self._rate,
                "aggregate_rate": self._rate * len(channels),
                "buffer_size": self._buffer_size,
                "last_update": self._last_update,
                "sample_counts": dict(self._sample_counts),
                "error": self._error,
            }

    def latest(self, channel: str) -> dict[str, Any]:
        """读取后台连续采样缓存里某个通道的最近一个值。"""

        physical_channel = self.device.normalize_ai_channel(channel)

        with self._lock:
            if physical_channel not in self._active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")
            if physical_channel not in self._latest:
                raise RuntimeError(f"{physical_channel} has no sampled data yet")

            return {
                "device": self.device.device_name,
                "channel": physical_channel,
                "rate": self._rate,
                "value": self._latest[physical_channel],
                "last_update": self._last_update,
                "sample_count": self._sample_counts.get(physical_channel, 0),
            }

    def buffer(self, channel: str, max_samples: int = 1000) -> dict[str, Any]:
        """读取后台连续采样缓存里某个通道最近的一段数据。"""

        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        physical_channel = self.device.normalize_ai_channel(channel)

        with self._lock:
            if physical_channel not in self._active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")

            values = list(self._buffers.get(physical_channel, []))[-max_samples:]
            return {
                "device": self.device.device_name,
                "channel": physical_channel,
                "rate": self._rate,
                "samples": len(values),
                "values": values,
                "last_update": self._last_update,
                "sample_count": self._sample_counts.get(physical_channel, 0),
            }

    def stats(self, channel: str, max_samples: int = 10000) -> dict[str, Any]:
        """返回某个通道最近一段缓存数据的统计量。"""

        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        physical_channel = self.device.normalize_ai_channel(channel)

        with self._lock:
            if physical_channel not in self._active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")

            values = list(self._buffers.get(physical_channel, []))[-max_samples:]
            rate = self._rate
            last_update = self._last_update
            sample_count = self._sample_counts.get(physical_channel, 0)

        if not values:
            raise RuntimeError(f"{physical_channel} has no sampled data yet")

        data = np.asarray(values, dtype=np.float64)
        return {
            "device": self.device.device_name,
            "channel": physical_channel,
            "rate": rate,
            "samples": int(data.size),
            "mean": float(np.mean(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "rms": float(np.sqrt(np.mean(np.square(data)))),
            "last": float(data[-1]),
            "last_update": last_update,
            "sample_count": sample_count,
        }

    def record_to_file(
        self,
        seconds: float,
        output_dir: str = AI_CAPTURE_OUTPUT_DIR,
        prefix: str = "ai_capture",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """把后台正在采集的 AI 数据记录到 .npy 文件。"""

        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        with self._lock:
            if not self._active_channels or not self._running:
                raise RuntimeError("AI sampling is not running. Call set_ai_channels first.")
            if self._recording is not None:
                raise RuntimeError("Another AI file recording is already running")

            channels = list(self._active_channels)
            rate = self._rate
            samples_per_channel = max(1, int(round(rate * seconds)))
            done_event = threading.Event()
            self._recording = {
                "channels": channels,
                "rate": rate,
                "seconds": seconds,
                "target_samples": samples_per_channel,
                "buffers": {channel: [] for channel in channels},
                "started_at": time.time(),
                "done_event": done_event,
            }

        wait_timeout = timeout if timeout is not None else seconds + 10.0
        if not done_event.wait(wait_timeout):
            with self._lock:
                self._recording = None
            raise TimeoutError("Timed out while recording AI data to file")

        with self._lock:
            recording = self._recording
            self._recording = None

        if recording is None:
            raise RuntimeError("AI recording ended without data")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        npy_path = output_path / f"{prefix}_{timestamp}.npy"
        json_path = output_path / f"{prefix}_{timestamp}.json"

        data_rows = []
        for channel in channels:
            channel_data = recording["buffers"][channel][:samples_per_channel]
            data_rows.append(np.asarray(channel_data, dtype=np.float64))
        data = np.vstack(data_rows)

        np.save(npy_path, data)

        metadata = {
            "device": self.device.device_name,
            "channels": channels,
            "rate_per_channel": rate,
            "aggregate_rate": rate * len(channels),
            "seconds_requested": seconds,
            "samples_per_channel": int(data.shape[1]),
            "shape": list(data.shape),
            "npy_file": str(npy_path.resolve()),
            "metadata_file": str(json_path.resolve()),
            "started_at": recording["started_at"],
            "finished_at": time.time(),
            "format": "numpy .npy, shape=(channel_count, samples_per_channel)",
        }

        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata

    def read_voltage(
        self,
        channel: str = "ai0",
        samples: int = 1,
        rate: float = 1000.0,
        terminal_config_name: str = "RSE",
        min_val: float = -10.0,
        max_val: float = 10.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """读取模拟输入 AI 电压。

        如果后台采样正在运行，已订阅通道会直接从缓存返回，避免抢设备。
        """

        if samples < 1:
            raise ValueError("samples must be >= 1")
        if rate <= 0:
            raise ValueError("rate must be > 0")

        physical_channel = self.device.normalize_ai_channel(channel)
        with self._lock:
            if self._active_channels:
                if physical_channel not in self._active_channels:
                    raise RuntimeError(
                        f"AI sampling is running, but {physical_channel} is not subscribed. "
                        "Subscribe it first or clear AI channels before direct read."
                    )
                if samples == 1:
                    if physical_channel not in self._latest:
                        raise RuntimeError(f"{physical_channel} has no sampled data yet")
                    return {
                        "device": self.device.device_name,
                        "channel": physical_channel,
                        "samples": 1,
                        "rate": self._rate,
                        "values": self._latest[physical_channel],
                    }

                values = list(self._buffers.get(physical_channel, []))[-samples:]
                return {
                    "device": self.device.device_name,
                    "channel": physical_channel,
                    "samples": len(values),
                    "rate": self._rate,
                    "values": values,
                }

        with self.device.lock:
            values = nidaqmx_driver.read_ai_voltage(
                device_name=self.device.device_name,
                physical_channel=physical_channel,
                samples=samples,
                rate=rate,
                terminal_config_name=terminal_config_name,
                min_val=min_val,
                max_val=max_val,
                timeout=timeout,
            )

        return {
            "device": self.device.device_name,
            "channel": physical_channel,
            "samples": samples,
            "rate": rate,
            "values": values,
        }

    def calculate_rate(self, channels: list[str]) -> float:
        """根据活跃 AI 通道数量计算每个通道的采样率。"""

        channel_count = len(channels)
        if channel_count == 0:
            return 0.0
        if channel_count == 1:
            return AI_SINGLE_CHANNEL_MAX_RATE
        return AI_MULTI_CHANNEL_AGGREGATE_RATE / channel_count

    def _restart_sampling(self) -> None:
        """按当前订阅通道重启后台 AI 连续采样任务。"""

        self._stop_sampling()

        with self._lock:
            channels = list(self._active_channels)
            if not channels:
                self._rate = 0.0
                self._running = False
                self._error = None
                return

            self._rate = self.calculate_rate(channels)
            self._error = None

            for channel in channels:
                self._buffers.setdefault(channel, deque(maxlen=self._buffer_size))
                self._sample_counts.setdefault(channel, 0)

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._sampling_worker,
                args=(channels, self._rate, stop_event),
                daemon=True,
                name="usb6363-ai-sampling",
            )
            self._stop_event = stop_event
            self._thread = thread
            self._running = True
            thread.start()

    def _stop_sampling(self) -> None:
        """停止当前后台 AI 连续采样任务。"""

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

    def _sampling_worker(
        self,
        channels: list[str],
        rate: float,
        stop_event: threading.Event,
    ) -> None:
        """后台 AI 采样线程。"""

        samples_per_read = max(2, int(rate * AI_READ_CHUNK_SECONDS))
        try:
            task = nidaqmx_driver.create_continuous_ai_task(
                channels=channels,
                    rate=rate,
                samples_per_read=samples_per_read,
                )
            try:
                while not stop_event.is_set():
                    channel_values = nidaqmx_driver.read_continuous_ai_chunk(
                        task=task,
                        samples_per_read=samples_per_read,
                        channel_count=len(channels),
                        timeout=2.0,
                    )
                    now = time.time()

                    with self._lock:
                        if stop_event.is_set():
                            break
                        for channel, values in zip(channels, channel_values):
                            if not values:
                                continue
                            self._latest[channel] = values[-1]
                            self._buffers.setdefault(
                                channel,
                                deque(maxlen=self._buffer_size),
                            ).extend(values)
                            self._sample_counts[channel] = (
                                self._sample_counts.get(channel, 0) + len(values)
                            )
                        self._append_recording_locked(channels, channel_values)
                        self._last_update = now
                        self._error = None
            finally:
                task.close()

        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._running = False

    @staticmethod
    def _split_read_values(raw_values: Any, channel_count: int) -> list[list[float]]:
        """把 nidaqmx.Task.read 的返回值统一整理成 list[list[float]]。"""

        if channel_count == 1:
            if isinstance(raw_values, list):
                return [[float(value) for value in raw_values]]
            return [[float(raw_values)]]

        return [
            [float(value) for value in channel_values]
            for channel_values in raw_values
        ]

    def _append_recording_locked(
        self,
        channels: list[str],
        channel_values: list[list[float]],
    ) -> None:
        """把当前数据块追加到文件记录器。"""

        recording = self._recording
        if recording is None:
            return
        if recording["channels"] != channels:
            recording["done_event"].set()
            return

        target_samples = recording["target_samples"]
        buffers = recording["buffers"]

        for channel, values in zip(channels, channel_values):
            current = buffers[channel]
            remaining = target_samples - len(current)
            if remaining > 0:
                current.extend(values[:remaining])

        if all(len(buffers[channel]) >= target_samples for channel in channels):
            recording["done_event"].set()
