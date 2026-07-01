"""USB-6363 核心控制逻辑。

这个文件是给人读的“主逻辑”：
    1. 设备信息和通道名校验
    2. AI 动态采样率管理、后台缓存、统计、写文件
    3. AO 输出
    4. PFI/数字线读写和边沿计数

重要边界：
    本文件不直接 import nidaqmx。
    唯一直接接触 NI-DAQmx 的文件是 usb6363/nidaqmx_driver.py。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from usb6363 import nidaqmx_driver


# NI MAX / NI-DAQmx 里给 USB-6363 设置的设备名。
DEVICE_NAME = "Dev2"

# AI 采样率策略：
# 1 个 AI 通道活跃时，单通道拉满到 2 MHz。
# 2 个及以上 AI 通道活跃时，多通道总采样率按 1 MHz 均分。
AI_SINGLE_CHANNEL_MAX_RATE = 2_000_000.0
AI_MULTI_CHANNEL_AGGREGATE_RATE = 1_000_000.0

# 后台连续采样时，每个通道最多缓存多少个最近数据点。
AI_DEFAULT_BUFFER_SIZE = 100_000

# 后台采样线程每次从 NI-DAQmx 读一小块数据。
AI_READ_CHUNK_SECONDS = 0.01

# 高速采样写文件时的默认目录。
AI_CAPTURE_OUTPUT_DIR = "data"

# capture_ai_frame 会把原始数组放进 JSON 返回，不能拿它做长时间高速采集。
# 超过这个点数时应该使用 record_ai_to_file，避免 Web/API 被巨大 JSON 拖垮。
AI_FRAME_MAX_JSON_SAMPLES = 200_000


_AI_RE = re.compile(r"^(?:(?P<device>Dev\d+)/)?ai(?P<index>\d+)$")
_AO_RE = re.compile(r"^(?:(?P<device>Dev\d+)/)?ao(?P<index>\d+)$")
_PFI_RE = re.compile(r"^/?(?:(?P<device>Dev\d+)/)?pfi(?P<index>\d+)$", re.IGNORECASE)
_DIO_RE = re.compile(
    r"^/?(?:(?P<device>Dev\d+)/)?port(?P<port>\d+)/line(?P<line>\d+)$",
    re.IGNORECASE,
)
_CTR_RE = re.compile(r"^(?:(?P<device>Dev\d+)/)?ctr(?P<index>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class DeviceInfo:
    """设备信息的小容器。"""

    name: str
    product_type: str
    serial_num: str


class DaqController:
    """USB-6363 的统一控制入口。

    server 只创建一个 DaqController 实例。
    其他程序都通过 server/client 调用它，避免多个程序直接抢采集卡。
    """

    def __init__(self, device_name: str = DEVICE_NAME) -> None:
        self.device_name = device_name

        # 用于保护短硬件操作，例如 AO、PFI、单点 AI。
        self._hardware_lock = threading.RLock()

        # 用于保护后台 AI 采样状态。
        self._ai_lock = threading.RLock()
        self._ai_active_channels: list[str] = []
        self._ai_rate = 0.0
        self._ai_running = False
        self._ai_error: str | None = None
        self._ai_last_update = 0.0
        self._ai_sample_counts: dict[str, int] = {}
        self._ai_latest: dict[str, float] = {}
        self._ai_buffers: dict[str, deque[float]] = {}
        self._ai_buffer_size = AI_DEFAULT_BUFFER_SIZE

        # record_ai_to_file 打开这个记录器，后台采样线程负责填数据。
        self._ai_recording: dict[str, Any] | None = None

        self._ai_thread: threading.Thread | None = None
        self._ai_stop_event: threading.Event | None = None

    # ---------------------------------------------------------------------
    # 设备信息
    # ---------------------------------------------------------------------
    def list_devices(self) -> list[DeviceInfo]:
        """列出当前 NI-DAQmx 能看到的所有 NI 设备。"""

        with self._hardware_lock:
            return [
                DeviceInfo(
                    name=device["name"],
                    product_type=device["product_type"],
                    serial_num=device["serial_num"],
                )
                for device in nidaqmx_driver.list_devices()
            ]

    def get_device_info(self) -> DeviceInfo:
        """读取当前目标设备的基本信息。"""

        with self._hardware_lock:
            device = nidaqmx_driver.get_device_info(self.device_name)
            return DeviceInfo(
                name=device["name"],
                product_type=device["product_type"],
                serial_num=device["serial_num"],
            )

    def list_signal_terminals(self) -> dict[str, list[str]]:
        """列出 PFI、数字线、计数器等常用端子。"""

        with self._hardware_lock:
            return nidaqmx_driver.list_signal_terminals(self.device_name)

    # ---------------------------------------------------------------------
    # AI 后台连续采样管理
    # ---------------------------------------------------------------------
    def subscribe_ai_channel(self, channel: str) -> dict[str, Any]:
        """订阅一个 AI 通道，让后台开始或继续连续采样。"""

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if physical_channel not in self._ai_active_channels:
                self._ai_active_channels.append(physical_channel)

        self._restart_ai_sampling()
        return self.get_ai_sampling_status()

    def unsubscribe_ai_channel(self, channel: str) -> dict[str, Any]:
        """取消订阅一个 AI 通道。"""

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if physical_channel in self._ai_active_channels:
                self._ai_active_channels.remove(physical_channel)

        self._restart_ai_sampling()
        return self.get_ai_sampling_status()

    def set_ai_channels(self, channels: list[str]) -> dict[str, Any]:
        """一次性设置当前需要连续采样的 AI 通道列表。

        这相当于“暂停其他通道，只保留我指定的这些通道”。
        """

        normalized_channels: list[str] = []
        for channel in channels:
            physical_channel = self._normalize_ai_channel(channel)
            if physical_channel not in normalized_channels:
                normalized_channels.append(physical_channel)

        with self._ai_lock:
            self._ai_active_channels = normalized_channels

        self._restart_ai_sampling()
        return self.get_ai_sampling_status()

    def clear_ai_channels(self) -> dict[str, Any]:
        """取消所有 AI 通道订阅，并停止后台连续采样。"""

        with self._ai_lock:
            self._ai_active_channels = []

        self._restart_ai_sampling()
        return self.get_ai_sampling_status()

    def get_ai_sampling_status(self) -> dict[str, Any]:
        """查询后台 AI 连续采样状态。"""

        with self._ai_lock:
            channels = list(self._ai_active_channels)
            return {
                "running": self._ai_running,
                "channels": channels,
                "channel_count": len(channels),
                "rate_per_channel": self._ai_rate,
                "aggregate_rate": self._ai_rate * len(channels),
                "buffer_size": self._ai_buffer_size,
                "last_update": self._ai_last_update,
                "sample_counts": dict(self._ai_sample_counts),
                "error": self._ai_error,
            }

    def get_ai_latest(self, channel: str) -> dict[str, Any]:
        """读取后台连续采样缓存里某个通道的最近一个值。"""

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if physical_channel not in self._ai_active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")
            if physical_channel not in self._ai_latest:
                raise RuntimeError(f"{physical_channel} has no sampled data yet")

            return {
                "device": self.device_name,
                "channel": physical_channel,
                "rate": self._ai_rate,
                "value": self._ai_latest[physical_channel],
                "last_update": self._ai_last_update,
                "sample_count": self._ai_sample_counts.get(physical_channel, 0),
            }

    def get_ai_buffer(self, channel: str, max_samples: int = 1000) -> dict[str, Any]:
        """读取后台连续采样缓存里某个通道最近的一段数据。"""

        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if physical_channel not in self._ai_active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")

            values = list(self._ai_buffers.get(physical_channel, []))[-max_samples:]
            return {
                "device": self.device_name,
                "channel": physical_channel,
                "rate": self._ai_rate,
                "samples": len(values),
                "values": values,
                "last_update": self._ai_last_update,
                "sample_count": self._ai_sample_counts.get(physical_channel, 0),
            }

    def get_ai_stats(self, channel: str, max_samples: int = 10000) -> dict[str, Any]:
        """返回某个通道最近一段缓存数据的统计量。

        这适合实时监测反馈，不返回完整高速原始数据。
        """

        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if physical_channel not in self._ai_active_channels:
                raise RuntimeError(f"{physical_channel} is not subscribed")

            values = list(self._ai_buffers.get(physical_channel, []))[-max_samples:]
            rate = self._ai_rate
            last_update = self._ai_last_update
            sample_count = self._ai_sample_counts.get(physical_channel, 0)

        if not values:
            raise RuntimeError(f"{physical_channel} has no sampled data yet")

        data = np.asarray(values, dtype=np.float64)
        return {
            "device": self.device_name,
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

    def record_ai_to_file(
        self,
        seconds: float,
        output_dir: str = AI_CAPTURE_OUTPUT_DIR,
        prefix: str = "ai_capture",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """把后台正在采集的 AI 数据记录到 .npy 文件。

        调用前必须先 set_ai_channels/subscribe_ai，让后台采样已经运行。
        """

        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        with self._ai_lock:
            if not self._ai_active_channels or not self._ai_running:
                raise RuntimeError("AI sampling is not running. Call set_ai_channels first.")
            if self._ai_recording is not None:
                raise RuntimeError("Another AI file recording is already running")

            channels = list(self._ai_active_channels)
            rate = self._ai_rate
            samples_per_channel = max(1, int(round(rate * seconds)))
            done_event = threading.Event()
            self._ai_recording = {
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
            with self._ai_lock:
                self._ai_recording = None
            raise TimeoutError("Timed out while recording AI data to file")

        with self._ai_lock:
            recording = self._ai_recording
            self._ai_recording = None

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
            "device": self.device_name,
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

    def read_ai_voltage(
        self,
        channel: str = "ai0",
        samples: int = 1,
        rate: float = 1000.0,
        terminal_config: str = "RSE",
        min_val: float = -10.0,
        max_val: float = 10.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """读取模拟输入 AI 电压。

        如果后台 AI 正在采样，已订阅通道会从缓存返回，避免抢设备。
        """

        if samples < 1:
            raise ValueError("samples must be >= 1")
        if rate <= 0:
            raise ValueError("rate must be > 0")

        physical_channel = self._normalize_ai_channel(channel)
        with self._ai_lock:
            if self._ai_active_channels:
                if physical_channel not in self._ai_active_channels:
                    raise RuntimeError(
                        f"AI sampling is running, but {physical_channel} is not subscribed. "
                        "Subscribe it first or clear AI channels before direct read."
                    )
                if samples == 1:
                    if physical_channel not in self._ai_latest:
                        raise RuntimeError(f"{physical_channel} has no sampled data yet")
                    return {
                        "device": self.device_name,
                        "channel": physical_channel,
                        "samples": 1,
                        "rate": self._ai_rate,
                        "values": self._ai_latest[physical_channel],
                    }

                values = list(self._ai_buffers.get(physical_channel, []))[-samples:]
                return {
                    "device": self.device_name,
                    "channel": physical_channel,
                    "samples": len(values),
                    "rate": self._ai_rate,
                    "values": values,
                }

        with self._hardware_lock:
            values = nidaqmx_driver.read_ai_voltage(
                device_name=self.device_name,
                physical_channel=physical_channel,
                samples=samples,
                rate=rate,
                terminal_config_name=terminal_config,
                min_val=min_val,
                max_val=max_val,
                timeout=timeout,
            )

        return {
            "device": self.device_name,
            "channel": physical_channel,
            "samples": samples,
            "rate": rate,
            "values": values,
        }

    def capture_ai_frame(
        self,
        channels: list[str],
        samples: int = 5000,
        rate: float = 50_000.0,
        terminal_config: str = "DIFF",
        min_val: float = -5.0,
        max_val: float = 5.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """同步采集多路 AI 的一帧数据。

        这个接口是给“双峰锁定”这类程序准备的：
        - channels 例如 ["ai0", "ai1"]。
        - samples 是每个通道读多少个点。
        - rate 是每个通道的采样率。

        注意：它会把数据直接放进 JSON 返回，所以只适合“一帧波形”。
        长时间高速记录请用 record_ai_to_file。
        """

        if not channels:
            raise ValueError("channels must not be empty")
        if samples < 1:
            raise ValueError("samples must be >= 1")
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if min_val >= max_val:
            raise ValueError("min_val must be smaller than max_val")

        physical_channels: list[str] = []
        for channel in channels:
            physical_channel = self._normalize_ai_channel(channel)
            if physical_channel not in physical_channels:
                physical_channels.append(physical_channel)

        total_json_samples = len(physical_channels) * samples
        if total_json_samples > AI_FRAME_MAX_JSON_SAMPLES:
            raise ValueError(
                f"capture_ai_frame would return {total_json_samples} samples as JSON. "
                "Use record_ai_to_file for large or high-speed captures."
            )

        started_at = time.time()
        t0 = time.perf_counter()
        with self._ai_lock:
            if self._ai_active_channels:
                raise RuntimeError(
                    "Background AI sampling is running. Call clear_ai_channels before "
                    "capture_ai_frame so the synchronous frame task can own the AI hardware."
                )

            # 持有 _ai_lock 可以防止另一个请求在本次同步采帧期间启动后台 AI。
            with self._hardware_lock:
                values = nidaqmx_driver.capture_ai_frame(
                    device_name=self.device_name,
                    physical_channels=physical_channels,
                    samples=samples,
                    rate=rate,
                    terminal_config_name=terminal_config,
                    min_val=min_val,
                    max_val=max_val,
                    timeout=timeout,
                )
        duration_seconds = time.perf_counter() - t0

        return {
            "device": self.device_name,
            "channels": physical_channels,
            "channel_count": len(physical_channels),
            "samples_per_channel": samples,
            "rate_per_channel": rate,
            "aggregate_rate": rate * len(physical_channels),
            "terminal_config": terminal_config,
            "min_val": min_val,
            "max_val": max_val,
            "duration_seconds": duration_seconds,
            "started_at": started_at,
            "finished_at": time.time(),
            "values": values,
        }

    # ---------------------------------------------------------------------
    # AO / PFI
    # ---------------------------------------------------------------------
    def write_ao_voltage(
        self,
        channel: str = "ao0",
        value: float = 0.0,
        min_val: float = -10.0,
        max_val: float = 10.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """输出一个静态模拟电压 AO。"""

        if not min_val <= value <= max_val:
            raise ValueError(f"value must be between {min_val} and {max_val}")

        physical_channel = self._normalize_ao_channel(channel)
        with self._hardware_lock:
            nidaqmx_driver.write_ao_voltage(
                device_name=self.device_name,
                physical_channel=physical_channel,
                value=float(value),
                min_val=min_val,
                max_val=max_val,
                timeout=timeout,
            )

        return {
            "device": self.device_name,
            "channel": physical_channel,
            "value": float(value),
        }

    def read_digital_line(self, line: str = "PFI0", timeout: float = 10.0) -> dict[str, Any]:
        """读取一个数字线或 PFI 端子的高低电平。"""

        physical_line = self._normalize_digital_line(line)
        with self._hardware_lock:
            value = nidaqmx_driver.read_digital_line(
                device_name=self.device_name,
                physical_line=physical_line,
                timeout=timeout,
            )

        return {
            "device": self.device_name,
            "line": physical_line,
            "value": value,
        }

    def write_digital_line(
        self,
        line: str = "PFI0",
        value: bool = False,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """写一个数字线或 PFI 端子的高低电平。"""

        physical_line = self._normalize_digital_line(line)
        with self._hardware_lock:
            nidaqmx_driver.write_digital_line(
                device_name=self.device_name,
                physical_line=physical_line,
                value=bool(value),
                timeout=timeout,
            )

        return {
            "device": self.device_name,
            "line": physical_line,
            "value": bool(value),
        }

    def count_pfi_edges(
        self,
        line: str = "PFI0",
        counter: str = "ctr0",
        seconds: float = 1.0,
        edge: str = "RISING",
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """用计数器统计某个 PFI 端子在一段时间内出现了多少个边沿。"""

        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        terminal = self._normalize_pfi_terminal(line)
        physical_counter = self._normalize_counter(counter)
        with self._hardware_lock:
            result = nidaqmx_driver.count_pfi_edges(
                device_name=self.device_name,
                terminal=terminal,
                physical_counter=physical_counter,
                seconds=seconds,
                edge_name=edge,
                timeout=timeout,
            )

        return {
            "device": self.device_name,
            "line": terminal,
            "counter": physical_counter,
            "seconds": seconds,
            "edge": result["edge"],
            "count": result["count"],
        }

    # ---------------------------------------------------------------------
    # 内部 AI 线程
    # ---------------------------------------------------------------------
    def _calculate_ai_rate(self, channels: list[str]) -> float:
        """根据活跃 AI 通道数量计算每个通道的采样率。"""

        channel_count = len(channels)
        if channel_count == 0:
            return 0.0
        if channel_count == 1:
            return AI_SINGLE_CHANNEL_MAX_RATE
        return AI_MULTI_CHANNEL_AGGREGATE_RATE / channel_count

    def _restart_ai_sampling(self) -> None:
        """按当前订阅通道重启后台 AI 连续采样任务。"""

        self._stop_ai_sampling()

        with self._ai_lock:
            channels = list(self._ai_active_channels)
            if not channels:
                self._ai_rate = 0.0
                self._ai_running = False
                self._ai_error = None
                return

            self._ai_rate = self._calculate_ai_rate(channels)
            self._ai_error = None

            for channel in channels:
                self._ai_buffers.setdefault(channel, deque(maxlen=self._ai_buffer_size))
                self._ai_sample_counts.setdefault(channel, 0)

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._ai_sampling_worker,
                args=(channels, self._ai_rate, stop_event),
                daemon=True,
                name="usb6363-ai-sampling",
            )
            self._ai_stop_event = stop_event
            self._ai_thread = thread
            self._ai_running = True
            thread.start()

    def _stop_ai_sampling(self) -> None:
        """停止当前后台 AI 连续采样任务。"""

        with self._ai_lock:
            stop_event = self._ai_stop_event
            thread = self._ai_thread
            if stop_event is not None:
                stop_event.set()

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

        with self._ai_lock:
            if self._ai_thread is thread:
                self._ai_thread = None
                self._ai_stop_event = None
                self._ai_running = False

    def _ai_sampling_worker(
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

                    with self._ai_lock:
                        if stop_event.is_set():
                            break
                        for channel, values in zip(channels, channel_values):
                            if not values:
                                continue
                            self._ai_latest[channel] = values[-1]
                            self._ai_buffers.setdefault(
                                channel,
                                deque(maxlen=self._ai_buffer_size),
                            ).extend(values)
                            self._ai_sample_counts[channel] = (
                                self._ai_sample_counts.get(channel, 0) + len(values)
                            )
                        self._append_ai_recording_locked(channels, channel_values)
                        self._ai_last_update = now
                        self._ai_error = None
            finally:
                task.close()

        except Exception as exc:
            with self._ai_lock:
                self._ai_error = str(exc)
        finally:
            with self._ai_lock:
                if self._ai_thread is threading.current_thread():
                    self._ai_running = False

    def _append_ai_recording_locked(
        self,
        channels: list[str],
        channel_values: list[list[float]],
    ) -> None:
        """把当前数据块追加到文件记录器。

        调用这个函数时外层已经持有 _ai_lock。
        """

        recording = self._ai_recording
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

    # ---------------------------------------------------------------------
    # 通道名校验
    # ---------------------------------------------------------------------
    def _normalize_ai_channel(self, channel: str) -> str:
        return self._normalize_channel(channel, kind="ai")

    def _normalize_ao_channel(self, channel: str) -> str:
        return self._normalize_channel(channel, kind="ao")

    def _normalize_channel(self, channel: str, kind: str) -> str:
        """统一并校验 AI/AO 通道名。"""

        pattern = _AI_RE if kind == "ai" else _AO_RE
        match = pattern.match(channel)
        if match is None:
            raise ValueError(f"Invalid {kind.upper()} channel: {channel!r}")

        device = match.group("device") or self.device_name
        index = int(match.group("index"))
        if device != self.device_name:
            raise ValueError(f"Only {self.device_name} is allowed, got {device}")

        max_index = 31 if kind == "ai" else 3
        if index > max_index:
            raise ValueError(f"{kind.upper()} channel index must be 0-{max_index}")

        return f"{self.device_name}/{kind}{index}"

    def _normalize_digital_line(self, line: str) -> str:
        """统一并校验数字线或 PFI 端子名。"""

        pfi_match = _PFI_RE.match(line)
        if pfi_match is not None:
            return self._normalize_pfi_terminal(line).lstrip("/")

        dio_match = _DIO_RE.match(line)
        if dio_match is None:
            raise ValueError(f"Invalid digital line: {line!r}")

        device = dio_match.group("device") or self.device_name
        port = int(dio_match.group("port"))
        line_index = int(dio_match.group("line"))

        if device != self.device_name:
            raise ValueError(f"Only {self.device_name} is allowed, got {device}")
        if port == 0:
            max_line = 31
        elif port in (1, 2):
            max_line = 7
        else:
            raise ValueError("Digital port must be 0, 1, or 2")
        if line_index > max_line:
            raise ValueError(f"port{port} line index must be 0-{max_line}")

        return f"{self.device_name}/port{port}/line{line_index}"

    def _normalize_pfi_terminal(self, line: str) -> str:
        """统一并校验 PFI 端子名，返回 /Dev2/PFIx。"""

        match = _PFI_RE.match(line)
        if match is None:
            raise ValueError(f"Invalid PFI terminal: {line!r}")

        device = match.group("device") or self.device_name
        index = int(match.group("index"))

        if device != self.device_name:
            raise ValueError(f"Only {self.device_name} is allowed, got {device}")
        if index > 15:
            raise ValueError("PFI index must be 0-15")

        return f"/{self.device_name}/PFI{index}"

    def _normalize_counter(self, counter: str) -> str:
        """统一并校验计数器名，USB-6363 有 ctr0-ctr3。"""

        match = _CTR_RE.match(counter)
        if match is None:
            raise ValueError(f"Invalid counter: {counter!r}")

        device = match.group("device") or self.device_name
        index = int(match.group("index"))

        if device != self.device_name:
            raise ValueError(f"Only {self.device_name} is allowed, got {device}")
        if index > 3:
            raise ValueError("Counter index must be 0-3")

        return f"{self.device_name}/ctr{index}"


__all__ = ["DaqController", "DEVICE_NAME", "DeviceInfo"]
