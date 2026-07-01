"""设备信息和通道名校验工具。"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from usb6363 import nidaqmx_driver


DEVICE_NAME = "Dev2"


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


class DeviceContext:
    """设备上下文，保存设备名和共用硬件锁。"""

    def __init__(self, device_name: str = DEVICE_NAME) -> None:
        self.device_name = device_name
        self.lock = threading.RLock()

    def list_devices(self) -> list[DeviceInfo]:
        """列出当前 NI-DAQmx 能看到的所有 NI 设备。"""

        with self.lock:
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

        with self.lock:
            device = nidaqmx_driver.get_device_info(self.device_name)
            return DeviceInfo(
                name=device["name"],
                product_type=device["product_type"],
                serial_num=device["serial_num"],
            )

    def list_signal_terminals(self) -> dict[str, list[str]]:
        """列出 PFI、数字线、计数器等常用端子。"""

        with self.lock:
            return nidaqmx_driver.list_signal_terminals(self.device_name)

    def get_device(self) -> Any:
        """确认目标设备存在，并返回 NI-DAQmx 的设备对象。"""

        device_names = [device["name"] for device in nidaqmx_driver.list_devices()]
        if self.device_name not in device_names:
            raise RuntimeError(
                f"{self.device_name!r} was not found in NI-DAQmx. "
                "Check NI MAX / NI-DAQmx device name and USB connection."
            )
        return self.device_name

    def normalize_ai_channel(self, channel: str) -> str:
        """把 AI 通道名统一成 Dev2/aiX。"""

        return self.normalize_channel(channel, kind="ai")

    def normalize_ao_channel(self, channel: str) -> str:
        """把 AO 通道名统一成 Dev2/aoX。"""

        return self.normalize_channel(channel, kind="ao")

    def normalize_channel(self, channel: str, kind: str) -> str:
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

    def normalize_digital_line(self, line: str) -> str:
        """统一并校验数字线或 PFI 端子名。"""

        pfi_match = _PFI_RE.match(line)
        if pfi_match is not None:
            return self.normalize_pfi_terminal(line).lstrip("/")

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

    def normalize_pfi_terminal(self, line: str) -> str:
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

    def normalize_counter(self, counter: str) -> str:
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
