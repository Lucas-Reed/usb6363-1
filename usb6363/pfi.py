"""PFI、数字线和计数器功能。"""

from __future__ import annotations

from typing import Any

from usb6363 import nidaqmx_driver
from usb6363.device import DeviceContext


class PfiController:
    """负责 PFI/数字线读写，以及 PFI 边沿计数。"""

    def __init__(self, device: DeviceContext) -> None:
        self.device = device

    def read_digital_line(self, line: str = "PFI0", timeout: float = 10.0) -> dict[str, Any]:
        """读取一个数字线或 PFI 端子的高低电平。"""

        physical_line = self.device.normalize_digital_line(line)

        with self.device.lock:
            value = nidaqmx_driver.read_digital_line(
                device_name=self.device.device_name,
                physical_line=physical_line,
                timeout=timeout,
            )

        return {
            "device": self.device.device_name,
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

        physical_line = self.device.normalize_digital_line(line)

        with self.device.lock:
            nidaqmx_driver.write_digital_line(
                device_name=self.device.device_name,
                physical_line=physical_line,
                value=bool(value),
                timeout=timeout,
            )

        return {
            "device": self.device.device_name,
            "line": physical_line,
            "value": bool(value),
        }

    def count_edges(
        self,
        line: str = "PFI0",
        counter: str = "ctr0",
        seconds: float = 1.0,
        edge_name: str = "RISING",
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """用计数器统计某个 PFI 端子在一段时间内出现了多少个边沿。"""

        if seconds <= 0:
            raise ValueError("seconds must be > 0")

        terminal = self.device.normalize_pfi_terminal(line)
        physical_counter = self.device.normalize_counter(counter)

        with self.device.lock:
            result = nidaqmx_driver.count_pfi_edges(
                device_name=self.device.device_name,
                terminal=terminal,
                physical_counter=physical_counter,
                seconds=seconds,
                edge_name=edge_name,
                timeout=timeout,
            )

        return {
            "device": self.device.device_name,
            "line": terminal,
            "counter": physical_counter,
            "seconds": seconds,
            "edge": result["edge"],
            "count": result["count"],
        }
