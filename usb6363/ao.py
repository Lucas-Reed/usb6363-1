"""AO 模拟输出功能。"""

from __future__ import annotations

from typing import Any

from usb6363.device import DeviceContext
from usb6363 import nidaqmx_driver


class AoController:
    """负责 USB-6363 的模拟输出 AO。"""

    def __init__(self, device: DeviceContext) -> None:
        self.device = device

    def write_voltage(
        self,
        channel: str = "ao0",
        value: float = 0.0,
        min_val: float = -10.0,
        max_val: float = 10.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """输出一个静态模拟电压 AO。

        注意：这个函数会真实改变对应 AO 端口电压。
        """

        if not min_val <= value <= max_val:
            raise ValueError(f"value must be between {min_val} and {max_val}")

        physical_channel = self.device.normalize_ao_channel(channel)

        with self.device.lock:
            nidaqmx_driver.write_ao_voltage(
                device_name=self.device.device_name,
                physical_channel=physical_channel,
                value=float(value),
                min_val=min_val,
                max_val=max_val,
                timeout=timeout,
            )

        return {
            "device": self.device.device_name,
            "channel": physical_channel,
            "value": float(value),
        }
