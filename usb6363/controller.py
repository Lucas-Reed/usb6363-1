"""统一控制器入口。

DaqController 对外保持原来的方法名，内部把工作分发给：
    DeviceContext
    AiManager
    AoController
    PfiController
"""

from __future__ import annotations

from typing import Any

from usb6363.ai import AiManager
from usb6363.ao import AoController
from usb6363.device import DEVICE_NAME, DeviceContext, DeviceInfo
from usb6363.pfi import PfiController


class DaqController:
    """USB-6363 控制器门面。

    server 只需要使用这个类，不需要知道内部拆成了几个模块。
    """

    def __init__(self, device_name: str = DEVICE_NAME) -> None:
        self.device = DeviceContext(device_name=device_name)
        self.ai = AiManager(self.device)
        self.ao = AoController(self.device)
        self.pfi = PfiController(self.device)

    @property
    def device_name(self) -> str:
        """当前 NI-DAQmx 设备名，例如 Dev2。"""

        return self.device.device_name

    def list_devices(self) -> list[DeviceInfo]:
        return self.device.list_devices()

    def get_device_info(self) -> DeviceInfo:
        return self.device.get_device_info()

    def list_signal_terminals(self) -> dict[str, list[str]]:
        return self.device.list_signal_terminals()

    def subscribe_ai_channel(self, channel: str) -> dict[str, Any]:
        return self.ai.subscribe_channel(channel)

    def unsubscribe_ai_channel(self, channel: str) -> dict[str, Any]:
        return self.ai.unsubscribe_channel(channel)

    def set_ai_channels(self, channels: list[str]) -> dict[str, Any]:
        return self.ai.set_channels(channels)

    def clear_ai_channels(self) -> dict[str, Any]:
        return self.ai.clear_channels()

    def get_ai_sampling_status(self) -> dict[str, Any]:
        return self.ai.status()

    def get_ai_latest(self, channel: str) -> dict[str, Any]:
        return self.ai.latest(channel)

    def get_ai_buffer(self, channel: str, max_samples: int = 1000) -> dict[str, Any]:
        return self.ai.buffer(channel, max_samples=max_samples)

    def get_ai_stats(self, channel: str, max_samples: int = 10000) -> dict[str, Any]:
        return self.ai.stats(channel, max_samples=max_samples)

    def record_ai_to_file(
        self,
        seconds: float,
        output_dir: str = "data",
        prefix: str = "ai_capture",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.ai.record_to_file(
            seconds=seconds,
            output_dir=output_dir,
            prefix=prefix,
            timeout=timeout,
        )

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
        return self.ai.read_voltage(
            channel=channel,
            samples=samples,
            rate=rate,
            terminal_config_name=terminal_config,
            min_val=min_val,
            max_val=max_val,
            timeout=timeout,
        )

    def write_ao_voltage(
        self,
        channel: str = "ao0",
        value: float = 0.0,
        min_val: float = -10.0,
        max_val: float = 10.0,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        return self.ao.write_voltage(
            channel=channel,
            value=value,
            min_val=min_val,
            max_val=max_val,
            timeout=timeout,
        )

    def read_digital_line(self, line: str = "PFI0", timeout: float = 10.0) -> dict[str, Any]:
        return self.pfi.read_digital_line(line=line, timeout=timeout)

    def write_digital_line(
        self,
        line: str = "PFI0",
        value: bool = False,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        return self.pfi.write_digital_line(line=line, value=value, timeout=timeout)

    def count_pfi_edges(
        self,
        line: str = "PFI0",
        counter: str = "ctr0",
        seconds: float = 1.0,
        edge: str = "RISING",
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        return self.pfi.count_edges(
            line=line,
            counter=counter,
            seconds=seconds,
            edge_name=edge,
            timeout=timeout,
        )
