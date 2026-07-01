"""USB-6363 底层控制兼容入口。

这个文件现在故意保持很薄：
    旧代码仍然可以 `from usb6363_core import DaqController`
    真正实现已经拆到 usb6363/ 包里。

主要实现位置：
    usb6363/device.py      设备信息和通道名校验
    usb6363/ai.py          AI 采样、缓存、统计、写文件
    usb6363/ao.py          AO 模拟输出
    usb6363/pfi.py         PFI/数字线/计数器
    usb6363/controller.py  统一入口 DaqController
"""

from __future__ import annotations

from usb6363 import DEVICE_NAME, DaqController, DeviceInfo

__all__ = ["DaqController", "DEVICE_NAME", "DeviceInfo"]
