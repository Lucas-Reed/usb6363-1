"""USB-6363 底层控制包。

拆分后的结构：
    device.py      设备信息和通道名校验
    ai.py          AI 采样、缓存、统计、写文件
    ao.py          AO 模拟输出
    pfi.py         PFI/数字线/计数器
    controller.py 统一入口
"""

from usb6363.controller import DaqController
from usb6363.device import DEVICE_NAME, DeviceInfo

__all__ = ["DaqController", "DEVICE_NAME", "DeviceInfo"]
