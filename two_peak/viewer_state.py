"""双峰波形查看器的运行状态。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from two_peak.config import TwoPeakSettings
from usb6363_client import Usb6363Client


class ViewerState:
    """保存查看器自己的内存状态。

    这里保存最近采到的一帧，方便 WebUI 点击“保存样本”时不用重新采集。
    这不是采集卡状态，也不会直接访问 NI-DAQmx。
    """

    def __init__(self, api_base_url: str, sample_dir: Path) -> None:
        self.daq = Usb6363Client(base_url=api_base_url)
        self.sample_dir = sample_dir
        self.settings = TwoPeakSettings.defaults()
        self.latest_frame: dict[str, Any] | None = None
        self.latest_measurement: dict[str, Any] | None = None
