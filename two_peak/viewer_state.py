"""双峰波形查看器的运行状态。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from two_peak.ao_scan_calibrator import AoScanCalibrator
from two_peak.config import TwoPeakSettings
from two_peak.eom_identification import CalibrationModel, identify_eom_aom_spectrum
from two_peak.power_lock import PowerLockController
from two_peak.pfi1_feedback import Pfi1FeedbackController
from two_peak.sync_test import SyncTestCoordinator
from two_peak.scan_source import RigolScanSource, plan_centering
from two_peak.trend_logger import AreaTrendLogger
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
        # 用户在 WebUI 里点击“保存为默认值”后，会写到这个 JSON 文件。
        # 它放在 data/ 下面，属于实验运行时配置，不进入 git。
        self.defaults_path = sample_dir.parent / "two_peak_defaults.json"
        self.calibration_path = sample_dir.parent / "eom_aom_calibration.json"
        self.user_defaults = self._load_user_defaults()
        self.identification_settings: dict[str, Any] = {}
        self.calibration_model = self._load_calibration()
        self.latest_frame: dict[str, Any] | None = None
        self.latest_measurement: dict[str, Any] | None = None
        self.latest_eom_identification: dict[str, Any] | None = None
        self.scan_source = RigolScanSource()
        self.scan_centering_proposal: dict[str, Any] | None = None
        # 慢漂记录器会在后端线程里读取底层 frame_stream 最新帧并写 CSV。
        # 它不依赖浏览器是否一直打开。
        self.trend_logger = AreaTrendLogger(
            daq=self.daq,
            output_dir=sample_dir.parent / "two_peak_trends",
        )
        # AO 扫描标定器读取上面的面积慢漂统计，并通过 daq client 写 AO。
        # 它不直接访问 NI-DAQmx，因此不会破坏“只有 8765 底层服务碰硬件”的边界。
        self.ao_scan_calibrator = AoScanCalibrator(
            daq=self.daq,
            trend_logger=self.trend_logger,
            output_dir=sample_dir.parent / "ao_scan_calibrations",
        )
        # 双路慢速功率锁定器读取面积慢漂统计，并通过 daq client 写 AO。
        # 它只在用户点击“启动锁定”后运行，默认不会自动闭环。
        self.power_lock = PowerLockController(
            daq=self.daq,
            trend_logger=self.trend_logger,
            # 每次锁定单独记录采样峰、锁定峰、实际比值、PI 误差和 AO 输出，
            # 便于实验结束后判断稳定效果以及排查达到限幅的时段。
            output_dir=sample_dir.parent / "power_lock_runs",
        )
        # PFI1 后续窗口反馈是可选的独立消费者，默认不运行，不影响已有双峰锁定。
        self.pfi1_feedback = Pfi1FeedbackController(self.daq)
        # 临时同步测试只协调现有记录器，不直接接触采集卡。
        self.sync_test = SyncTestCoordinator(
            daq=self.daq,
            trend_logger=self.trend_logger,
            output_dir=sample_dir.parent / "sync_tests",
        )

    def factory_web_defaults(self) -> dict[str, Any]:
        """返回 WebUI 可以直接使用的出厂默认值。

        TwoPeakSettings 里保留的是比较结构化的实验参数；
        WebUI 的 input id 更扁平，所以这里额外给出一份方便前端填表的字段。
        """

        parameters = self.settings.to_web_parameters()
        peak_indices = parameters["peak_indices"]
        parameters.update(
            {
                "channels": ", ".join(parameters["ai_channels"]),
                "rate": parameters["sample_rate"],
                "samples": parameters["samples_per_frame"],
                "min_val": parameters["ai_min_val"],
                "max_val": parameters["ai_max_val"],
                "timeout": self.settings.daq.timeout,
                "trigger_enabled": False,
                "trigger_source": "PFI0",
                "trigger_edge": "RISING",
                "peak0": peak_indices[0],
                "peak1": peak_indices[1],
                "analysis_channel_index": 0,
                "search_window_half": parameters["window_half"],
                "measure_half": parameters["peak_avg_half"],
                # 双峰慢漂与逐帧 NPZ 默认记录两小时；WebUI 填 0 可恢复为不限时。
                "duration_minutes": 120.0,
                "top_percent": 10.0,
                "record_full_frame": False,
            }
        )
        return parameters

    def active_web_defaults(self) -> dict[str, Any]:
        """返回当前启动时实际使用的默认值。

        如果用户保存过默认值，就在出厂默认值上覆盖用户默认值；
        这样以后新增字段时，旧的 JSON 文件也不会缺字段。
        """

        parameters = self.factory_web_defaults()
        parameters.update(self.user_defaults)
        parameters["defaults_source"] = "user" if self.user_defaults else "factory"
        parameters["defaults_file"] = str(self.defaults_path.resolve())
        return parameters

    def save_user_defaults(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """把当前 WebUI 参数保存为下次启动时的默认值。"""

        self.defaults_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_defaults = dict(parameters)
        self.defaults_path.write_text(
            json.dumps(self.user_defaults, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.active_web_defaults()

    def reset_user_defaults(self) -> dict[str, Any]:
        """删除用户默认值，恢复到代码里的出厂默认值。"""

        self.user_defaults = {}
        if self.defaults_path.exists():
            self.defaults_path.unlink()
        return self.active_web_defaults()

    def _load_user_defaults(self) -> dict[str, Any]:
        """启动查看器时读取用户默认值文件。"""

        if not self.defaults_path.exists():
            return {}
        try:
            data = json.loads(self.defaults_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def calibration_status(self) -> dict[str, Any]:
        """返回当前 EOM/AOM 标定，不包含任何硬件控制状态。"""

        return {
            "configured": self.calibration_model is not None,
            "model": self.calibration_model.to_dict() if self.calibration_model else None,
            "identification_settings": self.identification_settings,
            "file": str(self.calibration_path.resolve()),
        }

    def save_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        """保存用户手动确认的标定模型。"""

        model_payload = payload.get("model", payload)
        model_data = dict(model_payload)
        if not model_data.get("breakpoints"):
            count = _frame_sample_count(self.latest_frame) or 10000
            model_data["breakpoints"] = [count // 4, 3 * count // 4]
        model_data["updated_at"] = time.time()
        model = CalibrationModel.from_dict(model_data)
        sample_count = _frame_sample_count(self.latest_frame)
        model.validate(sample_count=sample_count)
        self.identification_settings = dict(payload.get("identification_settings") or {})
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(
            json.dumps(dict(model.to_dict(), identification_settings=self.identification_settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.calibration_model = model
        return self.calibration_status()

    def reset_calibration(self) -> dict[str, Any]:
        self.calibration_model = None
        self.identification_settings = {}
        if self.calibration_path.exists():
            self.calibration_path.unlink()
        return self.calibration_status()

    def calibration_candidates(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """基于标定模型返回候选边带；该接口不会启动锁定。"""

        if self.calibration_model is None:
            raise ValueError("no EOM/AOM calibration model has been saved")
        body = payload or {}
        sample_count = body.get("sample_count")
        if sample_count in (None, ""):
            sample_count = _frame_sample_count(self.latest_frame)
        sample_count = None if sample_count in (None, "") else int(sample_count)
        orders = body.get("orders", (-2, -1, 1, 2))
        if orders in (None, ""):
            orders = (-2, -1, 1, 2)
        if isinstance(orders, str):
            orders = [item.strip() for item in orders.split(",") if item.strip()]
        candidates = self.calibration_model.candidate_sidebands(
            sample_count=sample_count,
            orders=orders,
        )
        return {
            "ok": True,
            "model": self.calibration_model.to_dict(),
            "sample_count": sample_count,
            "candidates": candidates,
            "carrier": {
                "kind": "carrier",
                "label": "EOM carrier",
                "index": self.calibration_model.carrier_index,
                "frequency_offset_mhz": 0.0,
                "in_range": sample_count is None or 0 <= self.calibration_model.carrier_index < sample_count,
            },
            "aom_zero": {
                "kind": "aom_zero",
                "label": "AOM zero",
                "index": self.calibration_model.aom_zero_index,
                "frequency_offset_mhz": self.calibration_model.frequency_difference_mhz,
                "in_range": sample_count is None or 0 <= self.calibration_model.aom_zero_index < sample_count,
            },
            "display_only": True,
        }

    def identify_eom_aom(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run width-filtered automatic identification on the latest waveform."""

        frame = self.latest_frame
        if not frame or not frame.get("values"):
            raise RuntimeError("capture or load a waveform before automatic identification")
        channel_index = int(payload.get("analysis_channel_index", 0))
        values = frame["values"]
        if channel_index < 0 or channel_index >= len(values):
            raise ValueError("analysis_channel_index is out of range")
        raw_breakpoints = payload.get("breakpoints")
        if isinstance(raw_breakpoints, str):
            raw_breakpoints = [item.strip() for item in raw_breakpoints.split(",") if item.strip()]
        breakpoints = tuple(int(item) for item in raw_breakpoints) if raw_breakpoints else None
        result = identify_eom_aom_spectrum(
            values[channel_index],
            spacing_samples=float(payload.get("spacing_samples", 400.0)),
            spacing_mhz=float(payload.get("spacing_mhz", 190.0)),
            spacing_tolerance_samples=float(payload.get("spacing_tolerance_samples", 3.0)),
            eom_frequency_mhz=float(payload.get("eom_frequency_mhz", 6800.0)),
            fsr_mhz=float(payload.get("fsr_mhz", 2500.0)),
            breakpoints=breakpoints,  # type: ignore[arg-type]
            max_eom_order=int(payload.get("max_eom_order", 4)),
            match_tolerance_samples=float(payload.get("match_tolerance_samples", 3.0)),
            min_peak_width=int(payload["min_peak_width"]) if payload.get("min_peak_width") else None,
            max_peak_width=int(payload["max_peak_width"]) if payload.get("max_peak_width") else None,
        )
        result["frame_id"] = frame.get("frame_id")
        result["analysis_channel_index"] = channel_index
        result["analysis_channel"] = frame.get("channels", [None] * len(values))[channel_index]
        self.latest_eom_identification = result
        self.scan_centering_proposal = None
        return result

    def preview_scan_centering(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.scan_centering_proposal = None
        if not self.latest_eom_identification:
            raise ValueError("Identify the current waveform first")
        current = self.scan_source.read()
        proposal = plan_centering(
            self.latest_eom_identification, current,
            resize=bool(payload.get("resize", False)),
            separation_fraction=float(payload.get("separation_fraction", 0.4)),
            min_voltage=float(payload.get("min_voltage", 0.0)),
            max_voltage=float(payload.get("max_voltage", 5.0)),
        )
        self.scan_centering_proposal = proposal
        return {"current": current, "proposal": proposal}

    def apply_scan_centering(self) -> dict[str, Any]:
        proposal = self.scan_centering_proposal
        if not proposal:
            raise ValueError("Preview the centering settings first")
        self.scan_centering_proposal = None
        result = self.scan_source.apply(proposal)
        self.latest_eom_identification = None
        return result

    def _load_calibration(self) -> CalibrationModel | None:
        if not self.calibration_path.exists():
            return None
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            self.identification_settings = dict(payload.get("identification_settings") or {})
            return CalibrationModel.from_dict(payload)
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None


def _frame_sample_count(frame: dict[str, Any] | None) -> int | None:
    if not frame:
        return None
    values = frame.get("values")
    if isinstance(values, list) and values and isinstance(values[0], list):
        return len(values[0])
    value = frame.get("samples_per_channel")
    return int(value) if value not in (None, "") else None
