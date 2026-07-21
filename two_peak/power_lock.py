"""双路慢速功率锁定器。

本模块负责把面积慢漂记录里的 A/B 峰面积，转换成两个 AO 输出的慢速 PI 修正。

重要边界：
- 不直接 import nidaqmx。
- 只通过 Usb6363Client 写 AO。
- 只读取 AreaTrendLogger 已经算好的慢漂统计值，不自己采集波形。
- 第一版只做慢速 PI，不做 D 项，不追逐 shot-to-shot 高频噪声。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from two_peak.trend_logger import AreaTrendLogger
from usb6363_client import Usb6363Client


class PowerLockController:
    """后台双路 PI 锁定器。"""

    def __init__(self, daq: Usb6363Client, trend_logger: AreaTrendLogger) -> None:
        self._daq = daq
        self._trend_logger = trend_logger
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._error: str | None = None
        self._settings: dict[str, Any] = {}
        self._controllers: list[dict[str, Any]] = []
        self._states: list[dict[str, Any]] = []
        self._iterations = 0
        self._last_update = 0.0

    def start(self, controllers: list[dict[str, Any]], update_s: float = 1.0) -> dict[str, Any]:
        """启动慢速 PI 锁定。

        controllers 是前端传来的两路配置。每一路至少包含：
        channel, feedback_field, target, initial_voltage, min_voltage, max_voltage,
        direction, max_step_v, kp, ki。
        """

        if update_s <= 0:
            raise ValueError("update_s must be > 0")

        validated: list[dict[str, Any]] = []
        for controller in controllers:
            normalized = _validate_controller(controller)
            if normalized is not None:
                validated.append(normalized)
        if not validated:
            raise ValueError("controllers must contain at least one enabled controller")

        trend_status = self._trend_logger.status()
        if trend_status.get("running") is not True:
            raise RuntimeError("请先开始面积慢漂记录，再启动功率锁定")

        with self._lock:
            if self._running:
                raise RuntimeError("power lock is already running")

            states = []
            for controller in validated:
                states.append(
                    {
                        "name": controller["name"],
                        "channel": controller["channel"],
                        "feedback_field": controller["feedback_field"],
                        "target": controller["target"],
                        "voltage": controller["initial_voltage"],
                        "integral": 0.0,
                        "measured": None,
                        "relative_error": None,
                        "command_delta": 0.0,
                        "limited": False,
                    }
                )

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._worker,
                args=(validated, float(update_s), stop_event),
                daemon=True,
                name="two-peak-power-lock",
            )
            self._stop_event = stop_event
            self._thread = thread
            self._running = True
            self._error = None
            self._settings = {"update_s": float(update_s)}
            self._controllers = [dict(controller) for controller in validated]
            self._states = states
            self._iterations = 0
            self._last_update = 0.0
            thread.start()

        return self.status()

    def stop(self) -> dict[str, Any]:
        """停止 PI 锁定。停止时不改 AO，保持最后输出。"""

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

        return self.status()

    def status(self) -> dict[str, Any]:
        """返回 WebUI 需要显示的锁定状态。"""

        with self._lock:
            return {
                "running": self._running,
                "error": self._error,
                "settings": dict(self._settings),
                "controllers": [dict(controller) for controller in self._controllers],
                "states": [dict(state) for state in self._states],
                "iterations": self._iterations,
                "last_update": self._last_update,
            }

    def _set_error(self, message: str) -> None:
        """记录后台线程里可以恢复的错误，方便前端显示。"""

        with self._lock:
            self._error = message

    def _worker(
        self,
        controllers: list[dict[str, Any]],
        update_s: float,
        stop_event: threading.Event,
    ) -> None:
        """锁定线程主体。"""

        try:
            # 启动锁定时先写一次初始电压，确保软件状态和硬件状态一致。
            for controller in controllers:
                self._daq.write_ao(
                    channel=controller["channel"],
                    value=controller["initial_voltage"],
                    min_val=controller["min_voltage"],
                    max_val=controller["max_voltage"],
                )

            last_time = time.time()
            while not stop_event.is_set():
                now = time.time()
                dt = max(1e-6, now - last_time)
                last_time = now

                trend_status = self._trend_logger.status()
                latest = trend_status.get("latest_stats") or {}
                if trend_status.get("running") is not True:
                    self._set_error("area trend logger is not running")
                    _sleep_until_stop(stop_event, update_s)
                    continue

                next_states: list[dict[str, Any]] = []
                for index, controller in enumerate(controllers):
                    previous = self._states[index] if index < len(self._states) else {}
                    state = self._update_one_controller(controller, previous, latest, dt)
                    next_states.append(state)

                with self._lock:
                    self._states = next_states
                    self._iterations += 1
                    self._last_update = time.time()
                    self._error = None

                _sleep_until_stop(stop_event, update_s)

        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False
                self._thread = None
                self._stop_event = None

    def _update_one_controller(
        self,
        controller: dict[str, Any],
        previous: dict[str, Any],
        latest: dict[str, Any],
        dt: float,
    ) -> dict[str, Any]:
        """根据某一路的反馈值更新一路 AO。"""

        field = controller["feedback_field"]
        measured = _read_feedback_value(latest, field)
        if measured is None:
            raise RuntimeError(f"feedback field {field!r} is not available")

        target = float(controller["target"])
        if abs(target) < 1e-30:
            raise RuntimeError(f"{controller['name']} target is too close to zero")

        old_voltage = float(previous.get("voltage", controller["initial_voltage"]))
        integral = float(previous.get("integral", 0.0))
        relative_error = (target - measured) / abs(target)
        integral += relative_error * dt

        direction = float(controller["direction"])
        raw_delta = direction * (
            float(controller["kp"]) * relative_error
            + float(controller["ki"]) * integral
        )
        step = _clamp(raw_delta, -abs(controller["max_step_v"]), abs(controller["max_step_v"]))
        requested_voltage = old_voltage + step
        voltage = _clamp(requested_voltage, controller["min_voltage"], controller["max_voltage"])
        limited = abs(voltage - requested_voltage) > 1e-12

        # 如果已经顶到边界，并且积分还在继续把输出往边界外推，就撤销本次积分，
        # 这是最简单的抗积分饱和处理。
        if limited:
            integral -= relative_error * dt

        self._daq.write_ao(
            channel=controller["channel"],
            value=voltage,
            min_val=controller["min_voltage"],
            max_val=controller["max_voltage"],
        )

        return {
            "name": controller["name"],
            "channel": controller["channel"],
            "feedback_field": field,
            "target": target,
            "measured": measured,
            "relative_error": relative_error,
            "voltage": voltage,
            "command_delta": voltage - old_voltage,
            "integral": integral,
            "limited": limited,
        }


def _validate_controller(controller: dict[str, Any]) -> dict[str, Any] | None:
    """校验并标准化一路控制参数。"""

    if controller.get("enabled", True) is False:
        return None

    normalized = {
        "name": str(controller.get("name", "")),
        "channel": str(controller.get("channel", "")).strip(),
        "feedback_field": str(controller.get("feedback_field", "")).strip(),
        "target": float(controller.get("target")),
        "initial_voltage": float(controller.get("initial_voltage")),
        "min_voltage": float(controller.get("min_voltage")),
        "max_voltage": float(controller.get("max_voltage")),
        "direction": float(controller.get("direction")),
        "max_step_v": float(controller.get("max_step_v")),
        "kp": float(controller.get("kp")),
        "ki": float(controller.get("ki")),
    }

    name = normalized["name"] or normalized["channel"] or "controller"
    if not normalized["channel"]:
        raise ValueError(f"{name} channel is empty")
    if not normalized["feedback_field"]:
        raise ValueError(f"{name} feedback_field is empty")
    if normalized["min_voltage"] >= normalized["max_voltage"]:
        raise ValueError(f"{name} min_voltage must be smaller than max_voltage")
    if normalized["initial_voltage"] < normalized["min_voltage"]:
        raise ValueError(f"{name} initial_voltage is below min_voltage")
    if normalized["initial_voltage"] > normalized["max_voltage"]:
        raise ValueError(f"{name} initial_voltage is above max_voltage")
    if normalized["direction"] not in (-1.0, 1.0):
        raise ValueError(f"{name} direction must be -1 or 1")
    if normalized["max_step_v"] < 0:
        raise ValueError(f"{name} max_step_v must be >= 0")
    return normalized


def _read_feedback_value(latest: dict[str, Any], field: str) -> float | None:
    """读取反馈字段。EMA 关闭时自动退回到对应的均值字段。"""

    value = latest.get(field)
    if value is None:
        fallback = {
            "area_ema": "area_mean",
            "area2_ema": "area2_mean",
            "area_sum_ema": "area_sum_mean",
            # Top 与面积沿用相同约定：EMA alpha=0 时，锁定器退回到
            # 最近 N 帧的 Top 均值，而不是因为滤波关闭而失去反馈。
            "top_ema": "top_mean",
            "top2_ema": "top2_mean",
        }.get(field)
        if fallback is not None:
            value = latest.get(fallback)
    if value is None:
        return None
    return float(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sleep_until_stop(stop_event: threading.Event, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if stop_event.is_set():
            return True
        time.sleep(min(0.05, max(0.0, deadline - time.time())))
    return stop_event.is_set()
