"""双路功率锁定反馈字段读取的无硬件测试。"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

from two_peak.power_lock import PowerLockController, _read_feedback_value


class _FakeDaq:
    """记录 AO 写入但不接触真实 USB-6363。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.writes: list[dict[str, Any]] = []

    def write_ao(self, channel: str, value: float, min_val: float, max_val: float) -> dict[str, Any]:
        row = {
            "channel": channel,
            "value": float(value),
            "min_val": float(min_val),
            "max_val": float(max_val),
        }
        with self._lock:
            self.writes.append(row)
        return row

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.writes]


class _FakeTrendLogger:
    """始终提供一个有效面积反馈值。"""

    def status(self) -> dict[str, Any]:
        return {
            "running": True,
            "latest_stats": {"area_ema": 8.0},
        }


def _controller(**overrides: Any) -> dict[str, Any]:
    """生成一路可用于测试的完整 PI 配置。"""

    result = {
        "name": "EOM",
        "channel": "ao0",
        "feedback_field": "area_ema",
        "target": 10.0,
        "initial_voltage": 2.0,
        "min_voltage": 1.0,
        "max_voltage": 3.5,
        "direction": 1.0,
        "max_step_v": 0.1,
        "kp": 1.0,
        "ki": 0.0,
        "enabled": True,
    }
    result.update(overrides)
    return result


def _wait_until(predicate: Any, timeout: float = 2.0) -> None:
    """等待后台 PI 线程达到测试状态。"""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("等待功率锁定线程超时")


class PowerLockFeedbackTests(unittest.TestCase):
    def test_reads_top_feedback_for_both_windows(self) -> None:
        """A/B 两个窗口的 Top EMA 都应能直接作为锁定反馈。"""

        latest = {
            "top_ema": 0.041,
            "top2_ema": 0.026,
        }

        self.assertAlmostEqual(_read_feedback_value(latest, "top_ema"), 0.041)
        self.assertAlmostEqual(_read_feedback_value(latest, "top2_ema"), 0.026)

    def test_top_ema_falls_back_to_recent_mean_when_disabled(self) -> None:
        """EMA关闭后，保持和面积反馈一致的自动退回行为。"""

        latest = {
            "top_ema": None,
            "top_mean": 0.042,
            "top2_ema": None,
            "top2_mean": 0.027,
        }

        self.assertAlmostEqual(_read_feedback_value(latest, "top_ema"), 0.042)
        self.assertAlmostEqual(_read_feedback_value(latest, "top2_ema"), 0.027)

    def test_missing_top_feedback_returns_none(self) -> None:
        self.assertIsNone(_read_feedback_value({}, "top_ema"))
        self.assertIsNone(_read_feedback_value({}, "top2_ema"))


class PowerLockRuntimeUpdateTests(unittest.TestCase):
    def test_measurement_window_revision_resets_integral(self) -> None:
        """面积边界改变后，PI 只能从新测量口径重新累计积分。"""

        daq = _FakeDaq()
        lock = PowerLockController(daq, _FakeTrendLogger())  # type: ignore[arg-type]
        state = lock._update_one_controller(
            _controller(kp=0.0, ki=1.0),
            {
                "voltage": 2.0,
                "integral": 5.0,
                "measurement_revision": 1,
            },
            {"area_ema": 8.0, "window_revision": 2},
            dt=1.0,
        )
        self.assertAlmostEqual(state["integral"], 0.2)
        self.assertEqual(state["measurement_revision"], 2)

    def test_runtime_update_preserves_voltage_and_uses_new_parameters(self) -> None:
        """热更新不能重写初始电压，新参数应从下一轮开始生效。"""

        daq = _FakeDaq()
        lock = PowerLockController(daq, _FakeTrendLogger())  # type: ignore[arg-type]
        lock.start([_controller()], update_s=0.01)
        try:
            _wait_until(lambda: lock.status()["iterations"] >= 3)
            voltage_before = float(lock.status()["states"][0]["voltage"])
            self.assertGreater(voltage_before, 2.0)

            write_index = len(daq.snapshot())
            updated = lock.update_parameters(
                [_controller(target=9.0, kp=0.0, ki=0.0, max_step_v=0.02)],
                update_s=0.005,
            )
            self.assertEqual(updated["parameter_revision"], 2)
            self.assertEqual(updated["controllers"][0]["target"], 9.0)
            self.assertEqual(updated["controllers"][0]["max_step_v"], 0.02)

            previous_iterations = updated["iterations"]
            _wait_until(lambda: lock.status()["iterations"] > previous_iterations)
            writes_after_update = daq.snapshot()[write_index:]
            self.assertTrue(writes_after_update)
            for row in writes_after_update:
                self.assertAlmostEqual(row["value"], voltage_before)
                self.assertNotAlmostEqual(row["value"], 2.0)
        finally:
            lock.stop()

    def test_runtime_update_rejects_physical_definition_change_atomically(self) -> None:
        """反馈字段等停锁后参数发生变化时，原配置和版本号都不能被部分修改。"""

        daq = _FakeDaq()
        lock = PowerLockController(daq, _FakeTrendLogger())  # type: ignore[arg-type]
        lock.start([_controller(kp=0.0)], update_s=0.05)
        try:
            _wait_until(lambda: lock.status()["iterations"] >= 1)
            with self.assertRaisesRegex(ValueError, "feedback_field"):
                lock.update_parameters(
                    [_controller(feedback_field="top_ema", target=9.0, kp=0.0)],
                    update_s=0.01,
                )
            status = lock.status()
            self.assertEqual(status["parameter_revision"], 1)
            self.assertEqual(status["controllers"][0]["feedback_field"], "area_ema")
            self.assertEqual(status["controllers"][0]["target"], 10.0)
        finally:
            lock.stop()

    def test_runtime_update_requires_running_lock(self) -> None:
        lock = PowerLockController(_FakeDaq(), _FakeTrendLogger())  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeError, "not running"):
            lock.update_parameters([_controller()], update_s=1.0)


if __name__ == "__main__":
    unittest.main()
