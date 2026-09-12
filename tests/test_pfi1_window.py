from __future__ import annotations

import unittest
import time

from two_peak.pfi1_feedback import Pfi1FeedbackController
from two_peak.pfi1_window import Pfi1FollowupWindow


class Pfi1WindowTests(unittest.TestCase):
    def test_pulse_interval_is_the_reported_average_interval(self) -> None:
        values = [0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0]
        processor = Pfi1FollowupWindow(
            read_samples=lambda _start, _count: values,
            sample_rate_hz=1000,
            delay_ms=1,
            window_samples=len(values),
            threshold=1.0,
            ema_alpha=0.2,
        )

        result = processor.process_event(100)

        self.assertEqual(result.window_start, 101)
        self.assertEqual((result.pulse_start, result.pulse_end), (103, 105))
        self.assertEqual((result.average_start, result.average_end), (103, 105))
        self.assertEqual(result.value, 2.0)

    def test_full_window_is_reported_when_no_complete_pulse_exists(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4]
        processor = Pfi1FollowupWindow(
            read_samples=lambda _start, _count: values,
            sample_rate_hz=100_000,
            delay_ms=0,
            window_samples=len(values),
            threshold=1.0,
            ema_alpha=0.2,
        )

        result = processor.process_event(250)

        self.assertFalse(result.pulse_detected)
        self.assertEqual((result.average_start, result.average_end), (250, 253))
        self.assertAlmostEqual(result.value, 0.25)


class _FeedbackDaq:
    def __init__(self) -> None:
        self.served = False
        self.ao_writes: list[float] = []
        self.sample_range_end = 200

    def get_unified_ai_stream_status(self) -> dict:
        return {
            "running": True,
            "frame_id": 42,
            "sample_range_start": 0,
            "sample_range_end": self.sample_range_end,
            "settings": {
                "event_timeline_enabled": True,
                "rate_per_channel": 1000,
            },
        }

    def get_unified_ai_frame_batch(self, **_kwargs) -> dict:
        if self.served:
            return {"frames": [], "history_overrun": False}
        self.served = True
        return {
            "frames": [{"frame_id": 43, "pfi1_events": [{"sample_index": 100}]}],
            "history_overrun": False,
        }

    def get_unified_ai_range(self, **_kwargs) -> dict:
        return {"values": [0.0, 2.0, 2.0, 0.0]}

    def write_ao(self, **kwargs) -> None:
        self.ao_writes.append(float(kwargs["value"]))


class Pfi1FeedbackPauseTests(unittest.TestCase):
    def test_no_trigger_pause_holds_ao_and_integral(self) -> None:
        daq = _FeedbackDaq()
        controller = Pfi1FeedbackController(daq)
        controller.start(
            {
                "pd_channel": "ai2",
                "delay_ms": 0,
                "window_samples": 4,
                "threshold": 1.0,
                "ema_alpha": 0.2,
                "ao_feedback_enabled": True,
                "ao_channel": "ao0",
                "target": 1.0,
                "initial_voltage": 0.0,
                "min_voltage": -10.0,
                "max_voltage": 10.0,
                "direction": 1,
                "max_step_v": 1.0,
                "kp": 0.1,
                "ki": 0.2,
                "integral_dt_s": 0.5,
                "update_interval": 0.01,
            }
        )
        deadline = time.monotonic() + 1.0
        while controller.status()["windows_done"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        first = controller.status()
        writes = list(daq.ao_writes)
        time.sleep(0.05)
        second = controller.status()
        controller.stop()

        self.assertEqual(first["windows_done"], 1)
        self.assertEqual(second["integral"], first["integral"])
        self.assertEqual(second["ao_value"], first["ao_value"])
        self.assertEqual(daq.ao_writes, writes)

    def test_single_stale_event_is_not_applied_after_a_stall(self) -> None:
        daq = _FeedbackDaq()
        daq.sample_range_end = 1000
        controller = Pfi1FeedbackController(daq)
        controller.start(
            {
                "pd_channel": "ai2",
                "delay_ms": 0,
                "window_samples": 4,
                "threshold": 1.0,
                "ao_feedback_enabled": True,
                "ao_channel": "ao0",
                "target": 1.0,
                "initial_voltage": 0.0,
                "min_voltage": -10.0,
                "max_voltage": 10.0,
                "direction": 1,
                "max_step_v": 1.0,
                "kp": 0.1,
                "ki": 0.2,
                "integral_dt_s": 0.5,
                "update_interval": 0.01,
                "stale_event_after_s": 0.5,
            }
        )
        deadline = time.monotonic() + 1.0
        while controller.status()["stale_events"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        status = controller.stop()

        self.assertEqual(status["stale_events"], 1)
        self.assertEqual(status["windows_done"], 0)
        self.assertEqual(status["integral"], 0.0)
        self.assertEqual(daq.ao_writes, [0.0])


if __name__ == "__main__":
    unittest.main()
