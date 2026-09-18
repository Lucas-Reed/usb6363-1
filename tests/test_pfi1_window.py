from __future__ import annotations

import unittest

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
