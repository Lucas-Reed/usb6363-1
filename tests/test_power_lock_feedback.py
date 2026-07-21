"""双路功率锁定反馈字段读取的无硬件测试。"""

from __future__ import annotations

import unittest

from two_peak.power_lock import _read_feedback_value


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


if __name__ == "__main__":
    unittest.main()
