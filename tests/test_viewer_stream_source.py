"""双峰查看器数据源选择的无硬件回归测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from two_peak.viewer_capture import get_frame_stream_status


class _FakeDaq:
    """同时提供统一流和旧流状态，不访问真实 USB-6363。"""

    def __init__(
        self,
        unified_status: dict[str, Any],
        frame_status: dict[str, Any],
    ) -> None:
        self.unified_status = unified_status
        self.frame_status = frame_status

    def get_unified_ai_stream_status(self) -> dict[str, Any]:
        return dict(self.unified_status)

    def get_ai_frame_stream_status(self) -> dict[str, Any]:
        return dict(self.frame_status)


class ViewerStreamSourceTests(unittest.TestCase):
    """确保状态查询不会在统一流停止后偷偷切换到旧流。"""

    def test_unified_request_does_not_fall_back_to_running_legacy_stream(self) -> None:
        state = SimpleNamespace(
            daq=_FakeDaq(
                unified_status={"running": False, "error": "-200279", "frame_id": 99},
                frame_status={"running": True, "error": None, "frame_id": 5},
            )
        )

        status = get_frame_stream_status(state, "unified_stream")

        self.assertFalse(status["running"])
        self.assertEqual(status["error"], "-200279")
        self.assertEqual(status["frame_id"], 99)
        self.assertEqual(status["stream_source"], "unified_stream")

    def test_legacy_request_still_returns_legacy_status(self) -> None:
        state = SimpleNamespace(
            daq=_FakeDaq(
                unified_status={"running": True, "frame_id": 99},
                frame_status={"running": True, "frame_id": 5},
            )
        )

        status = get_frame_stream_status(state, "frame_stream")

        self.assertTrue(status["running"])
        self.assertEqual(status["frame_id"], 5)
        self.assertEqual(status["stream_source"], "frame_stream")


if __name__ == "__main__":
    unittest.main()
