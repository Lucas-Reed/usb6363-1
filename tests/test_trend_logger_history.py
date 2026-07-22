"""双峰统计逐帧补取的无硬件测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from two_peak.trend_logger import AreaTrendLogger


def _frame(frame_id: int, base_time: float) -> dict[str, Any]:
    """生成一帧单通道模拟波形，每帧数值略有变化。"""

    values = np.linspace(0.0, 0.9, 10, dtype=np.float32) + frame_id * 0.01
    return {
        "frame_id": frame_id,
        "segment_id": 1,
        "segment_frame_id": frame_id,
        "started_at": base_time + frame_id * 0.1 - 0.1,
        "finished_at": base_time + frame_id * 0.1,
        "channels": ["Dev2/ai0"],
        "channel_count": 1,
        "samples_per_channel": 10,
        "values": values.reshape(1, -1),
    }


class _BatchClient:
    """模拟统一流；每次最多返回三帧，迫使 logger 连续补取多批。"""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self.frames = frames
        self.status_calls = 0

    def get_unified_ai_stream_status(self) -> dict[str, Any]:
        self.status_calls += 1
        # start() 第一次查询用当前帧作基线。返回 0 表示记录从下一帧开始；
        # worker 后续查询则看到模拟流已经产生了全部帧。
        frame_id = 0 if self.status_calls == 1 else self.frames[-1]["frame_id"]
        return {
            "running": True,
            "has_frame": frame_id > 0,
            "frame_id": frame_id,
            "settings": {
                "channels": ["Dev2/ai0"],
                "rate_per_channel": 100_000.0,
                "samples_per_frame": 10,
            },
        }

    def get_unified_ai_frame_batch(
        self,
        after_frame_id: int,
        channels: list[str],
        max_frames: int = 100,
    ) -> dict[str, Any]:
        selected = [frame for frame in self.frames if frame["frame_id"] > after_frame_id][
            : min(3, max_frames)
        ]
        last_returned = selected[-1]["frame_id"] if selected else after_frame_id
        return {
            "frames": selected,
            "history_overrun": False,
            "missing_before_first": 0,
            "oldest_available_frame_id": self.frames[0]["frame_id"],
            "latest_available_frame_id": self.frames[-1]["frame_id"],
            "has_more": last_returned < self.frames[-1]["frame_id"],
        }


class _OverrunClient(_BatchClient):
    """模拟消费者停顿太久，所需旧帧已经被历史队列覆盖。"""

    def get_unified_ai_frame_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "frames": [],
            "history_overrun": True,
            "missing_before_first": 7,
            "oldest_available_frame_id": 8,
            "latest_available_frame_id": 10,
            "has_more": False,
        }


class _LateStartClient(_BatchClient):
    """模拟统一流已经运行到 frame 100 后，用户才点击开始记录。"""

    def get_unified_ai_stream_status(self) -> dict[str, Any]:
        self.status_calls += 1
        frame_id = 100 if self.status_calls == 1 else self.frames[-1]["frame_id"]
        return {
            "running": True,
            "has_frame": True,
            "frame_id": frame_id,
            "settings": {"channels": ["Dev2/ai0"]},
        }


def _wait_until(predicate: Any, timeout: float = 3.0) -> None:
    """等待后台线程达到测试状态，超时则让测试明确失败。"""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("等待后台记录线程超时")


class TrendLoggerHistoryTests(unittest.TestCase):
    def test_all_frames_are_processed_and_written_to_npz(self) -> None:
        base_time = time.time()
        frames = [_frame(frame_id, base_time) for frame_id in range(1, 13)]
        client = _BatchClient(frames)

        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            root = Path(temp_dir)
            logger = AreaTrendLogger(client, root / "trends")  # type: ignore[arg-type]
            logger.start(
                analysis_channel_index=0,
                area_left=2,
                area_right=5,
                area2_left=6,
                area2_right=8,
                window_frames=2,
                record_hz=10.0,
                top_percent=50.0,
                poll_interval=0.001,
                stream_source="unified_stream",
                channels=["ai0"],
                window_voltage_mode="both",
                record_full_frame=True,
                window_voltage_output_dir=root / "raw",
                session_id="history_test",
            )
            _wait_until(lambda: logger.status()["frames_seen"] == 12)
            status = logger.stop()

            self.assertIsNone(status["error"])
            self.assertEqual(status["frames_seen"], 12)
            self.assertEqual(status["window_voltage"]["frames_written"], 12)
            self.assertEqual(status["window_voltage"]["source_gap_frames"], 0)
            self.assertEqual(status["window_voltage"]["queue_dropped_frames"], 0)

            csv_path = Path(status["csv_file"])
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_id"]) for row in rows], list(range(1, 13)))
            # A 窗口含 4 点，50% 取最高 2 点；B 窗口含 3 点，取最高 2 点。
            # CSV 的 mean 使用最近 2 帧，验证 Top 与面积使用了同一统计窗口。
            self.assertEqual(int(rows[-1]["top_point_count"]), 2)
            self.assertEqual(int(rows[-1]["top2_point_count"]), 2)
            self.assertAlmostEqual(float(rows[-1]["top_current"]), 0.57)
            self.assertAlmostEqual(float(rows[-1]["top_mean"]), 0.565)
            self.assertAlmostEqual(float(rows[-1]["top2_current"]), 0.87)
            self.assertAlmostEqual(float(rows[-1]["top2_mean"]), 0.865)
            self.assertNotEqual(rows[-1]["top_ema"], "")
            self.assertNotEqual(rows[-1]["top2_ema"], "")

            chunk_path = root / "raw" / "window_voltage_history_test" / "chunk_000001.npz"
            with np.load(chunk_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["frame_id"], np.arange(1, 13))
                self.assertEqual(data["a_values"].shape, (12, 4))
                self.assertEqual(data["b_values"].shape, (12, 3))
                self.assertEqual(data["full_values"].shape, (12, 10))
                self.assertEqual(data["full_values"].dtype, np.float32)
                np.testing.assert_array_equal(data["segment_frame_id"], np.arange(1, 13))

            manifest_path = root / "raw" / "window_voltage_history_test" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["settings"]["source_stream_settings"]["rate_per_channel"],
                100_000.0,
            )

    def test_history_overrun_stops_with_clear_error(self) -> None:
        frames = [_frame(frame_id, time.time()) for frame_id in range(1, 11)]
        logger = AreaTrendLogger(_OverrunClient(frames), Path("unused"))  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            logger._output_dir = Path(temp_dir)
            logger.start(
                analysis_channel_index=0,
                area_left=2,
                area_right=5,
                window_frames=2,
                record_hz=1.0,
                poll_interval=0.001,
                stream_source="unified_stream",
                channels=["ai0"],
            )
            _wait_until(lambda: not logger.status()["running"])
            status = logger.status()

        self.assertIn("历史缓冲区已经覆盖 7 帧", status["error"])
        self.assertEqual(status["frames_seen"], 0)

    def test_normal_start_uses_current_stream_frame_as_baseline(self) -> None:
        base_time = time.time()
        frames = [_frame(frame_id, base_time) for frame_id in range(101, 104)]
        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            logger = AreaTrendLogger(
                _LateStartClient(frames), Path(temp_dir)
            )  # type: ignore[arg-type]
            logger.start(
                analysis_channel_index=0,
                area_left=2,
                area_right=5,
                window_frames=2,
                record_hz=10.0,
                poll_interval=0.001,
                stream_source="unified_stream",
                channels=["ai0"],
            )
            _wait_until(lambda: logger.status()["frames_seen"] == 3)
            status = logger.stop()

        self.assertEqual(status["settings"]["start_after_frame_id"], 100)
        self.assertEqual(status["last_frame_id"], 103)
        self.assertIsNone(status["error"])

    def test_internal_frame_gap_stops_recording(self) -> None:
        base_time = time.time()
        frames = [_frame(1, base_time), _frame(3, base_time)]
        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            logger = AreaTrendLogger(
                _BatchClient(frames), Path(temp_dir)
            )  # type: ignore[arg-type]
            logger.start(
                analysis_channel_index=0,
                area_left=2,
                area_right=5,
                window_frames=2,
                record_hz=1.0,
                poll_interval=0.001,
                stream_source="unified_stream",
                channels=["ai0"],
            )
            _wait_until(lambda: not logger.status()["running"])
            status = logger.status()

        self.assertIn("期望 frame_id=2，实际得到 3", status["error"])
        self.assertEqual(status["frames_seen"], 1)

    def test_duration_stops_recording_and_reports_normal_reason(self) -> None:
        """达到设定时长后应正常收尾，而不是产生错误状态。"""

        base_time = time.time()
        frames = [_frame(frame_id, base_time) for frame_id in range(1, 4)]
        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            logger = AreaTrendLogger(
                _BatchClient(frames), Path(temp_dir)
            )  # type: ignore[arg-type]
            logger.start(
                analysis_channel_index=0,
                area_left=2,
                area_right=5,
                window_frames=2,
                record_hz=10.0,
                # 约 0.03 秒，测试无需真的等待数分钟。
                duration_minutes=0.0005,
                poll_interval=0.001,
                stream_source="unified_stream",
                channels=["ai0"],
                window_voltage_mode="a",
                window_voltage_output_dir=Path(temp_dir) / "raw",
                session_id="duration_test",
            )
            _wait_until(lambda: not logger.status()["running"])
            status = logger.status()

        self.assertIsNone(status["error"])
        self.assertEqual(status["stop_reason"], "duration_elapsed")
        self.assertIsNotNone(status["finished_at"])
        self.assertGreaterEqual(status["elapsed_seconds"], 0.02)
        self.assertIsNone(status["remaining_seconds"])
        self.assertFalse(status["window_voltage"]["running"])


if __name__ == "__main__":
    unittest.main()
