"""窗口原始电压写盘和 manifest 容错的无硬件测试。"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from two_peak.window_voltage_recorder import WindowVoltageRecorder


def _record(frame_id: int) -> dict[str, Any]:
    """生成同时含 A/B 窗口的一帧模拟记录。"""

    return {
        "frame_id": frame_id,
        "finished_at": 1000.0 + frame_id,
        "a_values": np.asarray([1.0, 2.0, 3.0], dtype=np.float32) + frame_id,
        "a_left": 10,
        "a_right": 12,
        "a_track_peak_index": 11,
        "a_peak_height_index": 12,
        "b_values": np.asarray([4.0, 5.0], dtype=np.float32) + frame_id,
        "b_left": 20,
        "b_right": 21,
        "b_track_peak_index": 20,
        "b_peak_height_index": 21,
    }


def _wait_for_frames(recorder: WindowVoltageRecorder, count: int) -> None:
    """等待后台写盘达到指定帧数。"""

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if recorder.status()["frames_written"] >= count:
            return
        time.sleep(0.01)
    raise AssertionError("等待窗口电压写盘超时")


class WindowVoltageRecorderTests(unittest.TestCase):
    def test_a_b_and_both_modes_write_expected_arrays(self) -> None:
        for mode, expected, absent in (
            ("a", {"a_values"}, {"b_values"}),
            ("b", {"b_values"}, {"a_values"}),
            ("both", {"a_values", "b_values"}, set()),
        ):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
                    recorder = WindowVoltageRecorder(
                        output_dir=Path(temp_dir),
                        session_id=f"mode_{mode}",
                        mode=mode,
                        metadata={"start_after_frame_id": 0},
                        chunk_frames=2,
                    )
                    recorder.start()
                    for frame_id in range(1, 4):
                        recorder.append(_record(frame_id))
                    status = recorder.stop()

                    self.assertIsNone(status["error"])
                    self.assertEqual(status["frames_written"], 3)
                    self.assertEqual(status["source_gap_frames"], 0)
                    self.assertEqual(status["queue_dropped_frames"], 0)
                    self.assertEqual(status["chunks_written"], 2)

                    first_chunk = Path(status["directory"]) / "chunk_000001.npz"
                    with np.load(first_chunk, allow_pickle=False) as data:
                        keys = set(data.files)
                        self.assertTrue(expected <= keys)
                        self.assertTrue(absent.isdisjoint(keys))

                    manifest = json.loads(
                        Path(status["manifest_file"]).read_text(encoding="utf-8")
                    )
                    self.assertTrue(manifest["completed"])
                    self.assertEqual(manifest["frames_written"], 3)

    def test_transient_manifest_permission_error_recovers(self) -> None:
        real_replace = os.replace
        manifest_attempts = 0

        def flaky_replace(source: Any, target: Any) -> None:
            nonlocal manifest_attempts
            if Path(target).name == "manifest.json":
                manifest_attempts += 1
                if manifest_attempts <= 2:
                    raise PermissionError(5, "测试用临时占用", str(target))
            real_replace(source, target)

        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            with patch("two_peak.window_voltage_recorder.os.replace", side_effect=flaky_replace):
                recorder = WindowVoltageRecorder(
                    Path(temp_dir),
                    "transient_manifest",
                    "a",
                    {},
                    chunk_frames=1,
                )
                recorder.start()
                recorder.append(_record(1))
                status = recorder.stop()

            self.assertGreaterEqual(manifest_attempts, 3)
            self.assertIsNone(status["error"])
            self.assertIsNone(status["manifest_warning"])
            self.assertEqual(status["manifest_write_failures"], 0)
            self.assertEqual(status["frames_written"], 1)

    def test_persistent_primary_manifest_lock_uses_final_fallback(self) -> None:
        real_replace = os.replace

        def locked_primary_replace(source: Any, target: Any) -> None:
            if Path(target).name == "manifest.json":
                raise PermissionError(5, "测试用持续占用", str(target))
            real_replace(source, target)

        with tempfile.TemporaryDirectory(dir=Path("data")) as temp_dir:
            with (
                patch(
                    "two_peak.window_voltage_recorder.MANIFEST_REPLACE_TIMEOUT_SECONDS",
                    0.03,
                ),
                patch(
                    "two_peak.window_voltage_recorder.MANIFEST_REPLACE_INITIAL_DELAY_SECONDS",
                    0.005,
                ),
                patch(
                    "two_peak.window_voltage_recorder.os.replace",
                    side_effect=locked_primary_replace,
                ),
            ):
                recorder = WindowVoltageRecorder(
                    Path(temp_dir),
                    "fallback_manifest",
                    "a",
                    {},
                    chunk_frames=2,
                )
                recorder.start()
                recorder.append(_record(1))
                recorder.append(_record(2))
                status = recorder.stop()

            self.assertIsNone(status["error"])
            self.assertEqual(status["frames_written"], 2)
            self.assertGreaterEqual(status["manifest_write_failures"], 1)
            self.assertIn("最终清单已改写", status["manifest_warning"])
            final_manifest = Path(status["manifest_file"])
            self.assertTrue(final_manifest.name.startswith("manifest_final_"))
            payload = json.loads(final_manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["completed"])
            self.assertEqual(payload["frames_written"], 2)

    def test_full_queue_becomes_explicit_fatal_error(self) -> None:
        recorder = WindowVoltageRecorder(
            Path("data"),
            "queue_full_test",
            "a",
            {},
            chunk_frames=1,
            queue_frames=1,
        )
        # 不启动 worker，手动填满队列，稳定模拟磁盘长期追不上输入。
        recorder._running = True
        recorder._queue.put_nowait(_record(1))
        with patch(
            "two_peak.window_voltage_recorder.WINDOW_QUEUE_PUT_TIMEOUT_SECONDS",
            0.01,
        ):
            recorder.append(_record(2))

        status = recorder.status()
        self.assertIn("写入队列持续满", status["error"])
        self.assertEqual(status["queue_dropped_frames"], 1)
        self.assertEqual(status["frames_received"], 0)


if __name__ == "__main__":
    unittest.main()
