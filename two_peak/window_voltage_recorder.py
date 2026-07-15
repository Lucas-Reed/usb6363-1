"""动态窗口逐帧电压的分块二进制记录器。

这个模块只负责把 8766 已经拿到的窗口数据写入磁盘，不访问采集卡，也不参与找峰。
每个 NPZ 块保存固定数量的帧；停止时会把不足一个完整块的数据也写出来。
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


VALID_MODES = {"none", "a", "b", "both"}


class WindowVoltageRecorder:
    """用后台线程把 A/B 动态窗口逐帧写成 NPZ。"""

    def __init__(
        self,
        output_dir: Path,
        session_id: str,
        mode: str,
        metadata: dict[str, Any],
        chunk_frames: int = 100,
        queue_frames: int = 500,
    ) -> None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in VALID_MODES - {"none"}:
            raise ValueError("window voltage mode must be a, b or both")
        if chunk_frames < 1:
            raise ValueError("window voltage chunk_frames must be >= 1")
        if queue_frames < chunk_frames:
            raise ValueError("window voltage queue_frames must be >= chunk_frames")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("session_id 只能包含字母、数字、下划线和连字符")

        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_frames)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_dir = output_dir / f"window_voltage_{session_id}"
        self._manifest_path = self._session_dir / "manifest.json"
        self._session_id = session_id
        self._mode = normalized_mode
        self._chunk_frames = int(chunk_frames)
        self._metadata = dict(metadata)
        self._started_at = time.time()
        self._finished_at: float | None = None
        self._running = False
        self._error: str | None = None
        self._frames_received = 0
        self._frames_written = 0
        self._source_gap_frames = 0
        self._queue_dropped_frames = 0
        baseline_frame_id = int(metadata.get("start_after_frame_id", 0) or 0)
        # 同步触发时把基线当作上一帧，这样第一帧若已经跳号也能准确计入缺口。
        self._last_received_frame_id: int | None = (
            baseline_frame_id if baseline_frame_id > 0 else None
        )
        self._first_written_frame_id: int | None = None
        self._last_written_frame_id: int | None = None
        self._chunks: list[dict[str, Any]] = []
        self._bytes_written = 0

    def start(self) -> dict[str, Any]:
        """创建会话目录并启动写盘线程。"""

        self._session_dir.mkdir(parents=True, exist_ok=False)
        self._running = True
        self._write_manifest(completed=False)
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="two-peak-window-voltage-writer",
        )
        self._thread.start()
        return self.status()

    def append(self, record: dict[str, Any]) -> None:
        """把一帧窗口数据放入写盘队列，不阻塞趋势采集线程。"""

        frame_id = int(record["frame_id"])
        with self._lock:
            if not self._running or self._error:
                return
            if self._last_received_frame_id is not None and frame_id > self._last_received_frame_id + 1:
                self._source_gap_frames += frame_id - self._last_received_frame_id - 1
            self._last_received_frame_id = frame_id
            self._frames_received += 1

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # 队列满说明磁盘写入已经追不上输入；继续统计丢帧，方便实验后判断数据是否可用。
            with self._lock:
                self._queue_dropped_frames += 1

    def stop(self) -> dict[str, Any]:
        """停止接收新帧，等待队列写完并刷新最后一个不完整块。"""

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=15.0)
        with self._lock:
            if thread is not None and thread.is_alive() and self._error is None:
                self._error = "窗口电压写盘线程未能在 15 秒内停止"
            self._running = False
            self._finished_at = self._finished_at or time.time()
        self._write_manifest(completed=self._error is None and not (thread and thread.is_alive()))
        return self.status()

    def status(self) -> dict[str, Any]:
        """返回 WebUI 可以安全显示的小体积状态，不返回原始电压数组。"""

        with self._lock:
            return {
                "enabled": True,
                "running": self._running,
                "error": self._error,
                "mode": self._mode,
                "session_id": self._session_id,
                "directory": str(self._session_dir.resolve()),
                "manifest_file": str(self._manifest_path.resolve()),
                "chunk_frames": self._chunk_frames,
                "frames_received": self._frames_received,
                "frames_written": self._frames_written,
                "source_gap_frames": self._source_gap_frames,
                "queue_dropped_frames": self._queue_dropped_frames,
                "chunks_written": len(self._chunks),
                "bytes_written": self._bytes_written,
                "first_frame_id": self._first_written_frame_id,
                "last_frame_id": self._last_written_frame_id,
            }

    def _worker(self) -> None:
        """在后台聚合数据并写块。"""

        pending: list[dict[str, Any]] = []
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    pending.append(self._queue.get(timeout=0.1))
                except queue.Empty:
                    pass
                if len(pending) >= self._chunk_frames:
                    self._write_chunk(pending[: self._chunk_frames])
                    del pending[: self._chunk_frames]

            if pending:
                self._write_chunk(pending)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False
                self._finished_at = time.time()
            self._write_manifest(completed=self._error is None)

    def _write_chunk(self, records: list[dict[str, Any]]) -> None:
        """把一批同宽窗口写成一个未压缩 NPZ，并用原子改名完成落盘。"""

        chunk_index = len(self._chunks) + 1
        name = f"chunk_{chunk_index:06d}.npz"
        final_path = self._session_dir / name
        temp_path = self._session_dir / f".{name}.tmp"

        arrays: dict[str, np.ndarray] = {
            "frame_id": np.asarray([row["frame_id"] for row in records], dtype=np.int64),
            "finished_at": np.asarray([row["finished_at"] for row in records], dtype=np.float64),
        }
        if self._mode in ("a", "both"):
            arrays.update(_window_arrays(records, "a"))
        if self._mode in ("b", "both"):
            arrays.update(_window_arrays(records, "b"))

        with temp_path.open("wb") as file:
            np.savez(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        temp_path.replace(final_path)

        byte_count = final_path.stat().st_size
        first_frame_id = int(records[0]["frame_id"])
        last_frame_id = int(records[-1]["frame_id"])
        with self._lock:
            self._frames_written += len(records)
            self._first_written_frame_id = self._first_written_frame_id or first_frame_id
            self._last_written_frame_id = last_frame_id
            self._bytes_written += byte_count
            self._chunks.append(
                {
                    "file": name,
                    "frames": len(records),
                    "first_frame_id": first_frame_id,
                    "last_frame_id": last_frame_id,
                    "bytes": byte_count,
                }
            )
        self._write_manifest(completed=False)

    def _write_manifest(self, completed: bool) -> None:
        """原子更新清单，避免程序意外退出时留下半个 JSON。"""

        with self._lock:
            payload = {
                "format": "two_peak_window_voltage_npz_v1",
                "session_id": self._session_id,
                "mode": self._mode,
                "dtype": "float32",
                "chunk_frames": self._chunk_frames,
                "started_at": datetime.fromtimestamp(self._started_at).isoformat(timespec="milliseconds"),
                "finished_at": (
                    datetime.fromtimestamp(self._finished_at).isoformat(timespec="milliseconds")
                    if self._finished_at is not None
                    else None
                ),
                "completed": bool(completed),
                "error": self._error,
                "frames_received": self._frames_received,
                "frames_written": self._frames_written,
                "source_gap_frames": self._source_gap_frames,
                "queue_dropped_frames": self._queue_dropped_frames,
                "first_frame_id": self._first_written_frame_id,
                "last_frame_id": self._last_written_frame_id,
                "bytes_written": self._bytes_written,
                "chunks": list(self._chunks),
                "settings": dict(self._metadata),
            }
        temp_path = self._manifest_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._manifest_path)


def _window_arrays(records: list[dict[str, Any]], prefix: str) -> dict[str, np.ndarray]:
    """把同一个窗口的逐帧字典整理成 NPZ 数组。"""

    values = [np.asarray(row[f"{prefix}_values"], dtype=np.float32) for row in records]
    widths = {int(item.size) for item in values}
    if len(widths) != 1:
        raise RuntimeError(f"窗口 {prefix.upper()} 的点数在同一记录中发生了变化")
    return {
        f"{prefix}_values": np.stack(values, axis=0),
        f"{prefix}_left": np.asarray([row[f"{prefix}_left"] for row in records], dtype=np.int32),
        f"{prefix}_right": np.asarray([row[f"{prefix}_right"] for row in records], dtype=np.int32),
        f"{prefix}_track_peak_index": np.asarray(
            [_optional_index(row.get(f"{prefix}_track_peak_index")) for row in records],
            dtype=np.int32,
        ),
        f"{prefix}_peak_height_index": np.asarray(
            [_optional_index(row.get(f"{prefix}_peak_height_index")) for row in records],
            dtype=np.int32,
        ),
    }


def _optional_index(value: Any) -> int:
    """NPZ 的整数数组不能保存 None，因此用 -1 表示没有该索引。"""

    return -1 if value is None else int(value)
