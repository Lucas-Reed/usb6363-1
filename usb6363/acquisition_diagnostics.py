"""Bounded, in-memory acquisition timings; write only after a failure."""
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import time


class AcquisitionDiagnostics:
    def __init__(self):
        self.started = self.tick = self.last_snapshot = time.perf_counter()
        self.stage = "setup"
        self.failed_stage = None
        self.previous_read = None
        self.stats = {}
        self.recent = deque(maxlen=120)

    def mark(self, stage):
        now = time.perf_counter()
        self.record(self.stage, (now - self.tick) * 1000)
        self.stage, self.tick = stage, now

    def record(self, label, milliseconds):
        entry = self.stats.setdefault(label, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += milliseconds
        entry[2] = max(entry[2], milliseconds)

    def begin_read(self):
        now = time.perf_counter()
        if self.previous_read is not None:
            self.record("AI_read_start_interval", (now - self.previous_read) * 1000)
        self.previous_read = now
        self.mark("AI read")

    def snapshot(self, tasks, cursor, frame_id):
        now = time.perf_counter()
        if now - self.last_snapshot < 1:
            return
        item = dict(elapsed_seconds=now - self.started, sample_cursor=cursor,
                    frame_id=frame_id, timings_ms=self.summary())
        item["buffers"] = {}
        for label, task in tasks:
            try:
                item["buffers"][label] = dict(available=task.in_stream.avail_samp_per_chan,
                                             capacity=task.in_stream.input_buf_size)
            except Exception as exc:
                item["buffers"][label] = {"query_error": str(exc)}
        self.recent.append(item)
        self.stats = {}
        self.last_snapshot = now

    def summary(self):
        return {key: dict(count=n, mean=total / n, max=maximum)
                for key, (n, total, maximum) in self.stats.items()}

    def freeze_failure(self):
        if self.failed_stage is None:
            self.failed_stage = self.stage
            self.mark("task cleanup after failure")

    def save_failure(self, settings, exc):
        self.freeze_failure()
        report = dict(settings=settings, error=str(exc), error_stage=self.failed_stage,
                      elapsed_seconds=time.perf_counter() - self.started,
                      recent=list(self.recent), final_timings_ms=self.summary(),
                      note="Durations include scheduling delays. Buffer samples are per channel. No waveform data.")
        folder = Path(__file__).resolve().parent.parent / "diagnostics"
        folder.mkdir(exist_ok=True)
        path = folder / f"acquisition_error_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
