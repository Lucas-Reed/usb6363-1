"""PFI1 后续窗口的可选 AO 长漂反馈。

该模块只消费统一 AI 流，不创建第二个 AI task。PFI1 事件由 buffered
counter 以绝对 sample index 发布，窗口数据从统一流的绝对范围接口读取。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from two_peak.pfi1_window import Pfi1FollowupWindow


class Pfi1FeedbackController:
    def __init__(self, daq: Any) -> None:
        self._daq = daq
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._running = False
        self._error: str | None = None
        self._settings: dict[str, Any] = {}
        self._processor: Pfi1FollowupWindow | None = None
        self._pending: deque[int] = deque()
        self._last_event_sample = -1
        self._last_frame_id = 0
        self._events_seen = 0
        self._windows_done = 0
        self._expired_events = 0
        self._stale_events = 0
        self._skipped_backlog_events = 0
        self._latest_result: dict[str, Any] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=200)
        self._ao_value: float | None = None
        self._integral = 0.0
        self._last_result_time: float | None = None

    def start(self, settings: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise RuntimeError("PFI1 feedback is already running")
            normalized = self._validate_settings(settings)
            stream_status = self._daq.get_unified_ai_stream_status()
            if not stream_status.get("running"):
                raise RuntimeError("unified AI stream must be running before PFI1 monitoring")
            stream_settings = stream_status.get("settings") or {}
            if not stream_settings.get("event_timeline_enabled"):
                raise RuntimeError("unified AI stream must enable the buffered PFI event timeline")
            stream_rate = float(stream_settings.get("rate_per_channel") or 0.0)
            if stream_rate > 0:
                normalized["sample_rate_hz"] = stream_rate
            if "start_after_frame_id" not in settings:
                normalized["start_after_frame_id"] = int(stream_status.get("frame_id") or 0)
            self._settings = normalized
            self._error = None
            self._pending.clear()
            self._last_event_sample = -1
            self._last_frame_id = int(normalized.get("start_after_frame_id", 0))
            self._events_seen = 0
            self._windows_done = 0
            self._expired_events = 0
            self._stale_events = 0
            self._skipped_backlog_events = 0
            self._latest_result = None
            self._history.clear()
            self._ao_value = float(normalized["initial_voltage"])
            self._integral = 0.0
            self._last_result_time = None
            self._processor = Pfi1FollowupWindow(
                read_samples=self._read_samples,
                sample_rate_hz=float(normalized["sample_rate_hz"]),
                delay_ms=float(normalized["delay_ms"]),
                window_samples=int(normalized["window_samples"]),
                threshold=float(normalized["threshold"]),
                polarity=str(normalized["polarity"]),
                ema_alpha=float(normalized["ema_alpha"]),
            )
            if normalized["ao_feedback_enabled"]:
                self._daq.write_ao(
                    channel=str(normalized["ao_channel"]),
                    value=float(normalized["initial_voltage"]),
                    min_val=float(normalized["min_voltage"]),
                    max_val=float(normalized["max_voltage"]),
                )
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._running = True
            self._thread = threading.Thread(
                target=self._worker,
                args=(stop_event,),
                daemon=True,
                name="two-peak-pfi1-feedback",
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            event = self._stop_event
            thread = self._thread
            if event is not None:
                event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._running = False
            self._thread = None
            self._stop_event = None
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            age = None if self._last_result_time is None else max(0.0, time.time() - self._last_result_time)
            pause_after = max(1.0, 2.0 * float(self._settings.get("integral_dt_s", 0.5)))
            feedback_enabled = bool(self._settings.get("ao_feedback_enabled"))
            holding = feedback_enabled and age is not None and age >= pause_after
            if not feedback_enabled:
                feedback_state = "disabled"
            elif age is None:
                feedback_state = "waiting"
            elif holding:
                feedback_state = "holding"
            else:
                feedback_state = "active"
            return {
                "running": self._running,
                "error": self._error,
                "settings": dict(self._settings),
                "last_frame_id": self._last_frame_id,
                "events_seen": self._events_seen,
                "windows_done": self._windows_done,
                "expired_events": self._expired_events,
                "stale_events": self._stale_events,
                "skipped_backlog_events": self._skipped_backlog_events,
                "pending_events": len(self._pending),
                "latest_result": dict(self._latest_result) if self._latest_result else None,
                "history": [dict(item) for item in self._history],
                "ao_value": self._ao_value,
                "integral": self._integral,
                "last_measurement_age_s": age,
                "holding_last_ao": holding,
                "feedback_state": feedback_state,
            }

    def _validate_settings(self, raw: dict[str, Any]) -> dict[str, Any]:
        def f(name: str, default: float) -> float:
            return float(raw.get(name, default))

        result = {
            "pd_channel": str(raw.get("pd_channel", "ai2")),
            "sample_rate_hz": f("sample_rate_hz", 100_000.0),
            "delay_ms": f("delay_ms", 1.0),
            "window_samples": int(raw.get("window_samples", 200)),
            "threshold": f("threshold", 0.0),
            "polarity": str(raw.get("polarity", "positive")),
            "ema_alpha": f("ema_alpha", 0.02),
            "ao_channel": str(raw.get("ao_channel", "ao0")),
            "ao_feedback_enabled": bool(raw.get("ao_feedback_enabled", False)),
            "target": f("target", 0.0),
            "initial_voltage": f("initial_voltage", 0.0),
            "min_voltage": f("min_voltage", -10.0),
            "max_voltage": f("max_voltage", 10.0),
            "direction": int(raw.get("direction", 1)),
            "max_step_v": f("max_step_v", 0.01),
            "kp": f("kp", 0.0),
            "ki": f("ki", 0.0),
            "update_interval": f("update_interval", 0.05),
            "integral_dt_s": f("integral_dt_s", 0.5),
            "stale_event_after_s": f("stale_event_after_s", 0.5),
            "start_after_frame_id": int(raw.get("start_after_frame_id", 0)),
        }
        if result["sample_rate_hz"] <= 0 or result["delay_ms"] < 0:
            raise ValueError("sample_rate_hz must be > 0 and delay_ms must be >= 0")
        if result["window_samples"] < 1:
            raise ValueError("window_samples must be >= 1")
        if not 0 <= result["ema_alpha"] <= 1:
            raise ValueError("ema_alpha must be between 0 and 1")
        if result["min_voltage"] >= result["max_voltage"]:
            raise ValueError("min_voltage must be smaller than max_voltage")
        if not result["min_voltage"] <= result["initial_voltage"] <= result["max_voltage"]:
            raise ValueError("initial_voltage must be inside AO limits")
        if result["direction"] not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if (
            result["max_step_v"] < 0
            or result["update_interval"] <= 0
            or result["integral_dt_s"] <= 0
            or result["stale_event_after_s"] <= 0
        ):
            raise ValueError("max_step_v must be >= 0; timing values must be > 0")
        if result["start_after_frame_id"] < 0:
            raise ValueError("start_after_frame_id must be >= 0")
        return result

    def _read_samples(self, start_sample: int, count: int) -> list[float]:
        result = self._daq.get_unified_ai_range(
            channel=str(self._settings["pd_channel"]),
            start_sample=int(start_sample),
            end_sample=int(start_sample + count),
        )
        return list(result.get("values", []))

    def _worker(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                batch = self._daq.get_unified_ai_frame_batch(
                    after_frame_id=self._last_frame_id,
                    channels=[str(self._settings["pd_channel"])],
                    max_frames=20,
                )
                if batch.get("history_overrun"):
                    self._last_frame_id = max(
                        self._last_frame_id,
                        int(batch.get("oldest_available_frame_id", 0)) - 1,
                    )
                for frame in batch.get("frames", []):
                    self._last_frame_id = max(self._last_frame_id, int(frame["frame_id"]))
                    events = frame.get("pfi1_events") or []
                    for event in events:
                        sample = int(event.get("sample_index", event.get("event_sample", -1)))
                        if sample > self._last_event_sample:
                            self._last_event_sample = sample
                            self._pending.append(sample)
                            self._events_seen += 1
                self._drain_pending()
                stop_event.wait(float(self._settings["update_interval"]))
            except Exception as exc:  # keep the worker alive for a not-yet-available window
                with self._lock:
                    self._error = str(exc)
                stop_event.wait(float(self._settings["update_interval"]))

    def _drain_pending(self) -> None:
        processor = self._processor
        if processor is None:
            return
        if not self._pending:
            return
        stream_status = self._daq.get_unified_ai_stream_status()
        retained_start = int(stream_status.get("sample_range_start") or 0)
        retained_end = int(stream_status.get("sample_range_end") or 0)
        # At the expected sub-Hz/few-Hz trigger rate there is normally one event.
        # After a software/network stall, use only the newest measurement so old
        # errors cannot produce a burst of AO updates.
        if len(self._pending) > 1:
            skipped = len(self._pending) - 1
            newest = self._pending[-1]
            self._pending.clear()
            self._pending.append(newest)
            with self._lock:
                self._skipped_backlog_events += skipped
        remaining: deque[int] = deque()
        while self._pending:
            event = self._pending.popleft()
            window_start = event + int(
                round(
                    float(self._settings["delay_ms"])
                    * float(self._settings["sample_rate_hz"])
                    / 1000.0
                )
            )
            window_end = window_start + int(self._settings["window_samples"])
            if window_start < retained_start:
                with self._lock:
                    self._expired_events += 1
                continue
            if window_end > retained_end:
                remaining.append(event)
                continue
            event_age_s = (
                retained_end - window_end
            ) / float(self._settings["sample_rate_hz"])
            if event_age_s > float(self._settings["stale_event_after_s"]):
                with self._lock:
                    self._stale_events += 1
                continue
            try:
                result = processor.process_event(event)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)
                continue
            result_dict = result.as_dict()
            event_time = time.time()
            self._apply_control(float(result.ema if result.ema is not None else result.value))
            with self._lock:
                self._windows_done += 1
                result_dict["event_number"] = self._windows_done
                result_dict["event_time"] = event_time
                self._latest_result = result_dict
                self._last_result_time = event_time
                summary = {key: value for key, value in result_dict.items() if key != "samples"}
                summary["ao_value"] = self._ao_value
                self._history.append(summary)
                self._error = None
        self._pending = remaining

    def _apply_control(self, measured: float) -> None:
        if not bool(self._settings.get("ao_feedback_enabled", False)):
            return
        target = float(self._settings["target"])
        if abs(target) <= 1e-12:
            return
        # The feedback is event driven.  Use a fixed per-event integration time
        # so a several-second trigger pause does not integrate a stale error.
        dt = float(self._settings["integral_dt_s"])
        error = (target - measured) / abs(target)
        integral_before = self._integral
        self._integral += error * dt
        raw_delta = int(self._settings["direction"]) * (
            float(self._settings["kp"]) * error
            + float(self._settings["ki"]) * self._integral
        )
        max_step = float(self._settings["max_step_v"])
        delta = max(-max_step, min(max_step, raw_delta))
        current = float(self._ao_value if self._ao_value is not None else self._settings["initial_voltage"])
        next_value = max(
            float(self._settings["min_voltage"]),
            min(float(self._settings["max_voltage"]), current + delta),
        )
        if next_value in (float(self._settings["min_voltage"]), float(self._settings["max_voltage"])) and next_value != current:
            self._integral = integral_before
        if next_value != current:
            self._daq.write_ao(
                channel=str(self._settings["ao_channel"]),
                value=next_value,
                min_val=float(self._settings["min_voltage"]),
                max_val=float(self._settings["max_voltage"]),
            )
            self._ao_value = next_value


__all__ = ["Pfi1FeedbackController"]
