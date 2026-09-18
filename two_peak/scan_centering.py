"""Slow offset-only centering of two manually selected peaks."""

import math
import threading
import time

import numpy as np

from two_peak.scan_source import plan_centering
from two_peak.signal import smooth_moving_average


def track_selected_peaks(values, centers, half_width, smooth_window=5):
    """Follow nearby positive peaks; report loss instead of walking into noise."""
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or not np.all(np.isfinite(raw)):
        raise ValueError("峰跟踪需要有限的一维电压数据")
    half = max(3, int(half_width))
    smoothed = smooth_moving_average(raw, int(smooth_window))
    found = []
    for center in centers:
        left, right = max(0, int(round(center)) - half), min(len(raw), int(round(center)) + half + 1)
        if right - left < 5:
            raise ValueError("峰跟踪窗口超出波形")
        local = smoothed[left:right]
        top = int(np.argmax(local))
        # Second differences avoid interpreting a broad peak's slope as noise.
        noise = float(np.median(np.abs(np.diff(raw[left:right], n=2)))) / 1.652
        prominence = float(local[top] - np.percentile(local, 15))
        if top in (0, len(local) - 1) or prominence <= max(6 * noise, 1e-12):
            raise ValueError("已选峰跟踪丢失，请重新点击峰或调整搜索半宽")
        found.append(left + top)
    if len(set(found)) != len(found):
        raise ValueError("两个搜索窗口落在同一个峰，请重新选峰")
    return found


class ScanCenteringController:
    def __init__(self, daq, source):
        self.daq, self.source = daq, source
        self._stop = threading.Event()
        self._thread = None
        self._lifecycle = threading.Lock()
        self._lock = threading.Lock()
        self._status = dict(running=False, error=None)

    def status(self):
        with self._lock:
            return dict(self._status)

    def _update(self, **values):
        with self._lock:
            self._status.update(values)

    def start(self, payload):
        with self._lifecycle:
            if self._thread and self._thread.is_alive():
                raise ValueError("自动居中已在运行")
            source = self.source.read()
            frame = self.daq.get_unified_ai_stream_latest_frame()
            channel = str(payload["channel"])
            values = frame["values"][frame["channels"].index(channel)]
            centers = [int(i) for i in payload["peak_indices"]]
            breaks = [int(i.strip()) for i in str(payload.get("breakpoints", "2500,7500")).split(",")]
            if len(centers) != 2 or len(breaks) != 2 or not 0 <= breaks[0] < breaks[1] < len(values):
                raise ValueError("请设置两个峰和下降段拐点")
            config = dict(min_voltage=float(payload.get("min_voltage", .01)),
                          max_voltage=float(payload.get("max_voltage", 5)),
                          gain=float(payload.get("gain", .3)),
                          max_step_v=float(payload.get("max_step_v", .01)),
                          deadband_samples=float(payload.get("deadband_samples", 2)))
            interval = float(payload.get("interval", 1))
            half = int(payload.get("search_window_half", 20))
            smooth = int(payload.get("smooth_window", 5))
            if not math.isfinite(interval) or interval < .2 or half < 3 or smooth < 1:
                raise ValueError("调整周期至少 0.2 秒；请检查峰跟踪窗口")
            selection = dict(centering_peaks=[dict(index=i) for i in centers], fit=dict(breakpoints=breaks))
            plan_centering(selection, source, **config)
            self._stop.clear()
            with self._lock:
                self._status = dict(running=True, error=None, peak_indices=centers,
                                    target_index=sum(breaks) / 2, source=source, updates=0,
                                    message="等待新的采集帧")
            self._thread = threading.Thread(target=self._run,
                args=(channel, centers, breaks, config, interval, half, smooth, source),
                name="scan-offset-centering", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self):
        with self._lifecycle:
            self._stop.set()
            if self._thread:
                self._thread.join()
            self._update(running=False)
            return self.status()

    def _run(self, channel, centers, breaks, config, interval, half, smooth, expected):
        last_id = None
        try:
            while not self._stop.wait(interval):
                frame = self.daq.get_unified_ai_stream_latest_frame()
                if time.time() - float(frame.get("finished_at", 0)) > max(2, 2 * interval):
                    raise ValueError("采集帧已过期，已停止偏置调整")
                if frame.get("frame_id") == last_id:
                    continue
                last_id = frame["frame_id"]
                if frame.get("pfi1_triggered"):
                    self._update(message="跳过 PFI1 影响帧")
                    continue
                values = frame["values"][frame["channels"].index(channel)]
                centers = track_selected_peaks(values, centers, half, smooth)
                current = self.source.read()
                for key in ("resource", "channel", "amplitude_vpp", "offset_v", "frequency_hz"):
                    if current[key] != expected[key]:
                        raise ValueError("信号源参数已被手动修改，请重新启动居中")
                selection = dict(centering_peaks=[dict(index=i) for i in centers], fit=dict(breakpoints=breaks))
                plan = plan_centering(selection, current, **config)
                self._update(peak_indices=centers, error_samples=plan["error_samples"], frame_id=last_id)
                if self._stop.is_set():
                    break
                if abs(plan["correction_v"]) > 1e-9:
                    expected = self.source.apply(plan)
                    # Predict the translation before searching the next frame.
                    centers = plan["predicted_indices"]
                    self._update(source=expected, updates=self.status()["updates"] + 1,
                                 message="已微调偏置，Vpp 保持不变")
                else:
                    self._update(message="两峰中点处于容差内，保持偏置")
        except Exception as exc:
            self._update(error=str(exc), message="已停止调整，保持最后偏置")
        finally:
            self._update(running=False)
