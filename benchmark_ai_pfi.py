"""Standalone read benchmark. Stop all DAQ services before running.

No AO writes, business algorithms, web server, or production-code changes.
--suite pipeline adds production edge detection, framing and cache publication.
PFI0 rising edges must be present to start acquisition; light is not required.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from time import time


class ProductionProcessing:
    """Reuse production edge detection, assembler and frame publication, without a server.

    Only the short raw-deque update is mirrored here, because the production
    worker has not factored it into a callable method. No business threads run.
    """

    def __init__(self, args, level, channel):
        from collections import deque
        from usb6363_core import DaqController, _counter_transition_samples
        from usb6363_core import UNIFIED_HISTORY_MAX_BYTES, UNIFIED_HISTORY_MAX_FRAMES
        from usb6363.pfi_frames import PfiFrameAssembler

        self.level = level
        self.transition = _counter_transition_samples
        self.previous = [None, None]
        self.cursor = 0
        self.frames = 0
        self.events = [0, 0]
        self.channel = channel
        self.assembler = PfiFrameAssembler(args.block_samples, 1) if level >= 2 else None
        self.core = DaqController(args.device) if level >= 3 else None
        self.settings = dict(
            channels=[channel], rate_per_channel=args.rate, aggregate_rate=args.rate,
            terminal_config="DIFF", min_val=-5, max_val=5, trigger_enabled=True,
            trigger_source=f"/{args.device}/PFI0", trigger_edge="RISING",
            trigger_mode="start_only", resync_every_frames=0, frame_alignment="pfi0",
            frame_duration_seconds=args.block_samples / args.rate,
            frame_duration_ms=args.block_samples / args.rate * 1000,
            frame_rate_hz=args.rate / args.block_samples)
        if self.core:
            core = self.core
            core._unified_buffers = {channel: deque(maxlen=core._ai_buffer_size)}
            core._unified_range_buffers = {channel: deque(maxlen=core._ai_buffer_size)}
            core._unified_sample_counts = {channel: 0}
            capacity = min(UNIFIED_HISTORY_MAX_FRAMES, UNIFIED_HISTORY_MAX_BYTES // (args.block_samples * 4))
            core._unified_frame_history = deque(maxlen=max(1, capacity))

    def process(self, blocks, timings):
        import numpy as np

        start = perf_counter()
        # Preserve production's Python int conversion and per-sample comparison.
        events = []
        for i in range(2):
            counts = [int(value) for value in blocks[i + 1]]
            found, self.previous[i] = self.transition(counts, self.previous[i], self.cursor)
            self.events[i] += len(found)
            events.append(found)
        timings.setdefault("edge_detection", []).append((perf_counter() - start) * 1000)
        values = blocks[0]
        end = self.cursor + len(values)
        now = time()
        if self.core:
            start = perf_counter()
            core = self.core
            with core._ai_lock:
                core._unified_latest[self.channel] = values[-1]
                core._unified_buffers[self.channel].extend(values)
                core._unified_range_buffers[self.channel].extend(values)
                core._unified_sample_counts[self.channel] += len(values)
                core._unified_sample_cursor = end
                core._unified_last_update = now
            timings.setdefault("raw_cache", []).append((perf_counter() - start) * 1000)
        if self.assembler:
            start = perf_counter()
            windows = self.assembler.append([values], self.cursor, events[0], events[1])
            timings.setdefault("frame_assembly", []).append((perf_counter() - start) * 1000)
            start = perf_counter()
            for window in windows:
                self.frames += 1
                if self.core:
                    history_values = np.ascontiguousarray(window["values"], dtype=np.float32)
                    self.core._publish_unified_window(self.settings, window, history_values, 1, self.frames, now)
            if self.core:
                timings.setdefault("frame_publication", []).append((perf_counter() - start) * 1000)
        self.cursor = end


def run_case(args, numpy_read, with_pfi, processing_level=0):
    import nidaqmx
    import numpy as np
    from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogSingleChannelReader, CounterReader

    name = ("numpy" if numpy_read else "list") + ("_ai_pfi" if with_pfi else "_ai_only")
    if processing_level:
        name += ("", "_A_edges", "_B_frames", "_C_cache_publish")[processing_level]
    result = {"case": name, "ok": False, "samples_per_channel": 0,
              "backlog_observations": [], "buffers": {}}
    stage = "setup"
    timings = {}
    processing_timings = {}
    read_intervals = []
    last_read_start = None
    processor = None
    started = None
    print(f"\n--- {name} ---", flush=True)
    try:
        with ExitStack() as stack:
            stage = "AI setup"
            ai = stack.enter_context(nidaqmx.Task(new_task_name="bench_AI"))
            channel = args.channel.lstrip("/")
            if "/" not in channel:
                channel = f"{args.device}/{channel}"
            if processing_level:
                processor = ProductionProcessing(args, processing_level, channel)
            ai.ai_channels.add_ai_voltage_chan(
                channel, terminal_config=TerminalConfiguration.DIFF,
                min_val=-5, max_val=5)
            ai_buffer = max(args.block_samples * 50, int(args.rate * 5))
            ai.timing.cfg_samp_clk_timing(args.rate, sample_mode=AcquisitionType.CONTINUOUS,
                                         samps_per_chan=ai_buffer)
            ai.in_stream.input_buf_size = ai_buffer
            ai.triggers.start_trigger.cfg_dig_edge_start_trig(
                f"/{args.device}/PFI0", trigger_edge=Edge.RISING)
            tasks = [("AI", ai)]
            if with_pfi:
                for line, counter, edge in [("PFI0", args.pfi0_counter, Edge.RISING),
                                             ("PFI1", args.pfi1_counter, Edge.FALLING)]:
                    stage = f"{line} setup"
                    task = stack.enter_context(nidaqmx.Task(new_task_name=f"bench_{line}"))
                    physical_counter = counter.lstrip("/")
                    if "/" not in physical_counter:
                        physical_counter = f"{args.device}/{physical_counter}"
                    ci = task.ci_channels.add_ci_count_edges_chan(physical_counter, edge=edge)
                    ci.ci_count_edges_term = f"/{args.device}/{line}"
                    task.timing.cfg_samp_clk_timing(
                        args.rate, source=f"/{args.device}/ai/SampleClock",
                        sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=args.block_samples)
                    if args.counter_buffer_seconds:
                        task.in_stream.input_buf_size = max(
                            args.block_samples, int(args.rate * args.counter_buffer_seconds))
                    tasks.append((line, task))
            readers = []
            if numpy_read:
                readers = [(AnalogSingleChannelReader(ai.in_stream),
                            np.empty(args.block_samples, dtype=np.float64))]
                readers += [(CounterReader(task.in_stream),
                             np.empty(args.block_samples, dtype=np.uint32)) for _, task in tasks[1:]]
            for label, task in tasks[1:] + tasks[:1]:
                stage = f"{label} start"
                task.start()
            result["actual_rate"] = ai.timing.samp_clk_rate
            for label, task in tasks:
                result["buffers"][label] = task.in_stream.input_buf_size
                timings[label] = []
            print(f"buffers (samples/channel): {result['buffers']}", flush=True)
            last_log = 0.0
            previous_counts = {}
            edges = {label: 0 for label, _ in tasks[1:]}
            measured_samples = 0
            while started is None or perf_counter() - started < args.seconds:
                blocks = []
                for i, (label, task) in enumerate(tasks):
                    stage = f"{label} read"
                    before = perf_counter()
                    if i == 0:
                        if last_read_start is not None and started is not None:
                            read_intervals.append((before - last_read_start) * 1000)
                        last_read_start = before
                    if numpy_read:
                        reader, data = readers[i]
                        method = reader.read_many_sample if i == 0 else reader.read_many_sample_uint32
                        count = method(data, number_of_samples_per_channel=args.block_samples, timeout=10)
                        if count != args.block_samples:
                            raise RuntimeError(f"short read: {count}/{args.block_samples}")
                    else:
                        data = task.read(number_of_samples_per_channel=args.block_samples, timeout=10)
                    if started is not None:
                        timings[label].append((perf_counter() - before) * 1000)
                    blocks.append(data)
                    if i:
                        current = int(data[-1])
                        if label in previous_counts:
                            edges[label] += (current - previous_counts[label]) & 0xffffffff
                        previous_counts[label] = current
                if processor:
                    stage = "production processing"
                    processing_start = perf_counter()
                    processor.process(blocks, processing_timings)
                    processing_timings.setdefault("processing_total", []).append(
                        (perf_counter() - processing_start) * 1000)
                result["samples_per_channel"] += args.block_samples
                if started is None:
                    started = perf_counter()  # exclude initial trigger wait and first block
                else:
                    measured_samples += args.block_samples
                elapsed = perf_counter() - started
                if elapsed - last_log >= 1:
                    stage = "backlog query"
                    backlog = {label: task.in_stream.avail_samp_per_chan for label, task in tasks}
                    result["backlog_observations"].append({"seconds": round(elapsed, 3), **backlog})
                    print(f"{elapsed:6.1f}s  read={measured_samples / elapsed:,.0f} samples/s  backlog={backlog}",
                          flush=True)
                    last_log = elapsed
            result.update(ok=True, elapsed_seconds=elapsed,
                          measured_samples_per_second=measured_samples / elapsed,
                          observed_edges=edges)
    except Exception as exc:
        result.update(error_stage=stage, error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL at {stage}: {exc}", flush=True)
    if started is not None:
        result["elapsed_seconds"] = perf_counter() - started
        result["measured_samples_per_second"] = measured_samples / result["elapsed_seconds"]
        result["observed_edges"] = edges
    if processor:
        result["processing"] = {"samples_processed": processor.cursor,
                                "frames_assembled": processor.frames,
                                "pfi_transition_events": processor.events}
        if processor.level >= 2 and processor.frames == 0:
            result["warning"] = "No aligned frames produced: framing/publication load was not exercised. Check PFI0."
            print(result["warning"], flush=True)
        if processor.core:
            result["processing"].update(
                history_frames=len(processor.core._unified_frame_history),
                history_capacity=processor.core._unified_frame_history.maxlen,
                history_evicted=processor.core._unified_history_evicted_frames)
    def summary(values):
        return {"p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)), "max": max(values),
                "mean": float(np.mean(values))} if values else {}
    result["processing_duration_ms"] = {key: summary(values) for key, values in processing_timings.items()}
    result["AI_read_start_interval_ms"] = summary(read_intervals[1:])  # exclude first trigger wait
    result["read_duration_ms"] = {
        label: {"p50": float(np.percentile(values, 50)),
                "p99": float(np.percentile(values, 99)), "max": max(values)}
        for label, values in timings.items() if values}
    print(f"{'PASS' if result['ok'] else 'FAIL'} {name}: {result['read_duration_ms']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="Dev2")
    parser.add_argument("--channel", default="ai0")
    parser.add_argument("--rate", type=float, default=2_000_000)
    parser.add_argument("--block-samples", type=int, default=10_000)
    parser.add_argument("--seconds", type=float, default=60, help="seconds per case")
    parser.add_argument("--suite", choices=("read", "pipeline"), default="read",
                        help="pipeline: PFI read baseline, then production edges, frames, caches/publication")
    parser.add_argument("--pipeline-reader", choices=("list", "numpy"), default="list",
                        help="pipeline defaults to current production's list reader")
    parser.add_argument("--pfi0-counter", default="ctr0")
    parser.add_argument("--pfi1-counter", default="ctr1")
    parser.add_argument("--counter-buffer-seconds", type=float, default=0,
                        help="0 uses DAQmx automatic counter buffering, as production does")
    parser.add_argument("--output", type=Path,
                        default=Path(f"daq_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"))
    args = parser.parse_args()
    if args.rate <= 0 or args.block_samples <= 0 or args.seconds <= 0 or args.counter_buffer_seconds < 0:
        parser.error("rate, block-samples and seconds must be positive; buffer seconds cannot be negative")
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cases": []}
    print("Stop DAQ services first. PFI0 rising starts AI; PFI1 counts falling edges.", flush=True)
    cases = [(False, False, 0), (False, True, 0), (True, False, 0), (True, True, 0)]
    if args.suite == "pipeline":
        cases = [(args.pipeline_reader == "numpy", True, level) for level in range(4)]
    for numpy_read, with_pfi, level in cases:
        report["cases"].append(run_case(args, numpy_read, with_pfi, level))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {args.output.resolve()}")
    print("Read duration includes waiting for samples; it is not end-to-end acquisition latency.")
    return 0 if all(case["ok"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
