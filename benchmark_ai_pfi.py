"""Standalone read benchmark. Stop all DAQ services before running.

No AO writes, waveform processing, web server, or production-code changes.
PFI0 rising edges must be present to start acquisition; light is not required.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter


def run_case(args, numpy_read, with_pfi):
    import nidaqmx
    import numpy as np
    from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogSingleChannelReader, CounterReader

    name = ("numpy" if numpy_read else "list") + ("_ai_pfi" if with_pfi else "_ai_only")
    result = {"case": name, "ok": False, "samples_per_channel": 0,
              "backlog_observations": [], "buffers": {}}
    stage = "setup"
    timings = {}
    started = None
    print(f"\n--- {name} ---", flush=True)
    try:
        with ExitStack() as stack:
            stage = "AI setup"
            ai = stack.enter_context(nidaqmx.Task(new_task_name="bench_AI"))
            channel = args.channel.lstrip("/")
            if "/" not in channel:
                channel = f"{args.device}/{channel}"
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
                for i, (label, task) in enumerate(tasks):
                    stage = f"{label} read"
                    before = perf_counter()
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
                    if i:
                        current = int(data[-1])
                        if label in previous_counts:
                            edges[label] += (current - previous_counts[label]) & 0xffffffff
                        previous_counts[label] = current
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
    for numpy_read, with_pfi in [(False, False), (False, True), (True, False), (True, True)]:
        report["cases"].append(run_case(args, numpy_read, with_pfi))
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {args.output.resolve()}")
    print("Read duration includes waiting for samples; it is not end-to-end acquisition latency.")
    return 0 if all(case["ok"] for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
