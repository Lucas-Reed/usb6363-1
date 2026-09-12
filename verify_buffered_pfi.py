"""Verify buffered PFI counters sampled by the AI sample clock.

This is a one-shot hardware diagnostic. It does not start the production
server or alter the production acquisition code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from usb6363 import nidaqmx_driver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="Dev2")
    parser.add_argument("--channels", default="ai0,ai1,ai2")
    parser.add_argument("--rate", type=float, default=100000.0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--block-samples", type=int, default=1000)
    parser.add_argument("--pfi0-counter", default="ctr0")
    parser.add_argument("--pfi1-counter", default="ctr1")
    parser.add_argument("--output", type=Path, default=Path("buffered_pfi_report.json"))
    args = parser.parse_args()

    channels = [item.strip() for item in args.channels.split(",") if item.strip()]
    print("Starting buffered PFI diagnostic...")
    print(f"device={args.device} channels={channels} rate={args.rate:g} Hz")
    print("PFI0=RISING, PFI1=FALLING")
    print("counter sample clock=/" + args.device + "/ai/SampleClock")
    try:
        report = nidaqmx_driver.verify_buffered_pfi_with_ai_clock(
            device_name=args.device,
            physical_channels=channels,
            rate=args.rate,
            seconds=args.seconds,
            pfi0_counter=args.pfi0_counter,
            pfi1_counter=args.pfi1_counter,
            block_samples=args.block_samples,
        )
    except Exception as exc:
        print("RESULT: ROUTE_OR_HARDWARE_FAILURE")
        print(f"{type(exc).__name__}: {exc}")
        return 2

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"samples_checked={report['samples_checked']}")
    print(f"pfi0_change_count={report['pfi0_change_count']}")
    print(f"pfi1_change_count={report['pfi1_change_count']}")
    print(f"report={args.output}")
    if report["pfi0_change_count"] == 0:
        print("RESULT: NO_PFI0_TRANSITIONS")
        return 3
    if report["pfi1_change_count"] == 0:
        print("RESULT: NO_PFI1_TRANSITIONS")
        return 4
    print("RESULT: BUFFERED_PFI_AI_CLOCK_WORKS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
