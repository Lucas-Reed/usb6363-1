"""Rigol DG4000 scan settings and peak-based centering."""

import math
import threading


def plan_centering(identification: dict, source: dict, *, resize=False,
                   separation_fraction=0.4, min_voltage=0.0, max_voltage=5.0) -> dict:
    pair = identification.get("centering_peaks") or []
    if len(pair) != 2:
        raise ValueError("Select P1 and P2 on the descending scan branch first")
    first, second = identification["fit"]["breakpoints"]
    span = second - first
    amplitude, offset = float(source["amplitude_vpp"]), float(source["offset_v"])
    if source.get("unit", "").upper() != "VPP":
        raise ValueError("Set the source amplitude unit to Vpp before centering")
    if source.get("waveform", "").upper() != "RAMP" or abs(source.get("symmetry_percent", 0) - 50) > 0.01:
        raise ValueError("The calibrated triangular scan requires RAMP with 50% symmetry")
    if span <= 0 or any(not first <= p["index"] < second for p in pair):
        raise ValueError("Both selected peaks must be on the descending scan branch")
    if not all(math.isfinite(v) for v in (amplitude, offset, min_voltage, max_voltage, separation_fraction)):
        raise ValueError("Scan settings must be finite")
    if amplitude <= 0 or max_voltage <= min_voltage or not 0 < separation_fraction < 1:
        raise ValueError("Invalid scan amplitude, voltage limits or separation fraction")
    voltages = [offset + amplitude * ((second - p["index"]) / span - 0.5) for p in pair]
    center = sum(voltages) / 2
    separation_v = abs(voltages[0] - voltages[1])
    new_amplitude = min(amplitude, separation_v / separation_fraction) if resize else amplitude
    if new_amplitude <= separation_v:
        raise ValueError("Scan amplitude must span both selected peaks")
    low, high = center - new_amplitude / 2, center + new_amplitude / 2
    if low < min_voltage - 1e-9 or high > max_voltage + 1e-9:
        raise ValueError(f"Centered scan would span {low:.4f}..{high:.4f} V; adjust amplitude or voltage limits")
    predicted = [second - span * ((v - center) / new_amplitude + 0.5) for v in voltages]
    return dict(amplitude_vpp=new_amplitude, offset_v=center, minimum_v=low, maximum_v=high,
                original_amplitude_vpp=amplitude, original_offset_v=offset,
                original_frequency_hz=source["frequency_hz"], resource=source["resource"], channel=source["channel"],
                selected_indices=[p["index"] for p in pair], predicted_indices=predicted,
                selected_labels=[p.get("label", "peak") for p in pair],
                target_index=(first + second) / 2, voltage_separation_v=separation_v,
                ambiguous=bool(identification.get("ambiguous")))


class RigolScanSource:
    def __init__(self):
        self._lock = threading.Lock()
        self._manager = None
        self._instrument = None
        self.resource = ""
        self.channel = 1

    def connect(self, resource: str, channel: int = 1) -> dict:
        if not resource.strip() or channel not in (1, 2):
            raise ValueError("Enter the VISA resource and select channel 1 or 2")
        try:
            import pyvisa
        except ImportError as exc:
            raise RuntimeError("Install instrument support: python -m pip install pyvisa pyvisa-py") from exc
        with self._lock:
            self._close()
            self._manager = pyvisa.ResourceManager()
            try:
                self._instrument = self._manager.open_resource(resource.strip())
                self._instrument.timeout = 3000
                self._instrument.write_termination = "\n"
                self._instrument.read_termination = "\n"
                self.resource, self.channel = resource.strip(), channel
                status = self._read()
                if "RIGOL" not in status["identity"].upper() or "DG4" not in status["identity"].upper():
                    raise ValueError("Connected instrument is not a Rigol DG4000 series source")
                return status
            except Exception:
                self._close()
                raise

    def _close(self):
        if self._instrument is not None:
            self._instrument.close()
            self._instrument = None
        if self._manager is not None:
            self._manager.close()
            self._manager = None

    def _read(self) -> dict:
        if self._instrument is None:
            raise RuntimeError("Connect the scan source first")
        ask = self._instrument.query
        prefix = f":SOUR{self.channel}"
        waveform = ask(prefix + ":FUNC?").strip()
        return dict(identity=ask("*IDN?").strip(), resource=self.resource, channel=self.channel,
                    amplitude_vpp=float(ask(prefix + ":VOLT:AMPL?")),
                    offset_v=float(ask(prefix + ":VOLT:OFFS?")),
                    unit=ask(prefix + ":VOLT:UNIT?").strip(),
                    frequency_hz=float(ask(prefix + ":FREQ?")), waveform=waveform,
                    symmetry_percent=float(ask(prefix + ":FUNC:RAMP:SYMM?")) if waveform.upper() == "RAMP" else None)

    def read(self) -> dict:
        with self._lock:
            return self._read()

    def apply(self, plan: dict) -> dict:
        with self._lock:
            current = self._read()
            if current["resource"] != plan["resource"] or current["channel"] != plan["channel"]:
                raise ValueError("Source connection changed; preview the centering settings again")
            if current["unit"].upper() != "VPP":
                raise ValueError("Set the source amplitude unit to Vpp before centering")
            if current["waveform"].upper() not in ("RAMP", "TRI", "TRIANGLE"):
                raise ValueError("Centering requires the calibrated triangular scan")
            if current["waveform"].upper() == "RAMP":
                symmetry = float(self._instrument.query(f":SOUR{self.channel}:FUNC:RAMP:SYMM?"))
                if abs(symmetry - 50) > 0.01:
                    raise ValueError("The three-branch model requires 50% ramp symmetry")
            for key, old in (("amplitude_vpp", "original_amplitude_vpp"), ("offset_v", "original_offset_v"),
                             ("frequency_hz", "original_frequency_hz")):
                if abs(current[key] - plan[old]) > 1e-4:
                    raise ValueError("Source settings changed; calculate a new centering proposal")
            amplitude, offset = plan["amplitude_vpp"], plan["offset_v"]
            prefix = f":SOUR{self.channel}"
            # Narrow the span first so the offset change cannot widen the
            # intermediate voltage excursion. Output enable is never changed.
            if amplitude < current["amplitude_vpp"]:
                self._instrument.write(f"{prefix}:VOLT:AMPL {amplitude:.9g}")
            self._instrument.write(f"{prefix}:VOLT:OFFS {offset:.9g}")
            if amplitude > current["amplitude_vpp"]:
                self._instrument.write(f"{prefix}:VOLT:AMPL {amplitude:.9g}")
            self._instrument.query("*OPC?")
            error = self._instrument.query(":SYST:ERR?").strip()
            if int(error.split(",", 1)[0]) != 0:
                raise RuntimeError(f"DG4000 reported {error}; read source settings before retrying")
            actual = self._read()
            if abs(actual["amplitude_vpp"] - amplitude) > 1e-3 or abs(actual["offset_v"] - offset) > 1e-3:
                raise RuntimeError("Source readback differs from the proposed settings")
            return actual
