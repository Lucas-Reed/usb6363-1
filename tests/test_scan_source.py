import unittest

from two_peak.scan_source import RigolScanSource, plan_centering


class ScanCenteringTests(unittest.TestCase):
    def setUp(self):
        self.identification = dict(fit=dict(breakpoints=[2500, 7500]),
                                   centering_peaks=[dict(index=4000), dict(index=5000)])
        self.source = dict(amplitude_vpp=5, offset_v=2.5, frequency_hz=200,
                           resource="fake", channel=1, waveform="RAMP", unit="VPP", symmetry_percent=50)

    def test_voltage_mapping_centers_pair_without_inventing_new_peak_spacing(self):
        plan = plan_centering(self.identification, self.source, max_voltage=6)
        self.assertEqual(plan["amplitude_vpp"], 5)
        self.assertEqual(plan["offset_v"], 3)
        self.assertEqual(plan["predicted_indices"], [4500, 5500])

    def test_optional_narrowing_fits_pair_in_original_voltage_range(self):
        plan = plan_centering(self.identification, self.source, resize=True)
        self.assertEqual(plan["amplitude_vpp"], 2.5)
        self.assertEqual(plan["predicted_indices"], [4000, 6000])
        self.assertEqual(plan["minimum_v"], 1.75)
        self.assertEqual(plan["maximum_v"], 4.25)

    def test_apply_changes_amplitude_before_offset_and_reads_back(self):
        plan = plan_centering(self.identification, self.source, resize=True)
        instrument = RigolScanSource()
        writes = []
        class FakeVisa:
            def write(_self, command):
                writes.append(command)
            def query(_self, command):
                return {":SOUR1:FUNC:RAMP:SYMM?": "50", "*OPC?": "1", ":SYST:ERR?": '0,"No error"'}[command]
        instrument._instrument = FakeVisa()
        snapshots = iter([self.source, dict(self.source, amplitude_vpp=2.5, offset_v=3)])
        instrument._read = lambda: next(snapshots)
        actual = instrument.apply(plan)
        self.assertEqual(writes, [":SOUR1:VOLT:AMPL 2.5", ":SOUR1:VOLT:OFFS 3"])
        self.assertEqual(actual["offset_v"], 3)
