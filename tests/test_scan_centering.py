import time
import unittest
from types import SimpleNamespace

import numpy as np

from two_peak.scan_centering import ScanCenteringController, track_selected_peaks


class ScanOffsetLoopTests(unittest.TestCase):
    def test_tracks_drift_and_rejects_missing_or_merged_peaks(self):
        x = np.arange(10000)
        centers = [4200, 5600]
        for shift in range(0, 80, 4):
            values = sum(np.exp(-((x - (i + shift)) / 12) ** 2) for i in [4200, 5600])
            centers = track_selected_peaks(values, centers, 20)
            self.assertEqual(centers, [4200 + shift, 5600 + shift])
        with self.assertRaises(ValueError):
            track_selected_peaks(np.zeros(10000), centers, 20)
        with self.assertRaises(ValueError):
            track_selected_peaks(values, [centers[0], centers[0]], 20)

    def run_simulation(self, *, missing=False, stale=False):
        x = np.arange(10000)
        current = dict(amplitude_vpp=4., offset_v=2.5, frequency_hz=200.,
                       resource='fake', channel=1, waveform='RAMP', unit='VPP', symmetry_percent=50)
        writes, frames = [], []
        def apply(plan):
            self.assertEqual(plan['amplitude_vpp'], 4.)
            self.assertLessEqual(abs(plan['offset_v'] - current['offset_v']), .010001)
            current['offset_v'] = plan['offset_v']
            writes.append(dict(current))
            return dict(current)
        def frame():
            number = len(frames) + 1
            frames.append(number)
            shift = 1250 * (current['offset_v'] - 2.5)
            values = sum(np.exp(-((x - (i + shift)) / 12) ** 2) for i in [4200, 5600])
            if missing:
                values[:] = 0
            return dict(frame_id=number, finished_at=time.time() - (10 if stale else 0),
                        channels=['Dev2/ai0'], values=[values], pfi1_triggered=number == 1)
        controller = ScanCenteringController(SimpleNamespace(get_unified_ai_stream_latest_frame=frame),
                                            SimpleNamespace(read=lambda: dict(current), apply=apply))
        class FiniteRun:
            def wait(self, _interval):
                return len(frames) >= 40
            def is_set(self):
                return False
        controller._stop = FiniteRun()
        controller._update(running=True, updates=0)
        controller._run('Dev2/ai0', [4200, 5600], [2500, 7500],
                        dict(min_voltage=.01, max_voltage=5., gain=.3, max_step_v=.01, deadband_samples=2),
                        1., 20, 5, dict(current))
        return controller.status(), writes

    def test_feedback_converges_without_changing_vpp(self):
        status, writes = self.run_simulation()
        self.assertIsNone(status['error'])
        self.assertLessEqual(abs(status['error_samples']), 2)
        self.assertGreater(len(writes), 0)
        self.assertAlmostEqual(writes[-1]['offset_v'], 2.58, delta=.003)

    def test_missing_peak_or_stale_frame_never_writes(self):
        for options in [dict(missing=True), dict(stale=True)]:
            status, writes = self.run_simulation(**options)
            self.assertFalse(status['running'])
            self.assertTrue(status['error'])
            self.assertEqual(writes, [])
