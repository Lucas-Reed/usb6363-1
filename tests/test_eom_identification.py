from __future__ import annotations

import unittest

import numpy as np

from two_peak.eom_identification import (
    CalibrationModel,
    folded_scan_coordinate,
    identify_eom_aom_spectrum,
)


class EomIdentificationTests(unittest.TestCase):
    def test_candidate_sidebands_use_manual_frequency_scale(self) -> None:
        model = CalibrationModel(
            carrier_index=500,
            aom_zero_index=700,
            frequency_difference_mhz=100.0,
            fsr_mhz=50.0,
        )

        candidates = model.candidate_sidebands(sample_count=2000, orders=(-1, 1))

        self.assertEqual([item["index"] for item in candidates], [400, 600])
        self.assertEqual(
            [item["frequency_offset_mhz"] for item in candidates], [-50.0, 50.0]
        )
        self.assertTrue(all(item["in_range"] for item in candidates))

    def test_breakpoints_round_trip_through_json_shape(self) -> None:
        model = CalibrationModel(100, 200, 110.0, 80.0)
        loaded = CalibrationModel.from_dict(model.to_dict())

        self.assertEqual(loaded.breakpoints, (2500, 7500))
        self.assertAlmostEqual(loaded.frequency_for_index(200), 110.0)

    def test_invalid_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            CalibrationModel(10, 10, 110.0, 80.0)

    def test_folded_scan_uses_three_physical_branches(self) -> None:
        folded = folded_scan_coordinate([0, 2500, 7500, 10000])
        np.testing.assert_allclose(folded, [2500, 5000, 0, 2500])

    def test_automatic_identification_uses_spacing_pair(self) -> None:
        x = np.arange(10000, dtype=float)
        signal = np.zeros(10000, dtype=float)
        signal += np.exp(-0.5 * ((x - 1000) / 12.0) ** 2)
        signal += 0.8 * np.exp(-0.5 * ((x - 1400) / 12.0) ** 2)
        result = identify_eom_aom_spectrum(
            signal,
            spacing_samples=400,
            spacing_mhz=190,
            eom_frequency_mhz=6800,
            fsr_mhz=2500,
        )
        self.assertEqual(result["carrier"]["index"], 1000)
        self.assertEqual(result["aom_first"]["index"], 1400)
        self.assertAlmostEqual(result["fit"]["scan_amplitude_mhz"], 2375.0)
        self.assertIsNone(result["fit"]["residual_rms_mhz"])
        self.assertTrue(result["ambiguous"])
        self.assertLess(result["confidence"], 0.4)
