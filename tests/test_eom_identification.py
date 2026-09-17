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
        signal += np.exp(-0.5 * ((x - 4000) / 12.0) ** 2)
        signal += 0.8 * np.exp(-0.5 * ((x - 4400) / 12.0) ** 2)
        result = identify_eom_aom_spectrum(
            signal,
            spacing_samples=400,
            spacing_mhz=190,
            eom_frequency_mhz=6800,
            fsr_mhz=2500,
        )
        self.assertEqual({result["carrier"]["index"], result["aom_first"]["index"]}, {4000, 4400})
        self.assertEqual({h["carrier_index"] for h in result["hypotheses"]}, {4000, 4400})
        self.assertAlmostEqual(result["fit"]["scan_amplitude_mhz"], 2375.0)
        self.assertIsNone(result["fit"]["residual_rms_mhz"])
        self.assertTrue(result["ambiguous"])
        self.assertLess(result["confidence"], 0.4)

    def test_full_folded_comb_selects_carrier_in_both_frequency_directions(self) -> None:
        x = np.arange(10000)
        s = folded_scan_coordinate(x)
        for direction in (1, -1):
            with self.subTest(direction=direction):
                carrier = 2000.0
                partner = carrier + direction * 400
                shape = lambda v: v * (v - 5000) / 5000
                pair_slope = (shape(partner) - shape(carrier)) / (partner - carrier)
                q = direction * (.475 * (s - carrier) + .025 * (
                    shape(s) - shape(carrier) - pair_slope * (s - carrier)))
                signal = np.zeros(10000)
                for line in [n * 6800 for n in range(-3, 4)] + [190]:
                    for cavity in range(-10, 11):
                        target = line + cavity * 2500
                        signal += np.exp(-0.5 * ((q - target) / 5) ** 2)
                result = identify_eom_aom_spectrum(signal, spacing_samples=400,
                    spacing_mhz=190, match_tolerance_samples=3)
                self.assertAlmostEqual(result["carrier"]["folded_coordinate"], carrier, delta=1)
                self.assertFalse(result["ambiguous"])
                self.assertEqual(len(result["centering_peaks"]), 2)
                self.assertTrue(all(p["segment"] == 1 for p in result["centering_peaks"]))
                self.assertTrue(all(abs(p["residual_samples"]) <= 3 for p in result["peaks"] if p["accepted"]))
                wrong = [h for h in result["hypotheses"] if abs(h["carrier_index"] - (7500-partner)) <= 1]
                self.assertTrue(wrong)
                self.assertGreater(wrong[0]["score"], result["hypotheses"][0]["score"] + .1)

    def test_calibration_pair_is_required_on_descending_branch(self) -> None:
        x = np.arange(10000, dtype=float)
        signal = np.exp(-0.5 * ((x - 1000) / 12.0) ** 2)
        signal += 0.8 * np.exp(-0.5 * ((x - 1400) / 12.0) ** 2)
        with self.assertRaisesRegex(ValueError, "no peak pair"):
            identify_eom_aom_spectrum(signal, spacing_samples=400, spacing_mhz=190)
