from __future__ import annotations

import unittest

from two_peak.eom_identification import CalibrationModel


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
    self.assertEqual([item["frequency_offset_mhz"] for item in candidates], [-50.0, 50.0])
    self.assertTrue(all(item["in_range"] for item in candidates))


  def test_breakpoints_round_trip_through_json_shape(self) -> None:
    model = CalibrationModel(100, 200, 110.0, 80.0)
    loaded = CalibrationModel.from_dict(model.to_dict())

    self.assertEqual(loaded.breakpoints, (2500, 7500))
    self.assertAlmostEqual(loaded.frequency_for_index(200), 110.0)


  def test_invalid_model_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "different"):
        CalibrationModel(10, 10, 110.0, 80.0)
