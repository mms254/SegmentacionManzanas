import unittest

import numpy as np

from src.metricas.pixel import confusion_counts, pixel_metrics


class PixelMetricsTests(unittest.TestCase):
    def test_perfect_prediction_has_perfect_overlap_metrics(self) -> None:
        mask = np.array([[True, False], [True, False]])

        metrics = pixel_metrics(mask, mask.copy())

        self.assertEqual(confusion_counts(mask, mask), {"tp": 2, "fp": 0, "tn": 2, "fn": 0})
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["mcc"], 1.0)
        self.assertEqual(metrics["cohen_kappa"], 1.0)

    def test_all_background_prediction_does_not_hide_missed_apples(self) -> None:
        ground_truth = np.array([[True, False], [False, False]])
        prediction = np.zeros_like(ground_truth)

        metrics = pixel_metrics(ground_truth, prediction)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["iou"], 0.0)

    def test_mcc_handles_confusion_counts_at_dataset_image_scale(self) -> None:
        ground_truth = np.zeros((1280, 720), dtype=bool)
        ground_truth[:200, :200] = True

        metrics = pixel_metrics(ground_truth, ground_truth.copy())

        self.assertEqual(metrics["mcc"], 1.0)

    def test_rejects_incompatible_shapes_and_empty_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "formas deben coincidir"):
            confusion_counts(np.zeros((2, 2)), np.zeros((4,)))
        with self.assertRaisesRegex(ValueError, "máscaras vacías"):
            pixel_metrics(np.zeros((0,)), np.zeros((0,)))


if __name__ == "__main__":
    unittest.main()
