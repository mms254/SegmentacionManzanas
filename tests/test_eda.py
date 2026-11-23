import unittest

import numpy as np

from experimentos.eda import _summarize_channel


class EdaStatisticsTests(unittest.TestCase):
    def test_auc_and_variance_separability_for_disjoint_channels(self) -> None:
        positives = np.array([200, 201, 202], dtype=np.uint8)
        backgrounds = np.array([10, 11, 12], dtype=np.uint8)

        summary, positive_histogram, background_histogram = _summarize_channel(
            positives, backgrounds, dataset_positive_fraction=0.1
        )

        self.assertEqual(summary["auc_positive_if_higher"], 1.0)
        self.assertEqual(summary["auc_best_direction"], 1.0)
        self.assertEqual(summary["preferred_direction"], "mayor")
        self.assertEqual(summary["positive_fraction"], 0.1)
        self.assertEqual(summary["positive_fraction_sampled"], 0.5)
        self.assertGreater(summary["between_within_ratio"], 0)
        self.assertEqual(int(positive_histogram.sum()), 3)
        self.assertEqual(int(background_histogram.sum()), 3)

    def test_auc_accounts_for_ties_and_reverses_for_lower_positive_values(self) -> None:
        positives = np.array([1, 2], dtype=np.uint8)
        backgrounds = np.array([2, 3], dtype=np.uint8)

        summary, _, _ = _summarize_channel(
            positives, backgrounds, dataset_positive_fraction=0.5
        )

        self.assertEqual(summary["auc_positive_if_higher"], 0.125)
        self.assertEqual(summary["auc_best_direction"], 0.875)
        self.assertEqual(summary["preferred_direction"], "menor")


if __name__ == "__main__":
    unittest.main()
