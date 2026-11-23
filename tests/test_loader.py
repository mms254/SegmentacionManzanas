import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.datos.loader import DatasetSample, _variety_for, load_dataset, stratified_split


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.images_dir = self.root / "imagenes"
        self.masks_dir = self.root / "mascaras"
        self.images_dir.mkdir()
        self.masks_dir.mkdir()

    def _write_pair(self, image_id: str, color: tuple[int, int, int]) -> None:
        image = np.full((4, 5, 3), color, dtype=np.uint8)
        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 255
        self.assertTrue(cv2.imwrite(str(self.images_dir / f"{image_id}.png"), image))
        self.assertTrue(cv2.imwrite(str(self.masks_dir / f"{image_id}.png"), mask))

    def test_loads_rgb_images_and_binary_masks_from_cache(self) -> None:
        self._write_pair("img_000", (10, 20, 30))
        samples = load_dataset(self.root, self.root / "cache")
        cached_samples = load_dataset(self.root, self.root / "cache")

        self.assertEqual(len(samples), 1)
        np.testing.assert_array_equal(samples[0].image[0, 0], [30, 20, 10])
        self.assertEqual(samples[0].mask.dtype, np.bool_)
        self.assertEqual(int(samples[0].mask.sum()), 6)
        np.testing.assert_array_equal(samples[0].image, cached_samples[0].image)
        self.assertEqual(samples[0].variety, "amarilla")

    def test_rejects_missing_mask(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.images_dir / "img_000.png"), image))

        with self.assertRaisesRegex(ValueError, "no coinciden"):
            load_dataset(self.root, self.root / "cache")

    def test_split_is_deterministic_and_stratified(self) -> None:
        samples = []
        for index in range(100):
            variety = "amarilla" if index < 50 else "roja"
            samples.append(
                DatasetSample(
                    image_id=f"img_{index:03d}",
                    image=np.zeros((1, 1, 3), dtype=np.uint8),
                    mask=np.zeros((1, 1), dtype=bool),
                    variety=variety,
                )
            )

        split = stratified_split(samples, seed=7)
        repeated_split = stratified_split(samples, seed=7)
        self.assertEqual(split, repeated_split)
        self.assertEqual(
            (len(split.train), len(split.validation), len(split.test)), (60, 20, 20)
        )
        for ids in (split.train, split.validation, split.test):
            yellow_count = sum(int(image_id[-3:]) < 50 for image_id in ids)
            self.assertEqual(yellow_count * 2, len(ids))

    def test_split_keeps_60_20_20_with_unbalanced_varieties(self) -> None:
        samples = [
            DatasetSample(
                image_id=f"img_{index:03d}",
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                mask=np.zeros((1, 1), dtype=bool),
                variety="amarilla" if index <= 65 else "roja",
            )
            for index in range(100)
        ]

        split = stratified_split(samples, seed=42)

        self.assertEqual(
            (len(split.train), len(split.validation), len(split.test)), (60, 20, 20)
        )
        all_ids = split.train + split.validation + split.test
        self.assertEqual(len(set(all_ids)), 100)
        for ids in (split.train, split.validation, split.test):
            red_count = sum(int(image_id[-3:]) > 65 for image_id in ids)
            self.assertGreaterEqual(red_count, 6)

    def test_real_dataset_variety_boundary(self) -> None:
        self.assertEqual(_variety_for("img_065"), "amarilla")
        self.assertEqual(_variety_for("img_066"), "roja")


if __name__ == "__main__":
    unittest.main()
