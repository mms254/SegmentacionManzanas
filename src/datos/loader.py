from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    CACHE_DIR,
    DATASET_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    YELLOW_IMAGE_MAX_INDEX,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IMAGE_ID_PATTERN = re.compile(r"(\d+)$")


def _decode_image(path: Path, read_flags: int) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, read_flags)
    if decoded is None:
        raise ValueError(f"No se pudo leer el archivo de imagen: {path}")
    return decoded


@dataclass(frozen=True)
class DatasetSample:
    image_id: str
    image: np.ndarray
    mask: np.ndarray
    variety: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


def _read_or_cache(
    source_path: Path,
    cache_path: Path,
    read_flags: int,
    convert_image_to_rgb: bool = False,
) -> np.ndarray:
    if cache_path.is_file() and cache_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        cached = np.load(cache_path, allow_pickle=False)
        if cached.ndim == (3 if convert_image_to_rgb else 2):
            return cached

    decoded = _decode_image(source_path, read_flags)
    if convert_image_to_rgb:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp.npy")
    with temporary_path.open("wb") as cache_file:
        np.save(cache_file, decoded, allow_pickle=False)
    temporary_path.replace(cache_path)
    return decoded


def _variety_for(image_id: str) -> str:
    match = _IMAGE_ID_PATTERN.search(image_id)
    if match is None:
        raise ValueError(
            f"El identificador '{image_id}' no termina en un índice numérico; "
            "no se puede determinar la variedad a partir del orden del dataset."
        )
    index = int(match.group(1))
    return "amarilla" if index <= YELLOW_IMAGE_MAX_INDEX else "roja"


def load_dataset(
    dataset_dir: Path = DATASET_DIR,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
) -> list[DatasetSample]:
    """Carga pares imagen-máscara; entrega imágenes RGB y máscaras booleanas."""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagenes"
    masks_dir = dataset_dir / "mascaras"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Se esperaban las carpetas 'imagenes' y 'mascaras' en: {dataset_dir}"
        )

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )
    mask_paths = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    }
    image_stems = [path.stem for path in image_paths]
    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en: {images_dir}")
    if len(set(image_stems)) != len(image_stems):
        raise ValueError(f"Hay identificadores de imagen duplicados en: {images_dir}")

    missing_masks = sorted(set(image_stems) - set(mask_paths))
    orphan_masks = sorted(set(mask_paths) - set(image_stems))
    if missing_masks or orphan_masks:
        raise ValueError(
            f"Los pares imagen-máscara no coinciden. "
            f"Sin máscara: {missing_masks}; sin imagen: {orphan_masks}"
        )

    samples: list[DatasetSample] = []
    expected_shape: tuple[int, int] | None = None
    for image_path in image_paths:
        mask_path = mask_paths[image_path.stem]
        if use_cache:
            image_cache_path = cache_dir / "imagenes" / f"{image_path.stem}.npy"
            mask_cache_path = cache_dir / "mascaras" / f"{mask_path.stem}.npy"
            image = _read_or_cache(image_path, image_cache_path, cv2.IMREAD_COLOR, True)
            raw_mask = _read_or_cache(mask_path, mask_cache_path, cv2.IMREAD_GRAYSCALE)
        else:
            image_bgr = _decode_image(image_path, cv2.IMREAD_COLOR)
            raw_mask = _decode_image(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if image.shape[:2] != raw_mask.shape:
            raise ValueError(
                f"Dimensiones incompatibles para '{image_path.stem}': "
                f"imagen {image.shape[:2]}, máscara {raw_mask.shape}"
            )
        if expected_shape is None:
            expected_shape = image.shape[:2]
        elif image.shape[:2] != expected_shape:
            raise ValueError(
                f"Resolución inconsistente en '{image_path.stem}': "
                f"esperada {expected_shape}, obtenida {image.shape[:2]}"
            )

        samples.append(
            DatasetSample(
                image_id=image_path.stem,
                image=image,
                mask=raw_mask > 127,
                variety=_variety_for(image_path.stem),
            )
        )
    return samples


def stratified_split(
    samples: list[DatasetSample],
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> DatasetSplit:
    """Divide por imagen, manteniendo las proporciones dentro de cada variedad."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Las proporciones deben ser positivas y sumar 1.")

    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(sample.variety, []).append(sample.image_id)
    if not groups:
        raise ValueError("No se puede dividir un dataset vacío.")

    rng = np.random.default_rng(seed)
    split_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for variety in sorted(groups):
        ids = np.asarray(sorted(groups[variety]))
        if len(ids) < 3:
            raise ValueError(
                f"La variedad '{variety}' necesita al menos 3 imágenes para "
                "estar representada en train, validation y test."
            )
        shuffled_ids = rng.permutation(ids)
        train_count = int(len(ids) * train_ratio)
        validation_count = int(len(ids) * validation_ratio)
        if train_count == 0 or validation_count == 0 or train_count + validation_count >= len(ids):
            raise ValueError(
                f"No se pueden representar los tres splits en la variedad '{variety}' "
                f"con {len(ids)} imágenes y las proporciones indicadas."
            )
        split_ids["train"].extend(shuffled_ids[:train_count].tolist())
        split_ids["validation"].extend(
            shuffled_ids[train_count : train_count + validation_count].tolist()
        )
        split_ids["test"].extend(shuffled_ids[train_count + validation_count :].tolist())

    return DatasetSplit(
        train=tuple(sorted(split_ids["train"])),
        validation=tuple(sorted(split_ids["validation"])),
        test=tuple(sorted(split_ids["test"])),
    )
