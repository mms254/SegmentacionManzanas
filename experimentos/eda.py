from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IMAGE_SCALE, RANDOM_SEED
from src.datos.loader import DatasetSample, load_dataset, stratified_split

CHANNELS = {
    "RGB": ("R", "G", "B"),
    "HSV": ("H", "S", "V"),
    "Lab": ("L*", "a*", "b*"),
    "YCrCb": ("Y", "Cr", "Cb"),
    "Gris": ("Y",),
}
_CHANNEL_ORDER = tuple(
    (color_space, channel)
    for color_space, channel_names in CHANNELS.items()
    for channel in channel_names
)


def _channel_images(image_rgb: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    conversions = {
        "RGB": image_rgb,
        "HSV": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV),
        "Lab": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB),
        "YCrCb": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb),
        "Gris": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
    }
    return {
        (color_space, channel): (
            color_image if color_image.ndim == 2 else color_image[:, :, index]
        )
        for color_space, channel_names in CHANNELS.items()
        for color_image in (conversions[color_space],)
        for index, channel in enumerate(channel_names)
    }


def _sample_class_pixels(
    class_mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.flatnonzero(class_mask)
    if indices.size == 0:
        return indices
    if indices.size > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
    return indices


def _summarize_channel(
    positive_values: np.ndarray,
    background_values: np.ndarray,
    dataset_positive_fraction: float,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray]:
    if positive_values.size == 0 or background_values.size == 0:
        raise ValueError("El EDA necesita píxeles de manzana y de fondo.")

    positive_histogram = np.bincount(positive_values, minlength=256).astype(np.int64)
    background_histogram = np.bincount(background_values, minlength=256).astype(np.int64)
    positive_count = int(positive_values.size)
    background_count = int(background_values.size)

    positive_mean = float(positive_values.mean())
    background_mean = float(background_values.mean())
    positive_variance = float(positive_values.var())
    background_variance = float(background_values.var())
    sampled_positive_fraction = positive_count / (positive_count + background_count)
    sampled_background_fraction = 1.0 - sampled_positive_fraction

    between_variance = (
        sampled_positive_fraction
        * sampled_background_fraction
        * (positive_mean - background_mean) ** 2
    )
    within_variance = (
        sampled_positive_fraction * positive_variance
        + sampled_background_fraction * background_variance
    )
    variance_ratio = (
        between_variance / within_variance
        if within_variance > 0
        else (float("inf") if between_variance > 0 else 0.0)
    )

    background_below = np.cumsum(background_histogram) - background_histogram
    auc = float(
        np.sum(
            positive_histogram
            * (background_below + 0.5 * background_histogram)
        )
        / (positive_count * background_count)
    )
    auc_best_direction = max(auc, 1.0 - auc)
    preferred_direction = "mayor" if auc >= 0.5 else "menor"

    summary: dict[str, float | int | str] = {
        "positive_pixels_sampled": positive_count,
        "background_pixels_sampled": background_count,
        "positive_mean": positive_mean,
        "positive_std": float(np.sqrt(positive_variance)),
        "background_mean": background_mean,
        "background_std": float(np.sqrt(background_variance)),
        "positive_fraction": dataset_positive_fraction,
        "positive_fraction_sampled": sampled_positive_fraction,
        "between_class_variance": between_variance,
        "within_class_variance": within_variance,
        "between_within_ratio": variance_ratio,
        "auc_positive_if_higher": auc,
        "auc_best_direction": auc_best_direction,
        "preferred_direction": preferred_direction,
    }
    return summary, positive_histogram, background_histogram


def analyze_training_samples(
    samples: list[DatasetSample],
    max_samples_per_class_per_image: int = 5000,
    seed: int = RANDOM_SEED,
    image_scale: float = IMAGE_SCALE,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    dict[str, int | float],
]:
    """Analiza canales en train con muestreo balanceado por imagen y clase."""
    if not samples:
        raise ValueError("No hay imágenes de entrenamiento para analizar.")
    if max_samples_per_class_per_image <= 0:
        raise ValueError("El máximo de muestras debe ser positivo.")
    if not 0 < image_scale <= 1:
        raise ValueError("La escala de análisis debe estar en el intervalo (0, 1].")

    rng = np.random.default_rng(seed)
    positive_values: dict[tuple[str, str], list[np.ndarray]] = {
        key: [] for key in _CHANNEL_ORDER
    }
    background_values: dict[tuple[str, str], list[np.ndarray]] = {
        key: [] for key in _CHANNEL_ORDER
    }
    total_positive_pixels = 0
    total_pixels = 0

    for sample in samples:
        image_size = (
            max(1, round(sample.image.shape[1] * image_scale)),
            max(1, round(sample.image.shape[0] * image_scale)),
        )
        image_rgb = cv2.resize(sample.image, image_size, interpolation=cv2.INTER_AREA)
        mask_u8 = cv2.resize(
            sample.mask.astype(np.uint8),
            image_size,
            interpolation=cv2.INTER_NEAREST,
        )
        apple_mask = mask_u8 > 0
        apple_indices = _sample_class_pixels(
            apple_mask, max_samples_per_class_per_image, rng
        )
        background_indices = _sample_class_pixels(
            ~apple_mask, max_samples_per_class_per_image, rng
        )
        total_positive_pixels += int(np.count_nonzero(apple_mask))
        total_pixels += int(apple_mask.size)

        for key, channel in _channel_images(image_rgb).items():
            flattened = channel.reshape(-1)
            positive_values[key].append(flattened[apple_indices])
            background_values[key].append(flattened[background_indices])

    positive_fraction = total_positive_pixels / total_pixels
    rows: list[dict[str, float | int | str]] = []
    histograms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for color_space, channel in _CHANNEL_ORDER:
        key = (color_space, channel)
        positives = np.concatenate(positive_values[key])
        backgrounds = np.concatenate(background_values[key])
        summary, positive_histogram, background_histogram = _summarize_channel(
            positives, backgrounds, positive_fraction
        )
        rows.append({"color_space": color_space, "channel": channel, **summary})
        histograms[key] = (positive_histogram, background_histogram)

    metadata: dict[str, int | float] = {
        "images": len(samples),
        "max_samples_per_class_per_image": max_samples_per_class_per_image,
        "image_scale": image_scale,
        "positive_pixels": total_positive_pixels,
        "total_pixels": total_pixels,
        "positive_fraction": positive_fraction,
    }
    return rows, histograms, metadata


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_histograms(
    histograms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    color_spaces = list(CHANNELS)
    figure, axes = plt.subplots(
        len(color_spaces), 1, figsize=(12, 3.1 * len(color_spaces)), constrained_layout=True
    )
    bins = np.arange(257)

    for axis, color_space in zip(axes, color_spaces):
        for channel in CHANNELS[color_space]:
            positive_histogram, background_histogram = histograms[
                (color_space, channel)
            ]
            positive_distribution = positive_histogram / positive_histogram.sum()
            background_distribution = background_histogram / background_histogram.sum()
            axis.plot(
                bins[:-1],
                positive_distribution,
                label=f"{channel} manzana",
            )
            axis.plot(
                bins[:-1],
                background_distribution,
                linestyle="--",
                alpha=0.8,
                label=f"{channel} fondo",
            )
        axis.set_title(color_space)
        axis.set_xlabel("Intensidad del canal (0–255)")
        axis.set_ylabel("Frecuencia relativa")
        axis.set_xlim(0, 255)
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize="small")

    figure.suptitle("Distribuciones de canales: manzana frente a fondo (train)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_eda(
    output_dir: Path,
    dataset_dir: Path | None = None,
    max_samples_per_class_per_image: int = 5000,
    seed: int = RANDOM_SEED,
    image_scale: float = IMAGE_SCALE,
) -> tuple[list[dict[str, float | int | str]], dict[str, int | float]]:
    samples = load_dataset(dataset_dir) if dataset_dir is not None else load_dataset()
    split = stratified_split(samples, seed=seed)
    samples_by_id = {sample.image_id: sample for sample in samples}
    training_samples = [samples_by_id[image_id] for image_id in split.train]
    rows, histograms, metadata = analyze_training_samples(
        training_samples,
        max_samples_per_class_per_image=max_samples_per_class_per_image,
        seed=seed,
        image_scale=image_scale,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "separabilidad_canales.csv")
    _plot_histograms(histograms, output_dir / "histogramas_canales.png")

    print(f"Imágenes analizadas (solo train): {metadata['images']}")
    print(
        "Píxeles de manzana en train: "
        f"{metadata['positive_fraction']:.2%} "
        f"({metadata['positive_pixels']}/{metadata['total_pixels']})"
    )
    print(f"Resultados: {output_dir.resolve()}")
    print("\nCanales mejor separados por AUC (dirección óptima):")
    for row in sorted(rows, key=lambda item: float(item["auc_best_direction"]), reverse=True)[:5]:
        print(
            f"  {row['color_space']}.{row['channel']}: "
            f"AUC={float(row['auc_best_direction']):.4f}, "
            f"dirección={row['preferred_direction']}, "
            f"razón inter/intraclase={float(row['between_within_ratio']):.4f}"
        )
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDA de canales de color usando solo el split de entrenamiento."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("resultados") / "eda",
        help="Directorio para el CSV y la figura.",
    )
    parser.add_argument(
        "--max-samples-per-class-per-image",
        type=int,
        default=5000,
        help="Máximo de píxeles muestreados por clase y por imagen.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--scale", type=float, default=IMAGE_SCALE)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()
    run_eda(
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        max_samples_per_class_per_image=args.max_samples_per_class_per_image,
        seed=args.seed,
        image_scale=args.scale,
    )


if __name__ == "__main__":
    main()
