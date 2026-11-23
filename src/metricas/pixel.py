from __future__ import annotations

from math import sqrt

import numpy as np


def confusion_counts(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, int]:
    ground_truth = np.asarray(ground_truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Las formas deben coincidir: ground_truth={ground_truth.shape}, "
            f"prediction={prediction.shape}"
        )
    if ground_truth.size == 0:
        raise ValueError("No se pueden calcular métricas sobre máscaras vacías.")

    true_positive = int(np.count_nonzero(ground_truth & prediction))
    false_positive = int(np.count_nonzero(~ground_truth & prediction))
    true_negative = int(np.count_nonzero(~ground_truth & ~prediction))
    false_negative = int(np.count_nonzero(ground_truth & ~prediction))
    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


def _divide_or_zero(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def pixel_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    """Calcula métricas binarias de píxel sin depender de scikit-learn."""
    counts = confusion_counts(ground_truth, prediction)
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total
    precision = _divide_or_zero(tp, tp + fp)
    recall = _divide_or_zero(tp, tp + fn)
    specificity = _divide_or_zero(tn, tn + fp)
    f1 = _divide_or_zero(2 * tp, 2 * tp + fp + fn)
    iou = _divide_or_zero(tp, tp + fp + fn)

    mcc_denominator = sqrt(
        float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = _divide_or_zero(tp * tn - fp * fn, mcc_denominator)
    expected_agreement = (
        (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)
    ) / (total * total)
    cohen_kappa = _divide_or_zero(accuracy - expected_agreement, 1 - expected_agreement)

    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": 1.0 - specificity,
        "f1": f1,
        "dice": f1,
        "iou": iou,
        "mcc": mcc,
        "balanced_accuracy": (recall + specificity) / 2,
        "cohen_kappa": cohen_kappa,
    }
