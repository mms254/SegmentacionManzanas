"""Curvas ROC y Precision-Recall a partir de mapas de puntuación discretos.

Si el mapa de puntuación es uint8 (0–255), basta con los histogramas de
puntuaciones de píxeles positivos y negativos para obtener las curvas exactas
sobre millones de píxeles sin ordenarlos. Implementación propia (sin sklearn).
"""

from __future__ import annotations

import numpy as np


def score_histograms(
    scores: np.ndarray, ground_truth: np.ndarray, levels: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Histogramas de puntuación (enteros en [0, levels)) de positivos y negativos."""
    scores = np.asarray(scores)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if scores.shape != ground_truth.shape:
        raise ValueError(
            f"Las formas deben coincidir: scores={scores.shape}, "
            f"ground_truth={ground_truth.shape}"
        )
    flat = scores.reshape(-1).astype(np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= levels):
        raise ValueError(f"Las puntuaciones deben estar en [0, {levels}).")
    labels = ground_truth.reshape(-1)
    positive = np.bincount(flat[labels], minlength=levels).astype(np.int64)
    negative = np.bincount(flat[~labels], minlength=levels).astype(np.int64)
    return positive, negative


def curves_from_histograms(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> dict[str, np.ndarray | float]:
    """ROC, PR, AUC (trapecios) y Average Precision.

    El umbral t recorre los niveles de mayor a menor; se predice manzana si
    puntuación ≥ t. Los empates dentro de un nivel se tratan juntos, como en
    cualquier implementación exacta.
    """
    positive = np.asarray(positive_histogram, dtype=np.float64)
    negative = np.asarray(negative_histogram, dtype=np.float64)
    total_positive = positive.sum()
    total_negative = negative.sum()
    if total_positive == 0 or total_negative == 0:
        raise ValueError("Se necesitan píxeles positivos y negativos.")

    true_positive = np.cumsum(positive[::-1])
    false_positive = np.cumsum(negative[::-1])
    tpr = np.concatenate(([0.0], true_positive / total_positive))
    fpr = np.concatenate(([0.0], false_positive / total_negative))
    predicted = true_positive + false_positive
    precision = np.divide(
        true_positive, predicted, out=np.ones_like(true_positive), where=predicted > 0
    )

    roc_auc = float(np.trapezoid(tpr, fpr))
    recall_steps = np.diff(tpr)
    average_precision = float(np.sum(recall_steps * precision))
    f1 = np.divide(
        2 * precision * tpr[1:],
        precision + tpr[1:],
        out=np.zeros_like(precision),
        where=(precision + tpr[1:]) > 0,
    )
    best = int(np.argmax(f1))
    thresholds = np.arange(len(positive))[::-1]
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": tpr[1:],
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "prevalence": float(total_positive / (total_positive + total_negative)),
        "best_f1": float(f1[best]),
        "best_f1_threshold": int(thresholds[best]),
    }
