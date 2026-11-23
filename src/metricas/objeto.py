"""Medidas por objeto: componentes conexas de una máscara (tema 5).

Se usa en el EDA para describir las manzanas anotadas y, más adelante, para las
métricas por manzana (emparejamiento por IoU, conteo) y el filtro de forma del
postprocesado.
"""

from __future__ import annotations

import cv2
import numpy as np


def _shape_measures(component: np.ndarray) -> tuple[float, float]:
    """Circularidad 4πA/P² y solidez A/A_convexa del contorno externo."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    circularity = (
        4 * np.pi * contour_area / perimeter**2
        if contour_area > 0 and perimeter > 0
        else float("nan")
    )
    solidity = contour_area / hull_area if hull_area > 0 else float("nan")
    return circularity, solidity


_DISC_REFERENCE: dict[float, tuple[float, float]] = {}


def disc_reference(radius: float) -> tuple[float, float]:
    """Circularidad y solidez de un disco DIGITAL perfecto del mismo radio.

    En contornos pixelados el perímetro se sobreestima (escalones), así que un
    círculo perfecto de radio 12 px da 4πA/P² ≈ 0,86, no 1. Dividir por esta
    referencia da medidas normalizadas: 1 = tan redondo como un disco digital.
    """
    key = round(radius * 2) / 2
    if key not in _DISC_REFERENCE:
        size = int(2 * key + 7)
        disc = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(disc, (size // 2, size // 2), max(1, int(round(key))), 1, -1)
        _DISC_REFERENCE[key] = _shape_measures(disc)
    return _DISC_REFERENCE[key]


def component_stats(mask: np.ndarray) -> list[dict[str, float]]:
    """Área, radio equivalente, circularidad y solidez de cada componente (8-vecindad)."""
    mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    rows: list[dict[str, float]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        component = (labels[y : y + height, x : x + width] == label).astype(np.uint8)
        circularity, solidity = _shape_measures(component)
        radius = float(np.sqrt(area / np.pi))
        reference_circularity, reference_solidity = disc_reference(radius)
        rows.append(
            {
                "area": float(area),
                "radio_equivalente": radius,
                "ancho": float(width),
                "alto": float(height),
                "circularidad": circularity,
                "solidez": solidity,
                "circularidad_norm": circularity / reference_circularity,
                "solidez_norm": solidity / reference_solidity,
            }
        )
    return rows
