"""Estimación y simulación de ruido (tema 3).

La estimación se usa en el EDA para saber cuánto ruido tienen las imágenes
reales; la simulación servirá en el experimento de robustez del preprocesado.
"""

from __future__ import annotations

import cv2
import numpy as np

IMPULSE_DIFF = 80

_IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimate_gaussian_noise(gray: np.ndarray) -> float:
    """σ del ruido gaussiano (Immerkær, 1996).

    Convoluciona con una máscara laplaciana de suma cero: en zonas suaves la
    respuesta es casi solo ruido. σ = sqrt(π/2) · Σ|I * N| / (6 (W−2)(H−2)).
    """
    gray = np.asarray(gray, dtype=np.float64)
    height, width = gray.shape
    if height < 3 or width < 3:
        raise ValueError("La imagen es demasiado pequeña para estimar el ruido.")
    response = cv2.filter2D(gray, -1, _IMMERKAER_KERNEL, borderType=cv2.BORDER_REFLECT)
    response = response[1:-1, 1:-1]
    return float(
        np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (width - 2) * (height - 2))
    )


def impulse_noise_fraction(gray: np.ndarray, threshold: int = IMPULSE_DIFF) -> float:
    """Fracción de píxeles casi negros/blancos que difieren mucho de su mediana 3×3."""
    gray = np.asarray(gray, dtype=np.uint8)
    median = cv2.medianBlur(gray, 3)
    isolated = np.abs(gray.astype(np.int16) - median.astype(np.int16)) > threshold
    extreme = (gray <= 10) | (gray >= 245)
    return float(np.mean(isolated & extreme))


def add_salt_and_pepper(
    image: np.ndarray, amount: float, rng: np.random.Generator
) -> np.ndarray:
    """Pone a 0 o 255 (en todos los canales) una fracción `amount` de píxeles."""
    if not 0 <= amount <= 1:
        raise ValueError("amount debe estar en [0, 1].")
    noisy = np.array(image, copy=True)
    positions = rng.random(image.shape[:2]) < amount
    values = rng.choice(np.array([0, 255], dtype=image.dtype), size=int(positions.sum()))
    noisy[positions] = values[:, None] if image.ndim == 3 else values
    return noisy


def add_gaussian_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Suma ruido N(0, σ²) y recorta a [0, 255]."""
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)
