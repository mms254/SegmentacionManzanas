"""Canales de color y mapas de "gris de manzana" (temas 2, 3 y 6).

Un "gris de manzana" es una imagen de un solo canal (uint8, 0–255) calculada a
partir del color, en la que la manzana sale clara y el fondo oscuro. El EDA mide
qué candidato separa mejor; el preprocesado (P5/P6) reutiliza estas funciones.
"""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np

# Canales por espacio de color (tema 2). "Matiz" añade el matiz circular.
CHANNELS: dict[str, tuple[str, ...]] = {
    "RGB": ("R", "G", "B"),
    "HSV": ("H", "S", "V"),
    "Lab": ("L*", "a*", "b*"),
    "YCrCb": ("Y", "Cr", "Cb"),
    "Gris": ("Y",),
    "Matiz": ("dist_rojo",),
}
CHANNEL_ORDER: tuple[tuple[str, str], ...] = tuple(
    (color_space, channel)
    for color_space, channel_names in CHANNELS.items()
    for channel in channel_names
)

# Rango de cada canal de 8 bits en OpenCV (H va de 0 a 179).
CHANNEL_RANGES: dict[str, tuple[int, int]] = {"H": (0, 180)}


def hue_distance_to_red(hue: np.ndarray) -> np.ndarray:
    """Distancia angular del matiz al rojo, reescalada a 0–255.

    En OpenCV, H ∈ [0, 180) y el rojo está en los dos extremos (≈0 y ≈179): un
    umbral sobre H en bruto parte las manzanas rojas en dos. La distancia
    circular min(H, 180 − H) ∈ [0, 90] pone el rojo en 0 y el verde lejos.
    """
    hue = np.asarray(hue, dtype=np.int16)
    distance = np.minimum(hue, 180 - hue)
    return np.round(distance * (255.0 / 90.0)).astype(np.uint8)


def channel_images(image_rgb: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    """Devuelve cada canal de CHANNELS como imagen uint8 2D."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    conversions = {
        "RGB": image_rgb,
        "HSV": hsv,
        "Lab": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB),
        "YCrCb": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb),
        "Gris": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
        "Matiz": hue_distance_to_red(hsv[:, :, 0]),
    }
    return {
        (color_space, channel): (
            conversions[color_space]
            if conversions[color_space].ndim == 2
            else conversions[color_space][:, :, index]
        )
        for color_space, channel_names in CHANNELS.items()
        for index, channel in enumerate(channel_names)
    }


# --- Candidatos fijos (sin aprendizaje) ------------------------------------------
# Todos devuelven uint8 con "mayor = más manzana", para compararlos con la misma
# métrica. Son transformaciones de intensidad sobre canales del tema 2 (tema 3).


def _lab(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)


def _ycrcb(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)


def gray_standard(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def gray_cr(image_rgb: np.ndarray) -> np.ndarray:
    return _ycrcb(image_rgb)[:, :, 1]


def gray_b_star(image_rgb: np.ndarray) -> np.ndarray:
    return _lab(image_rgb)[:, :, 2]


def gray_inverted_cb(image_rgb: np.ndarray) -> np.ndarray:
    return 255 - _ycrcb(image_rgb)[:, :, 2]


def gray_max_a_b(image_rgb: np.ndarray) -> np.ndarray:
    """max(a*, b*): cada variedad destaca en uno de los dos ejes de Lab."""
    lab = _lab(image_rgb)
    return np.maximum(lab[:, :, 1], lab[:, :, 2])


def gray_cr_minus_cb(image_rgb: np.ndarray) -> np.ndarray:
    """(Cr + (255 − Cb)) / 2: rojo y amarillo alejados del azul/verde a la vez."""
    ycrcb = _ycrcb(image_rgb).astype(np.uint16)
    return ((ycrcb[:, :, 1] + (255 - ycrcb[:, :, 2])) // 2).astype(np.uint8)


def gray_hue_red(image_rgb: np.ndarray) -> np.ndarray:
    """Matiz cercano al rojo/amarillo = claro (255 − distancia circular al rojo)."""
    hue = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[:, :, 0]
    return 255 - hue_distance_to_red(hue)


FIXED_CANDIDATES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Gris estándar Y": gray_standard,
    "Cr (YCrCb)": gray_cr,
    "b* (Lab)": gray_b_star,
    "255 − Cb (YCrCb)": gray_inverted_cb,
    "max(a*, b*)": gray_max_a_b,
    "(Cr + 255 − Cb) / 2": gray_cr_minus_cb,
    "Matiz cercano al rojo": gray_hue_red,
}


# --- Candidato aprendido: retroproyección del histograma (tema 6) ---------------

_COLOR_SPACES = {
    "Lab": cv2.COLOR_RGB2LAB,
    "HSV": cv2.COLOR_RGB2HSV,
    "YCrCb": cv2.COLOR_RGB2YCrCb,
}


class HistogramBackprojection:
    """Gris de manzana = P(manzana | color) leído de histogramas 2D normalizados.

    `fit` acumula, en las imágenes de entrenamiento, el histograma 2D de los
    píxeles de manzana y el de fondo (cv2.calcHist con la máscara). Cada uno se
    normaliza a frecuencias relativas (tema 6: el histograma normalizado se
    interpreta como probabilidad) y se combinan con priors iguales:

        P(manzana | c) = h_m(c) / (h_m(c) + h_f(c))

    `transform` aplica cv2.calcBackProject: cada píxel recibe el valor de su
    celda, escalado a 0–255. Colores no vistos en train quedan a 0 (fondo).
    """

    def __init__(
        self,
        color_space: str = "Lab",
        channels: Sequence[int] = (1, 2),
        bins: int = 32,
    ) -> None:
        if color_space not in _COLOR_SPACES:
            raise ValueError(f"Espacio de color no soportado: {color_space}")
        if len(channels) != 2:
            raise ValueError("La retroproyección usa exactamente dos canales.")
        self.color_space = color_space
        self.channels = list(channels)
        self.bins = int(bins)
        self.ranges: list[float] = []
        for channel in self.channels:
            upper = 180 if color_space == "HSV" and channel == 0 else 256
            self.ranges.extend([0, upper])
        self.probability_: np.ndarray | None = None

    def _convert(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, _COLOR_SPACES[self.color_space])

    def fit(
        self, images_rgb: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> "HistogramBackprojection":
        if len(images_rgb) == 0 or len(images_rgb) != len(masks):
            raise ValueError("Se necesitan imágenes y máscaras emparejadas.")
        hist_size = [self.bins, self.bins]
        apple = np.zeros(hist_size, dtype=np.float64)
        background = np.zeros(hist_size, dtype=np.float64)
        for image_rgb, mask in zip(images_rgb, masks):
            converted = self._convert(image_rgb)
            mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8)
            apple += cv2.calcHist(
                [converted], self.channels, mask_u8, hist_size, self.ranges
            )
            background += cv2.calcHist(
                [converted], self.channels, 1 - mask_u8, hist_size, self.ranges
            )
        if apple.sum() == 0 or background.sum() == 0:
            raise ValueError("El entrenamiento necesita píxeles de manzana y de fondo.")
        apple /= apple.sum()
        background /= background.sum()
        total = apple + background
        probability = np.divide(apple, total, out=np.zeros_like(apple), where=total > 0)
        self.probability_ = probability
        return self

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.probability_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        lookup = (self.probability_ * 255.0).astype(np.float32)
        return cv2.calcBackProject(
            [self._convert(image_rgb)], self.channels, lookup, self.ranges, 1
        )

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.transform(image_rgb)


LEARNED_CANDIDATES: dict[str, dict[str, object]] = {
    "Retroproyección a*–b* (Lab)": {"color_space": "Lab", "channels": (1, 2)},
    "Retroproyección H–S (HSV)": {"color_space": "HSV", "channels": (0, 1)},
    "Retroproyección Cr–Cb (YCrCb)": {"color_space": "YCrCb", "channels": (1, 2)},
}
