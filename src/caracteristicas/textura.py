"""Características de textura por píxel (temas 3 y 4).

Todas reciben una imagen en escala de grises (uint8) y devuelven un mapa float
del mismo tamaño. Se usan en el EDA y servirán como características para los
modelos supervisados (KNN, SVM, Random Forest).

- desviacion_local: desviación típica en una ventana (filtro de media, tema 3).
- magnitud_sobel: magnitud del gradiente con Sobel 3×3 (tema 4).
- magnitud_log: |Laplaciana de la Gaussiana| (tema 4).
- codigos_lbp / contraste_lbp: Local Binary Pattern de 8 vecinos y su medida de
  contraste, tal como se definen en el tema 4.
- RetroproyeccionLBP: P(manzana | código LBP) aprendida en train con histogramas
  normalizados (misma idea que la retroproyección de color del tema 6).
"""

from __future__ import annotations

import cv2
import numpy as np

# Desplazamientos (fila, columna) de los 8 vecinos, en sentido horario desde
# la esquina superior izquierda. El vecino i aporta el bit de peso 2**(7 - i).
VECINOS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def media_local(mapa: np.ndarray, ventana: int) -> np.ndarray:
    """Media en una ventana cuadrada (filtro de media del tema 3)."""
    return cv2.blur(np.asarray(mapa, dtype=np.float64), (ventana, ventana))


def desviacion_local(gris: np.ndarray, ventana: int) -> np.ndarray:
    """Desviación típica local: sqrt(media(I²) − media(I)²)."""
    imagen = np.asarray(gris, dtype=np.float64)
    media = cv2.blur(imagen, (ventana, ventana))
    media_cuadrados = cv2.blur(imagen * imagen, (ventana, ventana))
    varianza = np.maximum(media_cuadrados - media * media, 0.0)
    return np.sqrt(varianza)


def magnitud_sobel(gris: np.ndarray) -> np.ndarray:
    """Magnitud del gradiente sqrt(Gx² + Gy²) con operadores de Sobel 3×3."""
    imagen = np.asarray(gris, dtype=np.float64)
    gradiente_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    gradiente_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gradiente_x * gradiente_x + gradiente_y * gradiente_y)


def magnitud_log(gris: np.ndarray, sigma: float) -> np.ndarray:
    """Valor absoluto de la Laplaciana de la Gaussiana (suavizado + Laplace)."""
    suavizada = cv2.GaussianBlur(np.asarray(gris, dtype=np.float64), (0, 0), sigma)
    return np.abs(cv2.Laplacian(suavizada, cv2.CV_64F, ksize=3))


def _vecinos(gris: np.ndarray) -> list[np.ndarray]:
    """Los 8 vecinos de cada píxel (bordes replicados), como imágenes float."""
    imagen = np.asarray(gris, dtype=np.float64)
    ampliada = cv2.copyMakeBorder(imagen, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    alto, ancho = imagen.shape
    vecinos = []
    for desplazamiento_fila, desplazamiento_columna in VECINOS:
        inicio_fila = 1 + desplazamiento_fila
        inicio_columna = 1 + desplazamiento_columna
        vecinos.append(ampliada[inicio_fila : inicio_fila + alto, inicio_columna : inicio_columna + ancho])
    return vecinos


def codigos_lbp(gris: np.ndarray) -> np.ndarray:
    """Código LBP (0–255) de cada píxel: bit 1 si el vecino es ≥ que el centro."""
    centro = np.asarray(gris, dtype=np.float64)
    codigos = np.zeros(centro.shape, dtype=np.int32)
    vecinos = _vecinos(gris)
    for indice in range(8):
        bit = (vecinos[indice] >= centro).astype(np.int32)
        codigos += bit * (2 ** (7 - indice))
    return codigos.astype(np.uint8)


def contraste_lbp(gris: np.ndarray) -> np.ndarray:
    """Medida de contraste del LBP (tema 4).

    Media de los vecinos ≥ centro menos media de los vecinos < centro. Si todos
    los vecinos están en el mismo grupo, el contraste es 0.
    """
    centro = np.asarray(gris, dtype=np.float64)
    suma_altos = np.zeros(centro.shape)
    cuenta_altos = np.zeros(centro.shape)
    suma_bajos = np.zeros(centro.shape)
    cuenta_bajos = np.zeros(centro.shape)
    for vecino in _vecinos(gris):
        alto = vecino >= centro
        suma_altos += np.where(alto, vecino, 0.0)
        cuenta_altos += alto
        suma_bajos += np.where(alto, 0.0, vecino)
        cuenta_bajos += ~alto
    contraste = np.zeros(centro.shape)
    hay_ambos = (cuenta_altos > 0) & (cuenta_bajos > 0)
    contraste[hay_ambos] = (
        suma_altos[hay_ambos] / cuenta_altos[hay_ambos]
        - suma_bajos[hay_ambos] / cuenta_bajos[hay_ambos]
    )
    return contraste


class RetroproyeccionLBP:
    """P(manzana | código LBP), aprendida con los histogramas de train.

    `fit` cuenta los códigos LBP de los píxeles de manzana y de fondo, normaliza
    cada histograma y calcula h_manzana / (h_manzana + h_fondo). `transform`
    sustituye cada código por esa probabilidad y la promedia en una ventana,
    porque un solo código es muy ruidoso y la textura es una propiedad de la
    vecindad.
    """

    def __init__(self, ventana: int = 9) -> None:
        self.ventana = ventana
        self.probabilidad = None

    def fit(self, grises, mascaras) -> "RetroproyeccionLBP":
        histograma_manzana = np.zeros(256)
        histograma_fondo = np.zeros(256)
        for gris, mascara in zip(grises, mascaras):
            codigos = codigos_lbp(gris)
            mascara = np.asarray(mascara, dtype=bool)
            histograma_manzana += np.bincount(codigos[mascara], minlength=256)
            histograma_fondo += np.bincount(codigos[~mascara], minlength=256)
        if histograma_manzana.sum() == 0 or histograma_fondo.sum() == 0:
            raise ValueError("Se necesitan píxeles de manzana y de fondo.")
        histograma_manzana /= histograma_manzana.sum()
        histograma_fondo /= histograma_fondo.sum()
        total = histograma_manzana + histograma_fondo
        self.probabilidad = np.zeros(256)
        con_datos = total > 0
        self.probabilidad[con_datos] = histograma_manzana[con_datos] / total[con_datos]
        return self

    def transform(self, gris: np.ndarray) -> np.ndarray:
        if self.probabilidad is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        mapa = self.probabilidad[codigos_lbp(gris)]
        return media_local(mapa, self.ventana)
