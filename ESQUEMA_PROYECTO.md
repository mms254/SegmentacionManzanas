# Esquema conceptual: segmentación de manzanas (VC-I, 2026/27)

> Documento de planificación previo a la implementación. 
> **Grupo:** María Mestre Sánchez · Elena Fernández Cebrián · Alejandro Operé Sirvent. 

## 0. Qué es el problema (según los datos)

| Aspecto | Observación sobre `dataset_manzanas/` | Consecuencia de diseño |
|---|---|---|
| Tarea | 100 imágenes RGB + 100 máscaras binarias {0,255} | **Segmentación semántica binaria a nivel de píxel** (manzana / fondo) |
| Tamaño | 720×1280 (vertical), todas iguales | Reescalar ×0.5 para entrenar y ajustar (más rápido); evaluar a resolución original |
| Desbalanceo | Píxeles de manzana: **0,2 % – 10 %** (media ≈ 4,3 %) | La *accuracy* engaña (~96 % prediciendo todo fondo). Métricas principales: F1/Dice, IoU, MCC y PR-AUC |
| Variedades | Manzanas **amarillo-verdosas** (imágenes 0–~50) y **rojas** (resto) | El fondo verde (hojas, césped) se confunde con la manzana verde, por lo que el **color es clave**. Hay que estratificar el split por variedad |
| Origen probable | Parece un subconjunto de **MinneApple** (Häni et al., 2020) | Hay métodos clásicos de referencia publicados sobre este mismo tipo de imágenes |

**Restricción del enunciado:** solo técnicas tradicionales vistas en la asignatura (temas 2–7). No se usa sklearn: todo con OpenCV (`cv2`, `cv2.ml`) y NumPy, más Matplotlib/PIL. Optuna lo sugiere el profesor para el ajuste de hiperparámetros.

---

## 1. Pipeline general

```
 imágenes ─► [0. Carga + split] ─► [1. PREPROCESADO] ─► [2. MODELO] ─► [3. POSTPROCESADO] ─► [4. MÉTRICAS]
   +máscaras   train/val/test      filtros, color,       6-7 modelos     morfología, CCL,      píxel, curvas,
               (60/20/20,          contraste, mapa       (1 .py c/u)     watershed            objeto, tiempo
               estratificado)      "manzana-gris"             ▲
                                                            Optuna (TPE + pruning) sobre VALIDACIÓN
```

Regla metodológica: **Optuna ajusta siempre sobre validación y el test se evalúa una sola vez** al final de cada experimento. Si no, las métricas estarían infladas.

---

## 2. Fase 0: Carga, split y análisis exploratorio (EDA)

1. `loader.py`: leer imágenes (BGR→RGB) y máscaras (binarizar `>127`), con caché en `.npy`.
2. Split por imagen, estratificado por variedad (roja/amarilla): 60 train / 20 val / 20 test, con semilla fija.
3. **EDA para justificar el preprocesado** (va en la memoria):
   - Histogramas manzana-vs-fondo de cada canal en RGB, HSV, Lab, YCrCb y gris (tema 2, tema 6).
   - Separabilidad de cada canal con la **varianza interclase / intraclase** entre manzana y fondo (el mismo criterio que usa Otsu, tema 5) y el AUC de cada canal usado como clasificador (métricas del tema 5). Así se elige el canal de forma objetiva.
   - Mostrar que el gris estándar (Y = 0,299R + 0,587G + 0,114B) **apenas separa** la manzana del follaje. Esto justifica crear un "gris de manzana" (ver 3.3).

---

## 3. Fase 1: Preprocesado

Cada técnica es una función pura `img → img`. Una **configuración de preprocesado** es una cadena de funciones con nombre, lo que permite generar tablas por configuración.

### 3.1 Reducción de ruido (tema 3)
| Filtro | Para qué | Hiperparámetros (Optuna) |
|---|---|---|
| Media (`cv2.blur`) | Referencia de suavizado | k ∈ {3,5,7} |
| Gaussiano (`cv2.GaussianBlur`) | Ruido gaussiano del sensor | k, σ ∈ [0.3, 3] |
| Mediana (`cv2.medianBlur`) | Ruido **sal y pimienta** y brillos especulares en la piel | k ∈ {3,5,7} |
| Bilateral (`cv2.bilateralFilter`) | Suaviza conservando bordes (Tomasi & Manduchi, 1998) | d, σColor, σSpace |

*(Kalman descartado: no está en el temario y está pensado para secuencias temporales, no para imágenes estáticas.)*

**Experimento de robustez:** añadir artificialmente ruido sal y pimienta y ruido gaussiano a las imágenes de test y medir cuánto recupera cada filtro. Eso demuestra para qué sirve cada uno, en vez de solo afirmarlo.

### 3.2 Contraste e iluminación (temas 3 y 6)
- Corrección gamma / transformada logarítmica, para escenas a contraluz.
- Ecualización del histograma (global).
- **CLAHE** sobre el canal L de Lab o el V de HSV (Zuiderveld, 1994). Es útil con iluminación desigual entre sol y sombra, y no altera el color.

### 3.3 Espacio de color y "gris de manzana" (temas 2, 3 y 6)
La nota del profesor ("al pasar a gris el rojo pasa a amarillo y queda más visible") la interpreto así: **no usar el gris estándar, sino convertir el color a un único canal donde la manzana quede clara.** Todas las propuestas salen del temario:
| Mapa | Definición | Tema | Apoyo en la literatura |
|---|---|---|---|
| Canal a\* (Lab) | Verde frente a rojo. Destaca las manzanas rojas | T2 (CIE Lab: "a: cuán verde o rojo es el píxel") | Uso estándar en fruta roja |
| Canal b\* (Lab) | Azul frente a amarillo. Destaca las manzanas amarillas | T2 (CIE Lab: "b: cuán azul o amarillo") | — |
| Canal H y S (HSV) | Matiz y saturación, separados del brillo | T2 (HSV: "segmentación basada en color", "más robusto ante cambios de iluminación") | Wang et al. (2013) |
| V (YUV) / Cr (YCrCb) | V = 0,877·(R − Y): diferencia de color rojo-luminancia | T2 (fórmulas de YUV y YCrCb) | Bulanon et al. (2002) usan R−Y para manzanas Fuji |
| **Retroproyección del histograma** | Histograma 2D (p. ej. a\*–b\* o H–S) de los píxeles de manzana de train, normalizado. Cada píxel recibe P(manzana \| color) y se obtiene una imagen en gris de probabilidad | T6 (normalización del histograma: "interpretar probabilidades, segmentar objetos") | — |
| Combinación ponderada | Gris = α·a\* + β·b\* + γ·Cr (normalizado), con pesos α, β, γ como hiperparámetros de Optuna | T3 (transformaciones de intensidad) + ajuste con Optuna | Propuesta propia, justificada con el EDA |

### 3.4 Configuraciones de preprocesado a comparar (tablas)
| ID | Cadena |
|---|---|
| **P0** | Sin preprocesado (RGB crudo, o gris estándar en los modelos de un canal) |
| P1 | Gaussiano |
| P2 | Mediana |
| P3 | Bilateral |
| P4 | Bilateral + CLAHE(L) |
| P5 | Bilateral + CLAHE + espacio de color (Lab/HSV) |
| **P6 (completo)** | P5 + "gris de manzana" (retroproyección del histograma o combinación ponderada) como canal adicional |

### 3.5 Postprocesado (temas 3 y 5): aplicado tras cualquier modelo
- Apertura y cierre morfológico (elimina puntos sueltos y rellena huecos por brillos).
- **Etiquetado de componentes conexas (CCL)**: filtrado por área mínima y circularidad 4πA/P², porque las manzanas son aproximadamente redondas.
- *(Para métricas a nivel de objeto)* **Watershed** con transformada de distancia, para separar manzanas que se tocan.

---

## 4. Fase 2: Modelos (un `.py` por modelo, con la misma interfaz)

```python
class ModeloBase:
    params_por_defecto: dict
    def espacio_busqueda(self, trial) -> dict  # rangos para Optuna
    def fit(self, imgs, masks)              # no hace nada en los no supervisados
    def score_map(self, img) -> float[H,W]  # "probabilidad" de manzana; hace falta para ROC/AUC
    def predict(self, img) -> bool[H,W]     # score_map > umbral, más postprocesado
```
El `score_map` continuo es necesario para calcular **ROC-AUC y PR-AUC**. Todos los modelos lo pueden dar (indicado en la tabla).

| # | Fichero | Técnica (tema) | Entrada | `score_map` | Hiperparámetros clave | Justificación en la literatura |
|---|---|---|---|---|---|---|
| M1 | `m1_umbralizacion.py` | Umbral **global iterativo / Otsu / multi-Otsu / adaptativo / Sauvola** (T5) | Canal de un mapa de color (3.3) | Valor del canal normalizado | método, canal, offset de T, bloque y k (Sauvola) | Bulanon et al. 2002 y Zhou et al. 2012: umbrales en diferencias de color para manzana en campo. Otsu (1979), Sauvola (2000) |
| M2 | `m2_kmeans.py` | **K-means** (`cv2.kmeans`) sobre color (+xy opcional) (T5) | Vector de color por píxel | Distancia al centroide "manzana" (softmax) | K, espacio de color, peso espacial, inicialización | Uso habitual de la agrupación de color en fruta; método base del temario |
| M3 | `m3_gmm_em.py` | **GMM con EM** (`cv2.ml.EM`). Dos mezclas (manzana y fondo) y clasificación por razón de verosimilitudes (T5) | Color (+gris de manzana) | Posterior P(manzana \| x) | nº de componentes por clase, tipo de covarianza, umbral | **Roy et al. 2019 y Häni et al. 2020 (MinneApple / JFR) usan segmentación por GMM de color como método clásico de referencia en manzanos** |
| M4 | `m4_meanshift.py` | **Mean Shift** (`cv2.pyrMeanShiftFiltering`) + clasificación de regiones por color medio (T5) | Imagen Lab | Similitud del color de la región con el prototipo de manzana | sp (radio espacial), sr (radio de color), nivel | Comaniciu & Meer (2002). Respeta la proximidad espacial, a diferencia de K-means |
| M5 | `m5_knn.py` | **KNN** (`cv2.ml.KNearest`) sobre color + textura LBP (T7) | Vector por píxel o parche | Fracción de vecinos manzana | K, ponderación, tipo de característica, tamaño de muestra | Linker et al. 2012: color + textura para manzanas **verdes**, donde el color solo no basta |
| M6 | `m6_svm.py` | **SVM** RBF/lineal (`cv2.ml.SVM`) sobre color + LBP + gradiente Sobel (T7) | Vector por píxel o parche | Distancia al hiperplano (RAW_OUTPUT) | kernel, C, γ, pesos de clase | Ji et al. 2012: SVM con color y forma en un robot cosechador de manzanas |
| M7 | `m7_random_forest.py` | **Random Forest** (`cv2.ml.RTrees`) con las mismas características (T7) | Vector por píxel o parche | Fracción de votos | nº de árboles, profundidad máxima, mínimo de muestras, nº de variables por nodo | Breiman (2001). Robusto con características heterogéneas y da la importancia de cada característica |
| *(M8 opc.)* | `m8_hough_watershed.py` | **Transformada de Hough de círculos** + Watershed (T4, T5): detección y conteo de instancias | Bordes Canny del mapa de color | Votos del acumulador | dp, minDist, param1/2, radios | Gongal et al. 2015 (revisión): Hough circular, clásico en fruta redonda |

**Notas de implementación:**
- **Vector de características por píxel** (`features.py`): color en 2–3 espacios, gris de manzana, media y desviación local (ventana w), magnitud del gradiente Sobel e histograma LBP local (T4, T6). Opciones también para Optuna.
- **Muestreo de entrenamiento** para KNN, SVM y RF: unos 20–50 mil píxeles **balanceados** (50/50, o con pesos de clase), porque entrenar con los ~55 M píxeles no es viable y el desbalanceo sesgaría el modelo.
- Los modelos supervisados (M3, M5, M6, M7) usan las máscaras de train. Los no supervisados (M1, M2, M4) solo las usan para elegir el umbral o el clúster de manzana.

---

## 5. Fase 3: Métricas (tema 5 + extras), implementadas en NumPy

### 5.1 A nivel de píxel (por imagen y luego media ± desviación; también micro-promedio global)
- Matriz de confusión: TP, FP, TN, FN.
- **Accuracy** (para mostrar que engaña), **Precision**, **Recall/TPR**, **Specificity**, **FPR**.
- **F1 = Dice**, **IoU/Jaccard**.
- Extras recomendados por el desbalanceo: **MCC** (Chicco & Jurman, 2020), **Balanced Accuracy**, **κ de Cohen**.

### 5.2 Curvas a partir del `score_map`
- **ROC** y **ROC-AUC** (trapecios, implementación propia).
- **Precision-Recall** y **PR-AUC / Average Precision**. Con un 4 % de positivos, el ROC-AUC sale optimista; el PR-AUC es más informativo (Saito & Rehmsmeier, 2015). Esto da pie a discusión en la memoria.
- Umbral óptimo elegido en validación (máximo F1 o Youden J = TPR − FPR).

### 5.3 Contorno y objeto (interesantes para un robot cosechador)
- **Distancia de Hausdorff** (y HD95) entre contornos (tema 5).
- **Por objeto**: componentes conexas de la predicción y del GT, emparejadas con IoU ≥ 0,5 → precisión, recall y F1 de detección de manzanas.
- **Error de conteo** (MAE en nº de manzanas por imagen), relevante para estimar cosecha.
- **Tiempo de inferencia por imagen** (para valorar si sirve en tiempo real en el robot).
- Desglose de métricas por **variedad roja / amarilla** (mostrará dónde falla cada modelo).

### 5.4 Métricas internas de agrupación (para M2, M4 y la K de K-means)
- Coeficiente de silueta (sobre una submuestra), índice Davies-Bouldin y WCSS para elegir K (método del codo). Todos están en el tema 5.

---

## 6. Diseño experimental

### 6.1 Ablación del preprocesado (hiperparámetros por defecto)
Tabla **modelo × configuración P0…P6** con F1, IoU, MCC y PR-AUC. Muestra qué aporta cada paso.

### 6.2 Los 4 experimentos del profesor (por modelo)
| Exp. | Preprocesado | Hiperparámetros |
|---|---|---|
| **E1** | No (P0) | Por defecto |
| **E2** | Sí (mejor P de 6.1) | Por defecto |
| **E3** | No (P0) | Ajustados con Optuna |
| **E4** | Sí | Ajustados con Optuna (**incluyendo los parámetros del preprocesado** en el espacio de búsqueda) |

Hipótesis esperada (la del profesor): E1 < E2 ≈ E3 < E4. La tabla final es de 6–7 modelos × 4 experimentos, más gráfico de barras y curvas ROC/PR superpuestas.

### 6.3 Optuna (Akiba et al., KDD 2019)
- **Sampler TPE** (Bergstra et al., 2011): búsqueda bayesiana que muestrea más en las zonas de rangos que dieron buenos resultados. Es lo opuesto a grid search.
- **Pruner** (`MedianPruner` o `SuccessiveHalving`): el trial se evalúa imagen a imagen sobre validación y reporta el F1 acumulado con `trial.report()`. Si va peor que la mediana, se **corta** antes de terminar (es lo de "mientras entrena corta").
- **Presupuesto**: `n_trials` (p. ej. 50–100 por modelo) o `timeout` (horas).
- **Almacenamiento** `sqlite:///results/optuna.db` + **`optuna-dashboard`**: es la página web para ver en directo cómo mejora la métrica, las coordenadas paralelas y las importancias.
- **Importancia de hiperparámetros** (`optuna.importance`, fANOVA): indica qué hiperparámetros pesan más (lo de "asigna pesos").
- Objetivo a maximizar: **F1 medio en validación** (o MCC). Opcional: multiobjetivo F1 frente a tiempo.

### 6.4 Rigor
- Semillas fijas y repetir los modelos estocásticos (K-means, RF, EM) con 3–5 semillas para dar media ± desviación.
- *(Opcional)* Test de Wilcoxon por imagen entre E1 y E4, implementado a mano en NumPy (aproximación normal con `math.erf`), para afirmar que la mejora es significativa.

---

## 7. Estructura del proyecto

```
proyecto/
├── dataset_manzanas/            # (dado) imagenes/ + mascaras/
├── config.py                    # rutas, semilla, split, escala, hiperparámetros por defecto
├── main.py                      # CLI: python main.py --exp E4 --modelo svm --prepro P6
├── requirements.txt             # opencv-python, numpy, matplotlib, pillow, optuna, optuna-dashboard
├── README.md                    # cómo reproducir cada tabla/figura
├── src/
│   ├── datos/
│   │   └── loader.py            # carga, caché, split estratificado, muestreo balanceado
│   ├── preprocesado/
│   │   ├── filtros.py           # media, gaussiano, mediana, bilateral, ruido sintético
│   │   ├── contraste.py         # gamma, log, ecualización, CLAHE
│   │   ├── color.py             # espacios de color, canales a*, b*, H, S, Cr, V(YUV), retroproyección de histograma
│   │   ├── postprocesado.py     # morfología, CCL + filtro área/circularidad, watershed
│   │   └── pipelines.py         # definiciones P0…P6
│   ├── caracteristicas/
│   │   └── features.py          # vector por píxel: color, estadísticos locales, Sobel, LBP
│   ├── modelos/
│   │   ├── base.py              # interfaz común ModeloBase
│   │   ├── m1_umbralizacion.py
│   │   ├── m2_kmeans.py
│   │   ├── m3_gmm_em.py
│   │   ├── m4_meanshift.py
│   │   ├── m5_knn.py
│   │   ├── m6_svm.py
│   │   ├── m7_random_forest.py
│   │   └── m8_hough_watershed.py   # opcional
│   ├── metricas/
│   │   ├── pixel.py             # confusión, P, R, F1/Dice, IoU, Spec, MCC, κ, BalAcc
│   │   ├── curvas.py            # ROC, PR, AUC, AP, umbral óptimo
│   │   ├── objeto.py            # emparejamiento por IoU, conteo, Hausdorff
│   │   └── clustering.py        # silueta, Davies-Bouldin, WCSS
│   ├── ajuste/
│   │   └── optuna_search.py     # objetivo, TPE, pruner, storage, importancias
│   └── visualizacion/
│       └── graficas.py          # overlays, curvas, barras, tablas a LaTeX/Markdown
├── experimentos/
│   ├── eda.py                   # histogramas por canal + separabilidad (varianza inter/intraclase)
│   ├── ablacion_preprocesado.py # §6.1
│   ├── experimentos_E1_E4.py    # §6.2
│   └── robustez_ruido.py        # §3.1
├── resultados/
│   ├── tablas/   (csv + .tex)
│   ├── figuras/
│   └── optuna.db
├── memoria/                     # LaTeX: main.tex, secciones/, figuras/, referencias.bib
│                                #   (incluye la declaración de uso de IA con el formato del enunciado)
└── presentacion/                # 10 min
```

---

## 8. Plan temporal orientativo (hoy: 17-nov)

| Fechas | Tarea |
|---|---|
| 17–20 nov | Entorno, loader, EDA y métricas (las métricas primero, para poder medir todo lo demás) |
| 21–26 nov | Preprocesado + M1–M4 + ablación §6.1 |
| 27 nov–1 dic | M5–M7 (features) + Optuna + E1–E4 |
| 2–6 dic | Memoria (resultados y discusión), figuras |
| 7–9 dic | Revisión y **entrega** |
| 10–14 dic | Presentación + ensayo → **defensa el 15-dic** |

---

## 9. Referencias de apoyo (verificar el DOI antes de citarlas en la memoria)
- Häni, N., Roy, P., Isler, V. (2020). *MinneApple: A benchmark dataset for apple detection and segmentation*. IEEE Robotics and Automation Letters 5(2).
- Häni, N., Roy, P., Isler, V. (2020). *A comparative study of fruit detection and counting methods for yield mapping in apple orchards*. Journal of Field Robotics 37(2).
- Roy, P., Kislay, A., Plonski, P., Luby, J., Isler, V. (2019). *Vision-based preharvest yield mapping for apple orchards*. Computers and Electronics in Agriculture 164.
- Gongal, A., Amatya, S., Karkee, M., Zhang, Q., Lewis, K. (2015). *Sensors and systems for fruit detection and localization: A review*. Computers and Electronics in Agriculture 116.
- Bulanon, D.M., Kataoka, T., Ota, Y., Hiroma, T. (2002). *A segmentation algorithm for the automatic recognition of Fuji apples at harvest*. Biosystems Engineering 83(4).
- Zhou, R., Damerow, L., Sun, Y., Blanke, M. (2012). *Using colour features of cv. 'Gala' apple fruits in an orchard in image processing to predict yield*. Precision Agriculture 13.
- Linker, R., Cohen, O., Naor, A. (2012). *Determination of the number of green apples in RGB images recorded in orchards*. Computers and Electronics in Agriculture 81.
- Ji, W., Zhao, D., Cheng, F., et al. (2012). *Automatic recognition vision system guided for apple harvesting robot*. Computers & Electrical Engineering 38(5).
- Otsu (1979) IEEE TSMC · Sauvola & Pietikäinen (2000) Pattern Recognition · Tomasi & Manduchi (1998) ICCV · Zuiderveld (1994) *CLAHE*, Graphics Gems IV · Ojala et al. (2002) *LBP*, TPAMI · Comaniciu & Meer (2002) *Mean Shift*, TPAMI · Cortes & Vapnik (1995) *SVM* · Breiman (2001) *Random Forests*.
- Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter optimization framework*. KDD · Bergstra, J. et al. (2011). *Algorithms for hyper-parameter optimization* (TPE). NeurIPS.
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple data sets*. JMLR 7 (para el test de Wilcoxon opcional).
- Saito, T., Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot…* PLoS ONE · Chicco, D., Jurman, G. (2020). *The advantages of the MCC over F1 and accuracy…* BMC Genomics.

---

**Pendiente:**
1. Presupuesto de cómputo para Optuna (n_trials / horas por modelo).
2. Número final de modelos (7, o 8 con Hough + Watershed).
3. Límite de páginas de la memoria.
4. Instalación del entorno: `opencv-python`, `optuna`, `optuna-dashboard`.
