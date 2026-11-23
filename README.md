# Segmentación clásica de manzanas

Proyecto de Visión por Computador I para segmentar manzanas en imágenes de campo
con técnicas tradicionales de procesamiento de imagen y aprendizaje clásico.
El diseño y las fases del trabajo están descritos en
[ESQUEMA_PROYECTO.md](./ESQUEMA_PROYECTO.md).

## Fase 0: datos, métricas y EDA

El loader de `src/datos/loader.py` empareja imágenes y máscaras por nombre,
convierte las imágenes BGR de OpenCV a RGB, binariza las máscaras con el umbral
`>127`, valida las dimensiones y guarda una caché local `.npy` en `.cache/`.
La partición es determinista (60/20/20) y estratificada por variedad:
`img_000`–`img_065` son amarillas (66) y `img_066`–`img_099` rojas (34),
comprobado midiendo el color de las máscaras. Los sobrantes del redondeo se
reparten por el método del mayor resto, así que el total queda en 60/20/20
(train 40+20, validación 13+7, test 13+7).

- `src/metricas/pixel.py`: F1/Dice, IoU, MCC, balanced accuracy, kappa de Cohen
  y tasas de confusión, en NumPy y sin scikit-learn.
- `src/metricas/curvas.py`: ROC, Precision-Recall, ROC-AUC y Average Precision
  exactas a partir de histogramas de puntuación (mapas uint8).
- `src/metricas/objeto.py`: componentes conexas con área, radio, circularidad y
  solidez normalizadas respecto a un disco digital (métricas por manzana).
- `src/preprocesado/color.py`: canales de color, distancia circular del matiz al
  rojo, candidatos a "gris de manzana" y retroproyección del histograma 2D.
  Las reutilizará el preprocesado (P5/P6).
- `src/preprocesado/ruido.py`: estimación de ruido (Immerkær, sal y pimienta) y
  simulación de ruido para el experimento de robustez.

## Entorno, EDA y pruebas

Se necesita Python 3.10 o posterior (NumPy ≥ 2.0). En Windows, crea y activa el
entorno virtual con:

```powershell
py -3.12 -m venv .venv-project
.\.venv-project\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name segmentacion-manzanas --display-name "Python (segmentacion-manzanas)"
```

El último comando registra el entorno como kernel de Jupyter dentro del propio
`.venv-project` (no toca la configuración global); el notebook lo usa por nombre.

## Análisis exploratorio (EDA)

Todo el EDA está en un único notebook, [`experimentos/eda.ipynb`](./experimentos/eda.ipynb),
que explica y justifica cada paso y termina con una tabla de decisiones para el
preprocesado, los modelos y las métricas. No usa el split de test. Para
abrirlo, selecciona el kernel *Python (segmentacion-manzanas)*; para volver a
ejecutarlo entero desde la terminal (≈20 s):

```powershell
.\.venv-project\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace experimentos/eda.ipynb
```

Secciones: verificación del dataset · variedades · split · desbalanceo ·
separabilidad de canales por variedad · candidatos a "gris de manzana" ·
tamaño y forma de las manzanas · iluminación, sobreexposición y ruido ·
anotaciones incompletas · conclusiones. Las figuras y tablas se guardan también
en `resultados/eda/` para la memoria.

Ejecuta las pruebas con:

```powershell
python -m unittest discover -s tests
```

El dataset debe estar en `dataset_manzanas/`, con las subcarpetas `imagenes/`
y `mascaras/`.

Para cargar y consultar la partición desde Python:

```python
from src.datos.loader import load_dataset, stratified_split

samples = load_dataset()
split = stratified_split(samples)
```
