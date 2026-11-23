# Segmentación clásica de manzanas

Proyecto de Visión por Computador I para segmentar manzanas en imágenes de campo
con técnicas tradicionales de procesamiento de imagen y aprendizaje clásico.
El diseño y las fases del trabajo están descritos en
[ESQUEMA_PROYECTO.md](./ESQUEMA_PROYECTO.md).

## Primer incremento: datos y métricas de píxel

El loader de `src/datos/loader.py` empareja imágenes y máscaras por nombre,
convierte las imágenes BGR de OpenCV a RGB, binariza las máscaras con el umbral
`>127`, valida las dimensiones y guarda una caché local `.npy` en `.cache/`.
La partición es determinista (60/20/20) y estratificada según el índice de las
imágenes: `img_000`–`img_049` se consideran amarillas y las restantes rojas.
Este límite sigue la descripción del dataset del esquema y está centralizado
en `config.py`.

Las métricas de `src/metricas/pixel.py` incluyen F1/Dice, IoU, MCC, balanced
accuracy y Cohen's kappa, además de las tasas de confusión. Se calculan con
NumPy y no dependen de scikit-learn.

## Entorno, EDA y pruebas

Se necesita Python 3.10 o posterior. En Windows, crea y activa el entorno virtual
con:

```powershell
py -3.12 -m venv .venv-project
.\.venv-project\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Ejecuta el análisis exploratorio:

```powershell
python -m experimentos.eda
```

El EDA usa únicamente el split de entrenamiento, reduce las imágenes a la mitad
de su tamaño, y toma hasta 5.000 píxeles de cada clase y cada imagen para que
las imágenes con más área de manzana no dominen las distribuciones. Genera
`resultados/eda/separabilidad_canales.csv` con medias, desviaciones,
separabilidad inter/intraclase y AUC por canal, y
`resultados/eda/histogramas_canales.png` con los histogramas normalizados de
manzana y fondo en RGB, HSV, Lab, YCrCb y gris. El AUC se calcula como si una
intensidad mayor indicara manzana; también se informa la mejor orientación
(intensidad mayor o menor) para medir la separabilidad sin presuponer el sentido.
La razón inter/intraclase pondera con la proporción de las muestras usadas para
el histograma; el CSV también conserva la prevalencia real de manzana en train.

Ejecuta las pruebas con:

```powershell
python -m unittest discover -s tests
```

El dataset debe estar en `dataset_manzanas/`, con las subcarpetas `imagenes/`
y `mascaras/`. En el EDA actual, los canales con mayor AUC en train son Cr de
YCrCb (0,877), Cb de YCrCb (0,876) y b* de Lab (0,869). Esto sugiere que las
diferencias cromáticas son más útiles que el gris estándar, pero no selecciona
por sí solo un pipeline ganador: la próxima fase comparará preprocesados y
umbrales usando validación, tal como indica el esquema.

Para cargar y consultar la partición desde Python:

```python
from src.datos.loader import load_dataset, stratified_split

samples = load_dataset()
split = stratified_split(samples)
```
