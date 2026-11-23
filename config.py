from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset_manzanas"
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_manzanas"

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
# img_000–img_065: huerto de manzanas amarillo-verdosas; img_066–img_099: manzanas
# rojas. Verificado midiendo el color (a* de Lab) de los píxeles de manzana de cada
# máscara y revisando las imágenes a mano (66 amarillas / 34 rojas).
YELLOW_IMAGE_MAX_INDEX = 65
IMAGE_SCALE = 0.5
EDA_OUTPUT_DIR = PROJECT_ROOT / "resultados" / "eda"
