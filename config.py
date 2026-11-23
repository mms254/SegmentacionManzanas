from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset_manzanas"
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_manzanas"

RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
YELLOW_IMAGE_MAX_INDEX = 49
IMAGE_SCALE = 0.5
