import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# RAW DATA PATHS
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset")
FOODS_CSV_PATH = os.path.join(RAW_DATA_DIR, "foods.csv")
RATINGS_CSV_PATH = os.path.join(RAW_DATA_DIR, "ratings.csv")

# PROCESSED DATA PATHS
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "processed_data")
FOODS_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "foods_processed.csv")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.csv")
VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "validation.csv")
FOOD_MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "food_mapping.csv")
USER_MAPPING_PATH = os.path.join(PROCESSED_DATA_DIR, "user_mapping.csv")

# MODEL PATHS
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
NCF_MODEL_DIR = os.path.join(MODEL_DIR, "ncf")
BERT_MODEL_DIR = os.path.join(MODEL_DIR, "bert")
HYBRID_BERT_MODEL_DIR = os.path.join(MODEL_DIR, "hybrid_bert")

# Ensure directories exist
os.makedirs(NCF_MODEL_DIR, exist_ok=True)
os.makedirs(BERT_MODEL_DIR, exist_ok=True)
os.makedirs(HYBRID_BERT_MODEL_DIR, exist_ok=True)

RATING_MIN = 0.0
RATING_MAX = 5.0
