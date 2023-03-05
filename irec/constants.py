#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from distutils.log import ERROR

DATASETS_PATH = "rsys-datasets"
MODELS_SAVE_PATH = "irec-saved_models"
# Default column names
DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
COL_DICT = {
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_prediction": DEFAULT_PREDICTION_COL,
}

# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10
SEED = 42

LOGGING_LEVEL = ERROR
