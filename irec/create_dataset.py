#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

from constants import *
from irec.data.implemented_datasets import AmazonElectronics, Gowalla, Movielens1M

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

DATASETS = [AmazonElectronics(), Movielens1M(), Gowalla()]
if __name__ == "__main__":
    for dataset in DATASETS:
        dataset.download()
        dataset.preprocess_dataset(k_core=10)
        dataset.create_stratified_split([0.8, 0.1, 0.1], k_core=10)
