#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

import pandas as pd

from irec.constants import *
from irec.data.data_preprocessing import kcore
from irec.data.dataset import Dataset
from irec.utils.pandas_utils import remap_column_consecutive
from irec.utils.utils import set_color

logger = logging.getLogger(__name__)


class AmazonElectronics(
    Dataset,
    name="Amazon Electronics",
    directory_name="Amazon_Electronics",
    url="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv",
    url_archive_format=None,
    expected_files=["ratings_Electronics.csv"],
    description="""
    This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014
                """,
    source="http://jmcauley.ucsd.edu/data/amazon/links.html",
):
    pass

    def _preprocess_dataset(self, k_core: int = 10):
        assert (
            self._is_downloaded()
        ), "Dataset is not downloaded call self.download() method first"
        data_path = [self._resolve_path(path) for path in self.expected_files]
        interactions = pd.read_csv(
            data_path[0], names=["userID", "itemID", "rating", "timestamp"]
        )

        users_num = len(interactions["userID"].unique())
        item_num = len(interactions["itemID"].unique())

        logger.info(
            set_color(
                f"Dataset Statistics\n"
                f"interactions count: {len(interactions)}\n"
                f"user count {users_num}\n"
                f"item count {item_num}\n",
                color="cyan",
            )
        )

        logger.warning(
            set_color(
                f"Dataset contain explicit feedback ranging from 1 to 5\n "
                f"Transform it into implicit\n"
                f"ratings >= 4 -> 1\n"
                f"ratings < 4 -> 0\n",
                color="white",
            )
        )

        interactions = interactions[interactions["rating"] >= 4]
        # drop timestamp
        interactions = interactions.drop("timestamp", axis=1).drop_duplicates()

        # drop ratings column
        interactions = interactions.drop("rating", axis=1)

        interactions = interactions.rename(
            columns={
                "UserID": DEFAULT_USER_COL,
                "ItemID": DEFAULT_ITEM_COL,
            }
        )

        interactions = kcore(interactions, k=k_core)
        logger.info(
            set_color(
                f"Dataset Statistics after k-core preprocessing: {k_core}\n"
                f"interactions count: {len(interactions)}\n"
                f"user count {len(interactions[DEFAULT_USER_COL].unique())}\n"
                f"item count {len(interactions[DEFAULT_ITEM_COL].unique())}\n",
                color="cyan",
            )
        )

        logger.info(set_color(f"Map {DEFAULT_USER_COL} to consecutive", "yellow"))
        interactions, _ = remap_column_consecutive(interactions, DEFAULT_USER_COL)
        logger.info(set_color(f"Map {DEFAULT_ITEM_COL} to consecutive", "yellow"))
        interactions, _ = remap_column_consecutive(interactions, DEFAULT_ITEM_COL)
        return interactions
