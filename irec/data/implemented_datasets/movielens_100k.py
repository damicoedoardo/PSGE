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


class Movielens100k(
    Dataset,
    name="Movielens100k",
    directory_name="ml-100k",
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    url_archive_format="zip",
    expected_files=[
        "u.data",
        "u.user",
        "u.item",
        "u.genre",
        "u.occupation",
    ],
    description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
    source="https://grouplens.org/datasets/movielens/100k/",
):
    pass

    def _preprocess_dataset(self, k_core: int = 10) -> pd.DataFrame:
        assert (
            self._is_downloaded()
        ), "Dataset is not downloaded call self.download() method first"

        ratings, *_ = [self._resolve_path(path) for path in self.expected_files]

        interactions = pd.read_csv(
            ratings,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            usecols=["user_id", "movie_id", "rating"],
        )

        users_num = len(interactions["user_id"].unique())
        item_num = len(interactions["movie_id"].unique())

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
        # dropping duplicates that can be produced from timestamp
        interactions = interactions.drop_duplicates()

        # drop ratings column
        interactions = interactions.drop("rating", axis=1)

        interactions = interactions.rename(
            columns={
                "user_id": DEFAULT_USER_COL,
                "movie_id": DEFAULT_ITEM_COL,
            }
        )

        logger.info(
            set_color(
                f"Dataset Statistics after implicitization:\n"
                f"interactions count: {len(interactions)}\n"
                f"user count {len(interactions[DEFAULT_USER_COL].unique())}\n"
                f"item count {len(interactions[DEFAULT_ITEM_COL].unique())}\n",
                color="cyan",
            )
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
