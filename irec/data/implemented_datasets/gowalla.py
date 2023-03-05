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


class Gowalla(
    Dataset,
    name="Gowalla",
    directory_name="gowalla",
    url="http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
    url_archive_format=None,
    expected_files=["loc-gowalla_totalCheckins.txt.gz"],
    description="""Gowalla is a location-based social networking website where users
     share their locations by checking-in. The friendship network is undirected and was
     collected using their public API, and consists of 196,591 nodes and 950,327 edges.
     We have collected a total of 6,442,890 check-ins of these users over the period of Feb. 2009 - Oct. 2010.
                """,
    source="https://snap.stanford.edu/data/loc-Gowalla.html",
):
    pass

    def _preprocess_dataset(self, k_core: int = 10):
        assert (
            self._is_downloaded()
        ), "Dataset is not downloaded call self.download() method first"
        data_path = [self._resolve_path(path) for path in self.expected_files]
        df = pd.read_csv(
            data_path[0],
            sep="\t",
            names=["userID", "check-in time", "latitude", "longitude", "itemID"],
        )
        # remove unused columns and drop duplicates
        interactions = df.drop(
            ["check-in time", "latitude", "longitude"], axis=1
        ).drop_duplicates()
        interactions = interactions.rename(
            columns={"userID": DEFAULT_USER_COL, "itemID": DEFAULT_ITEM_COL}
        )

        users_num = len(interactions[DEFAULT_USER_COL].unique())
        item_num = len(interactions[DEFAULT_ITEM_COL].unique())

        logger.info(
            set_color(
                f"Dataset Statistics\n"
                f"interactions count: {len(interactions)}\n"
                f"user count {users_num}\n"
                f"item count {item_num}\n",
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
