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


class LastFM(
    Dataset,
    name="LastFM",
    directory_name="hetrec2011-lastfm-2k",
    url="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
    url_archive_format="zip",
    expected_files=[
        "user_artists.dat",
        "tags.dat",
        "artists.dat",
        "user_taggedartists-timestamps.dat",
        "user_taggedartists.dat",
        "user_friends.dat",
    ],
    description="""92,800 artist listening records from 1892 users.
                   This dataset contains social networking, tagging, and music artist listening information 
                    from a set of 2K users from Last.fm online music system.
                    http://www.last.fm 

                    The dataset is released in the framework of the 2nd International Workshop on 
                    Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011) 
                    http://ir.ii.uam.es/hetrec2011 
                    at the 5th ACM Conference on Recommender Systems (RecSys 2011)
                    http://recsys.acm.org/2011 
                """,
    source="https://grouplens.org/datasets/hetrec-2011/",
):
    pass

    def _preprocess_dataset(self, k_core: int = 10) -> pd.DataFrame:
        assert (
            self._is_downloaded()
        ), "Dataset is not downloaded call self.download() method first"
        user_artists = self._resolve_path("user_artists.dat")
        with open(user_artists) as data:
            lines = data.readlines()
            # remove the first line which contains the columns name
            spl_lines = [line.replace("\n", "").split("\t") for line in lines]
            columns_name = spl_lines.pop(0)
        interactions = pd.DataFrame(spl_lines, columns=columns_name).astype(int)
        # remove unused columns and drop duplicates
        interactions = interactions.drop("weight", axis=1).drop_duplicates()
        interactions = interactions.rename(
            columns={"userID": DEFAULT_USER_COL, "artistID": DEFAULT_ITEM_COL}
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
