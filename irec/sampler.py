#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.utils.utils import *
import numpy as np
from irec.constants import *
from irec.utils import pandas_utils
import tensorflow as tf
from irec.data.dataset import Dataset
import logging

logger = logging.getLogger(__name__)


class TripletsBPRGenerator:
    def __init__(
        self,
        dataset,
        train_data,
        batch_size,
        items_after_users_ids=False,
        seed=SEED,
    ):
        """
        Creates batches of triplets of (user, positive_item, negative_item)

        Create batches of triplets required to train an algorithm with BPR loss function

        Args:
            dataset (Dataset): Dataset object
            train_data (pd.DataFrame): DataFrame containing user-item interactions
            batch_size (int): size of the batch to be generated
            items_after_users_ids (bool): whether or not make the ids of items start after the last user id
            seed (int): random seed used to generate samples
        """
        assert pandas_utils.has_columns(
            df=train_data, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )

        assert (
            dataset.dataset is not None
        ), "Dataset passed has not been loaded ! Call load() method"
        self.dataset = dataset
        self.train_data = train_data
        self.batch_size = batch_size
        self.items_after_users_ids = items_after_users_ids
        self.seed = SEED

        self.num_samples = train_data.shape[0]
        self.num_batches = self.num_samples // batch_size

        # set random seed
        np.random.seed(seed)

        # drop unnecessary columns
        train_data = train_data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]

        if items_after_users_ids:
            logger.warning(
                set_color(
                    "Changing item ids to make them start from last_user + 1", "white"
                )
            )
            train_data[DEFAULT_ITEM_COL] = (
                train_data[DEFAULT_ITEM_COL] + dataset.users_num
            )

        # save user and item ids
        self.user_ids = list(sorted(train_data[DEFAULT_USER_COL].unique()))
        self.item_ids = list(sorted(train_data[DEFAULT_ITEM_COL].unique()))

        # unroll train_df into a list
        interaction_list = np.array(list(train_data.itertuples(index=False, name=None)))
        self.interactions_list = interaction_list

        # create user-item dict {user_idx: { item_idx: rating, ..., }, ...}
        # useful for negative sampling
        train_data_grouped = (
            train_data.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
            .apply(list)
            .reset_index()
        )
        self.user_item_dict = dict(
            map(
                lambda x: (x[0], dict(zip(x[1], np.ones(len(x[1]))))),
                train_data_grouped.values,
            )
        )

    def _negative_sample(self, u):
        def _get_random_key(list):
            L = len(list)
            i = np.random.randint(0, L)
            return list[i]

        # sample negative sample
        j = _get_random_key(self.item_ids)
        while j in self.user_item_dict[u]:
            j = _get_random_key(self.item_ids)
        return j

    def sample(self):
        """Create batch of triplets to optimize BPR loss

        The data are provided as following: [[user_0, ... ,], [pos_item_0, ... ,] [neg_item_0], ... ,]]

        Returns:
            np.array: batch of triplets
        """
        u_list = []
        i_list = []
        j_list = []

        pos_sample_idx = np.random.random_integers(
            low=0, high=round(self.num_samples - 1), size=self.batch_size
        )
        pos_sample = self.interactions_list[pos_sample_idx]
        u, i = list(zip(*pos_sample))
        u_list.extend(u)
        i_list.extend(i)
        for u in u_list:
            j_list.append(self._negative_sample(u))
        return tf.constant(np.array([u_list, i_list, j_list]))
