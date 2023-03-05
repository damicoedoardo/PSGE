#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

from irec.losses import bpr_loss
from irec.model.recommender_interfaces import (
    TensorflowRecommender,
    RepresentationsBasedRecommender,
)
from irec.constants import *
import tensorflow as tf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BPR(TensorflowRecommender):
    """Matrix factorization BPR

    Note:
        paper: https://arxiv.org/abs/1205.2618

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        dataset (Dataset): Dataset used for training
        embedding_size (int): dimension of user-item embeddings
    """

    name = "BPR"

    def __init__(self, dataset, train_data, embedding_size=64):
        TensorflowRecommender.__init__(self, train_data=train_data, dataset=dataset)

        self.embedding_size = embedding_size

        # create embeddings
        initializer = tf.initializers.GlorotUniform(SEED)
        self.embeddings = tf.Variable(
            initializer(
                shape=[
                    self.dataset.users_num + self.dataset.items_num,
                    self.embedding_size,
                ]
            ),
            trainable=True,
        )

    def state_dict(self):
        return {"embedding_size": self.embedding_size}

    def __call__(self):
        """Return users and items embeddings

        Returns:
            tf.Variable: embeddings of users and items
        """
        return self.embeddings

    def compute_representations(self, interactions):
        user_id = interactions[DEFAULT_USER_COL].unique()
        embeddings = self()
        users_emb = tf.gather(embeddings, tf.constant(user_id)).numpy()
        items_emb = tf.gather(
            embeddings,
            tf.constant(
                np.arange(
                    self.dataset.users_num,
                    self.dataset.users_num + self.dataset.items_num,
                )
            ),
        ).numpy()
        users_repr_df = pd.DataFrame(users_emb, index=user_id)
        items_repr_df = pd.DataFrame(items_emb, index=np.arange(self.dataset.items_num))
        return users_repr_df, items_repr_df

    @tf.function
    def train_step(self, ids):
        x = self()
        u, i, j = ids[0], ids[1], ids[2]
        x_u = tf.gather(x, u)
        x_i = tf.gather(x, i)
        x_j = tf.gather(x, j)
        loss = bpr_loss(x_u, x_i, x_j)
        return loss
