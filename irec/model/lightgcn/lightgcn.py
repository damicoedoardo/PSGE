#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

from irec.losses import bpr_loss, signal_cross_correlation_reg
from irec.model.recommender_interfaces import (
    TensorflowRecommender,
    RepresentationsBasedRecommender,
)
from irec.constants import *
import tensorflow as tf
import pandas as pd
import numpy as np

from irec.utils.graph_utils import (
    adjacency_from_interactions,
    get_symmetric_normalized_laplacian,
    get_random_walk_matrix,
)
from irec.utils.tensorflow_utils import to_tf_sparse_tensor
import scipy.sparse as sps

logger = logging.getLogger(__name__)


class LightGCN(TensorflowRecommender):
    """LightGCN

    Note:
        paper: https://arxiv.org/abs/2002.02126

    Attributes:
        dataset (Dataset): Dataset used for training
        train_data (pd.DataFrame): dataframe containing user-item interactions
        embeddings_size (int): dimension of user-item embeddings
        k (int): Convolution depth
    """

    name = "LightGCN"

    def __init__(self, dataset, train_data, embedding_size, k):
        TensorflowRecommender.__init__(self, train_data=train_data, dataset=dataset)

        self.embedding_size = embedding_size
        self.k = k

        # create embeddings
        initializer = tf.initializers.GlorotUniform(SEED)
        self.embeddings = tf.Variable(
            initializer(
                shape=[self.dataset.users_num + self.dataset.items_num, embedding_size]
            ),
            trainable=True,
        )

        # Compute propagation matrix
        adj = adjacency_from_interactions(
            train_data, users_num=dataset.users_num, items_num=dataset.items_num
        )
        S = get_symmetric_normalized_laplacian(adj, self_loop=False)
        self.S = to_tf_sparse_tensor(S)

    def state_dict(self):
        sd = {"embedding_size": self.embedding_size, "k": self.k}
        return sd

    def __call__(self):
        """Return the embeddings associated to the ids inside inputs

        Args:
            inputs (list): Input tensor, or dict/list/tuple of input tensors
            training (bool): Boolean or boolean scalar tensor, indicating whether to run
                the `Network` in training mode or inference mode.
        Returns:
            tuple: tf.Tensor representing embeddings associated to the ids requested
        """
        x = self.embeddings
        depth_embeddings = [x]

        # propagation step
        for i in range(self.k):
            x = tf.sparse.sparse_dense_matmul(self.S, x)
            depth_embeddings.append(x)

        stackked_emb = tf.stack(depth_embeddings, axis=1)
        final_emb = tf.reduce_mean(stackked_emb, axis=1)
        return final_emb

    @tf.function
    def train_step(self, ids):
        x = self()
        u, i, j = ids[0], ids[1], ids[2]
        x_u = tf.gather(x, u)
        x_i = tf.gather(x, i)
        x_j = tf.gather(x, j)
        loss = bpr_loss(x_u, x_i, x_j)

        return loss

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
