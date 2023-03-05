#!/usr/bin/env python

import numpy as np
import scipy.sparse as sps
from irec.constants import *
from irec.model.recommender_interfaces import ItemSimilarityRecommender
from irec.utils.utils import interactions_to_sparse_matrix, timing


class EASE(ItemSimilarityRecommender):
    """EASE

    Note:
        paper: https://dl.acm.org/doi/abs/10.1145/3308558.3313710?casa_token=BtGI7FceWgYAAAAA:rz8xxtv4mlXjYIo6aWWlsAm9CP7zh-JZGGmN5UYUA4XwefaRfD6ZJ015GFkiMoBACF6GgKP9HEbMwQ

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
        l2 (float): l2 regularization
    """

    name = "EASE"

    def __init__(self, dataset, train_data, l2):
        """EASE

        Note:
            paper: https://dl.acm.org/doi/abs/10.1145/3308558.3313710?casa_token=BtGI7FceWgYAAAAA:rz8xxtv4mlXjYIo6aWWlsAm9CP7zh-JZGGmN5UYUA4XwefaRfD6ZJ015GFkiMoBACF6GgKP9HEbMwQ

        Attributes:
            train_data (pd.DataFrame): dataframe containing user-item interactions
            l2 (float): l2 regularization
        """
        super().__init__(train_data=train_data, dataset=dataset)
        self.l2 = l2

    @timing
    def compute_similarity_matrix(self):
        sp_int, _, _ = interactions_to_sparse_matrix(
            self.train_data,
            items_num=self.dataset.items_num,
            users_num=self.dataset.users_num,
        )
        # Compute gram matrix
        G = (sp_int.T * sp_int).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.l2 * self.dataset.items_num
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0
        self.similarity_matrix = B

    def state_dict(self):
        return {"l2": self.l2}
