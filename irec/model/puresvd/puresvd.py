#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from sparsesvd import sparsesvd

from irec.model.recommender_interfaces import ItemSimilarityRecommender
from irec.utils.utils import interactions_to_sparse_matrix, truncate_top_k
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.utils.extmath import randomized_svd
from irec.constants import *


class PureSVD(ItemSimilarityRecommender):
    """PureSVD"""

    name = "PureSVD"

    def __init__(self, dataset, train_data, n_components):
        super().__init__(dataset=dataset, train_data=train_data)

        self.n_components = n_components

    def compute_similarity_matrix(self):
        sp_int, _, _ = interactions_to_sparse_matrix(
            self.train_data,
            items_num=self.dataset.items_num,
            users_num=self.dataset.users_num,
        )

        sp_int = sp_int.tocsc()
        ut, s, vt = sparsesvd(sp_int, self.n_components)
        sim = vt.T @ vt
        self.similarity_matrix = sim

    def state_dict(self):
        sd = {"n_components": self.n_components}
        return sd
