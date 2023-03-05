#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.experiments.svd_dataset import load_svd
from irec.model.recommender_interfaces import ItemSimilarityRecommender
from irec.utils.utils import interactions_to_sparse_matrix, truncate_top_k
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sps
from sparsesvd import sparsesvd


class SGMC(ItemSimilarityRecommender):
    name = "SGMC"

    def __init__(self, dataset, train_data, k: int = 256):
        super().__init__(dataset=dataset, train_data=train_data)
        self.k = k
        self.u = None
        self.v = None
        self.s = None

    def state_dict(self):
        sd = {"k": self.k}
        return sd

    def compute_similarity_matrix(self):
        sp_int, _, _ = interactions_to_sparse_matrix(
            self.train_data,
            items_num=self.dataset.items_num,
            users_num=self.dataset.users_num,
        )

        rowsum = np.array(sp_int.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sps.diags(d_inv)
        norm_adj = d_mat.dot(sp_int)

        colsum = np.array(sp_int.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sps.diags(d_inv)
        d_mat_i = d_mat
        d_mat_i_inv = sps.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        ut, s, vt = sparsesvd(norm_adj, self.k)
        D_U_U_T_D = d_mat_i @ vt.T @ vt @ d_mat_i_inv
        self.similarity_matrix = D_U_U_T_D
