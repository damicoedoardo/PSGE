#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import numpy as np
import scipy.sparse as sps
from irec.experiments.svd_dataset import load_svd
from irec.model.recommender_interfaces import ItemSimilarityRecommender
from irec.utils.utils import interactions_to_sparse_matrix, set_color, truncate_top_k
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
from sparsesvd import sparsesvd


class PSGE(ItemSimilarityRecommender):
    name = "PSGE"

    def __init__(
        self,
        dataset,
        train_data,
        k: int,
        alpha: float,
        beta: float,
        precomputed_svd: bool = False,
    ):
        super().__init__(dataset=dataset, train_data=train_data)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.precomputed_svd = precomputed_svd

    def compute_similarity_matrix(self):
        sp_int, _, _ = interactions_to_sparse_matrix(
            self.train_data,
            items_num=self.dataset.items_num,
            users_num=self.dataset.users_num,
        )

        # computing user mat
        user_degree = np.array(sp_int.sum(axis=1))
        d_user_inv = np.power(user_degree, -self.beta).flatten()
        d_user_inv[np.isinf(d_user_inv)] = 0.0
        d_user_inv_diag = sps.diags(d_user_inv)

        item_degree = np.array(sp_int.sum(axis=0))
        d_item_inv = np.power(item_degree, -self.alpha).flatten()
        d_item_inv[np.isinf(d_item_inv)] = 0.0
        d_item_inv_diag = sps.diags(d_item_inv)

        d_item_alpha = np.power(item_degree, -self.alpha).flatten()
        d_item_alpha[np.isinf(d_item_alpha)] = 0.0
        d_item_alpha = sps.diags(d_item_alpha)

        d_item_inv_alpha = np.power(item_degree, self.alpha).flatten()
        d_item_inv_alpha[np.isinf(d_item_inv_alpha)] = 0.0
        d_item_inv_alpha_diag = sps.diags(d_item_inv_alpha)

        if self.precomputed_svd:
            print(set_color("Loading svd...", "yellow"))
            _, vt, s = load_svd(self.dataset, merge_val=True, alpha=0.4, beta=0.3)
            vt = vt[: self.k, :]

        else:
            int_norm = d_user_inv_diag.dot(sp_int).dot(d_item_inv_diag)
            # int_norm = sp_int.dot(d_item_inv_diag)
            # int_norm = d_user_inv_diag.dot(sp_int)
            _, _, vt = sparsesvd(int_norm.tocsc(), self.k)

        # self.similarity_matrix = vt.T @ vt
        self.similarity_matrix = (d_item_alpha @ vt.T) @ (
            d_item_inv_alpha_diag @ vt.T
        ).T

    def state_dict(self):
        sd = {"k": self.k, "alpha": self.alpha, "beta": self.beta}
        return sd
