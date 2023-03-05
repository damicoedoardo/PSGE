#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from irec.data.dataset import Dataset
from irec.data.implemented_datasets import *
from irec.utils.graph_utils import (
    adjacency_from_interactions,
    get_laplacian,
    get_propagation_matrix,
    get_symmetric_normalized_laplacian,
)
from irec.utils.utils import interactions_to_sparse_matrix, set_color, timing
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sparsesvd import sparsesvd

logger = logging.getLogger(__name__)


@timing
def compute_svd(
    data_df: pd.DataFrame,
    dataset: Dataset,
    k: int = None,
    alpha=0.5,
    beta=0.5,
    merge_val=False,
    val_df: pd.DataFrame = None,
):
    if merge_val:
        data_df = pd.concat([data_df, val_df], axis=0)

    sp_int, _, _ = interactions_to_sparse_matrix(
        data_df,
        items_num=dataset.items_num,
        users_num=dataset.users_num,
    )

    # computing user mat
    user_degree = np.array(sp_int.sum(axis=1))
    d_user_inv = np.power(user_degree, -beta).flatten()
    d_user_inv[np.isinf(d_user_inv)] = 0.0
    d_user_inv_diag = sps.diags(d_user_inv)

    item_degree = np.array(sp_int.sum(axis=0))
    d_item_inv = np.power(item_degree, -alpha).flatten()
    d_item_inv[np.isinf(d_item_inv)] = 0.0
    d_item_inv_diag = sps.diags(d_item_inv)

    d_item_alpha = np.power(item_degree, -alpha).flatten()
    d_item_alpha[np.isinf(d_item_alpha)] = 0.0
    d_item_alpha = sps.diags(d_item_alpha)

    d_item_inv_alpha = np.power(item_degree, alpha).flatten()
    d_item_inv_alpha[np.isinf(d_item_inv_alpha)] = 0.0
    d_item_inv_alpha_diag = sps.diags(d_item_inv_alpha)

    int_norm = d_user_inv_diag.dot(sp_int).dot(d_item_inv_diag)
    ut, s, vt = sparsesvd(int_norm.tocsc(), k)

    # perform SVD
    logger.info(
        set_color(
            f"Computing Eigenvalues and Eigenvectors it can take a while...:", "white"
        )
    )

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    eigh_dir_path = os.path.join(k_core_dir, f"eigh_alpha_{alpha}_beta_{beta}")
    Path(eigh_dir_path).mkdir(exist_ok=True)

    # save
    if merge_val:
        labels = ["user", "item", "singular_values"]
    else:
        labels = ["user_tv", "item_tv", "singular_values_tv"]
    for x in labels:
        with open(f"{eigh_dir_path}/{x}.npy", "wb") as f:
            if x == "user":
                np.save(f, ut)
            elif x == "item":
                np.save(f, vt)
            else:
                np.save(f, s)
    logger.info(
        set_color(
            f"SVD saved in {eigh_dir_path}",
            "white",
        )
    )


@timing
def load_svd(
    dataset: Dataset, merge_val=False, alpha=0.5, beta=0.5
) -> (np.array, np.array):
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    eigh_dir_path = os.path.join(k_core_dir, f"eigh_alpha_{alpha}_beta_{beta}")

    if merge_val:
        u = np.load(f"{eigh_dir_path}/user.npy")
        v = np.load(f"{eigh_dir_path}/item.npy")
        s = np.load(f"{eigh_dir_path}/singular_values.npy")
    else:
        u = np.load(f"{eigh_dir_path}/user_tv.npy")
        v = np.load(f"{eigh_dir_path}/item_tv.npy")
        s = np.load(f"{eigh_dir_path}/singular_values_tv.npy")
    return u, v, s


KCORE = 10
logging.basicConfig(level=logging.INFO)
DATASETS = [Movielens100k(), Movielens1M(), LastFM(), AmazonElectronics()]

if __name__ == "__main__":
    dataset = Gowalla()
    split_dict = dataset.load_split(k_core=KCORE, split_name="stratified_0.8_0.1_0.1")
    train, val, _ = split_dict["train"], split_dict["val"], split_dict["test"]

    for alpha in [0.3, 0.4, 0.5]:
        for beta in [0.3, 0.4, 0.5]:
            print(alpha)
            print(beta)
            compute_svd(
                data_df=train,
                dataset=dataset,
                k=3000,
                alpha=alpha,
                beta=beta,
                merge_val=True,
                val_df=val,
            )
