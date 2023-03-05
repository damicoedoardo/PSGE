#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from irec.constants import *
from irec.data.dataset import Dataset
from irec.data.implemented_datasets import *
from irec.utils.graph_utils import (
    adjacency_from_interactions,
    get_laplacian,
    get_propagation_matrix,
    get_symmetric_normalized_laplacian,
)
from irec.utils.utils import set_color, timing
from scipy import linalg
from scipy.sparse.linalg import eigsh

logger = logging.getLogger(__name__)


@timing
def compute_eigendecomposition(
    data_df: pd.DataFrame,
    dataset: Dataset,
    kind: str = "laplacian",
    normalised: bool = False,
    k: int = None,
):
    """Compute eigenvalues and eigenvectors associated to the symmetric normalized laplacian"""
    P = get_propagation_matrix(data_df, dataset, kind, normalised)

    # perform eigendecomposition
    logger.info(
        set_color(
            f"Computing Eigenvalues and Eigenvectors it can take a while...:", "white"
        )
    )

    if k is not None:
        logger.info("Using Lanczos...")
        if kind == "adjacency":
            W, V = eigsh(P, k=k, which="LA")
        else:
            W, V = eigsh(P, k=k, which="SA")
    else:
        W, V = linalg.eigh(P.toarray())

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    eigh_dir_path = os.path.join(k_core_dir, "eigh")
    Path(eigh_dir_path).mkdir(exist_ok=True)

    save_name = f"{kind}_normalised" if normalised else f"{kind}"
    # save
    for x in ["eigenvectors", "eigenvalues"]:
        with open(f"{eigh_dir_path}/{save_name}_{x}.npy", "wb") as f:
            np.save(f, V) if x == "eigenvectors" else np.save(f, W)

    logger.info(
        set_color(
            f"Eigenvalues and eigenvectors saved in {eigh_dir_path} with name {save_name}",
            "white",
        )
    )


@timing
def load_eigendecomposition(
    dataset: Dataset,
    kind: str = "laplacian",
    normalised: bool = True,
    trgt: str = "both",
) -> (np.array, np.array):
    """Load eigenvalues and eigenvectors"""
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    eigh_dir_path = os.path.join(k_core_dir, "eigh")

    eigenvalues = None
    eigenvectors = None
    file_name = f"{kind}_normalised" if normalised else f"{kind}"
    if trgt == "both":
        _eigenvectors = np.load(f"{eigh_dir_path}/{file_name}_eigenvectors.npy")
        _eigenvalues = np.load(f"{eigh_dir_path}/{file_name}_eigenvalues.npy")
        logger.info(
            set_color(
                f"Eigenvalues and eigenvectors with name {file_name} loaded.", "white"
            )
        )
        eigenvectors = _eigenvectors
        eigenvalues = _eigenvalues
    elif trgt == "eigenvectors":
        _eigenvectors = np.load(f"{eigh_dir_path}/{file_name}_eigenvectors.npy")
        logger.info(set_color(f"eigenvectors with name {file_name} loaded.", "white"))
        eigenvectors = _eigenvectors
    elif trgt == "eigenvalues":
        _eigenvalues = np.load(f"{eigh_dir_path}/{file_name}_eigenvalues.npy")
        logger.info(set_color(f"eigenvalues with name {file_name} loaded.", "white"))
        eigenvalues = _eigenvalues
    return eigenvectors, eigenvalues


KCORE = 10
logging.basicConfig(level=LOGGING_LEVEL)
DATASETS = [Movielens100k(), Movielens1M(), LastFM(), AmazonElectronics()]
if __name__ == "__main__":
    # # load full dataset
    # full_data = dataset.load(k_core=KCORE)
    # compute_eigendecomposition(data_df=full_data, dataset=dataset, save_name="full")

    # for dataset in DATASETS:
    #     for kind in ["laplacian", "adjacency"]:
    #         for normalised in [True, False]:
    #             # load dataset train split
    #             split_dict = dataset.load_split(
    #                 k_core=KCORE, split_name="stratified_0.8_0.1_0.1"
    #             )
    #             train, _, _ = split_dict["train"], split_dict["val"], split_dict["test"]
    #             compute_eigendecomposition(
    #                 data_df=train, dataset=dataset, kind=kind, normalised=normalised
    #             )

    dataset = Movielens1M()
    kind = "laplacian"
    split_dict = dataset.load_split(k_core=KCORE, split_name="stratified_0.8_0.1_0.1")
    train, _, _ = split_dict["train"], split_dict["val"], split_dict["test"]
    compute_eigendecomposition(
        data_df=train, dataset=dataset, kind=kind, normalised=True, k=None
    )
