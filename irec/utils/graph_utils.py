#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sps
import math
import irec.utils.pandas_utils as pu
from irec.constants import *
from irec.utils.utils import set_color

logger = logging.getLogger(__name__)


def adjacency_from_interactions(interactions, users_num=None, items_num=None):
    """Return adjacency from interactions

    Args:
        interactions (pd.DataFrame): interactions data
        users_num (int): number of users, used to initialise the shape of the sparse matrix,
            if None user ids are remapped to consecutive
        items_num (int): number of items, used to initialise the shape of the sparse matrix
            if None user ids are remapped to consecutive

    Returns:
        sp_m (sps.csr_matrix): users interactions in csr sparse format
        user_ids_mapping_dict (dict): dictionary mapping user ids to the rows of the sparse matrix
        user_ids_mapping_dict (dict): dictionary mapping item ids to the cols of the sparse matirx
    """
    for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
        assert c in interactions.columns, f"column {c} not present in train_data"

    row_num = users_num
    col_num = items_num

    if users_num is None:
        interactions, user_ids_mapping_dict = pu.remap_column_consecutive(
            interactions, DEFAULT_USER_COL
        )
        logger.warning(
            set_color("users_num is None, remap user ids to consecutive", "white")
        )
        row_num = len(user_ids_mapping_dict.keys())

    if items_num is None:
        interactions, item_ids_mapping_dict = pu.remap_column_consecutive(
            interactions, DEFAULT_ITEM_COL
        )
        logger.warning(
            set_color("items_num is None, remap item ids to consecutive", "white")
        )
        col_num = len(item_ids_mapping_dict.keys())

    user_data = interactions[DEFAULT_USER_COL].values
    item_data = interactions[DEFAULT_ITEM_COL].values + row_num

    row_data = np.concatenate((user_data, item_data), axis=0)
    col_data = np.concatenate((item_data, user_data), axis=0)

    data = np.ones(len(row_data))

    adj = sps.csr_matrix(
        (data, (row_data, col_data)), shape=(row_num + col_num, col_num + row_num)
    )
    return adj


def get_symmetric_normalized_laplacian(adj, self_loop=True):
    """Symmetric normalized Laplacian Matrix

    Compute the symmetric normalized Laplacian matrix of a given networkx graph, if `self_loop` is True
    self loop is added to the initial graph before computing the Laplacian matrix

    .. math::
        S = D^{- \\frac{1}{2}} A D^{- \\frac{1}{2}}

    Args:
        adj : adjacency matrix of the graph graph
        self_loop (bool): if add self loop to the initial graph

    Returns:
        sps.csr_matrix: sparse matrix containing symmetric normalized Laplacian
    """

    if self_loop:
        # convert to lil matrix for efficency of the method set diag
        adj = adj.tolil()
        # note: setdiag is an inplace operation
        adj.setdiag(1)
        # bring back the matrix into csr format
        adj = adj.tocsr()

    # compute the degree matrix D
    degree = np.array(adj.sum(axis=0)).squeeze() ** (-0.5)
    D = sps.diags(degree, format="csr")
    S = D * adj * D
    return S


def get_laplacian(adj):
    # compute the degree matrix D
    degree = np.array(adj.sum(axis=0)).squeeze()
    D = sps.diags(degree, format="csr")
    S = D - adj
    return S


def get_random_walk_matrix(adj, self_loop=True):
    if self_loop:
        # convert to lil matrix for efficency of the method set diag
        adj = adj.tolil()
        # note: setdiag is an inplace operation
        adj.setdiag(1)
        # bring back the matrix into csr format
        adj = adj.tocsr()

    # compute the degree matrix D
    degree = np.array(adj.sum(axis=0)).squeeze()
    # dd = degree**(0.5)
    # DD = sps.diags(dd, format="csr")
    D = sps.diags(1 / degree, format="csr")
    S = D * adj
    return S


def get_perfect_low_pass(adj, perc_cut=0.1):
    S = get_symmetric_normalized_laplacian(adj, self_loop=False)
    # S = get_random_walk_matrix(adj, self_loop=False)
    I = sps.eye(S.shape[0])
    # L_sym = I-DAD
    lsym = I - S
    W, V = np.linalg.eigh(lsym.toarray())
    num_eigenvectors = math.ceil(adj.shape[0] * perc_cut)
    low_eigv = V[:, :num_eigenvectors]
    high_eigv = V[:, (adj.shape[0] - num_eigenvectors) :]
    # eig=np.concatenate([low_eigv, high_eigv], axis=1)
    eig = low_eigv
    P = np.dot(eig, eig.T)
    return P


def get_perfect_band_pass(adj, delta=0.1):
    S = get_symmetric_normalized_laplacian(adj, self_loop=False)
    # S = get_random_walk_matrix(adj, self_loop=False)
    I = sps.eye(S.shape[0])
    # L_sym = I-DAD
    lsym = I - S
    W, V = np.linalg.eigh(lsym.toarray())
    lower_limit = 1 - delta
    uppper_limit = 1 + delta

    lower_index = (np.abs(W - lower_limit)).argmin()
    upper_index = (np.abs(W - uppper_limit)).argmin()

    low_eigv = V[:, :lower_index]
    high_eigv = V[:, upper_index:]

    eig = np.concatenate([low_eigv, high_eigv], axis=1)
    # eig = low_eigv
    P = np.dot(eig, eig.T)
    return P


def get_propagation_matrix(data_df, dataset, kind: str, normalised: bool):
    # get adjacency matrix
    kinds = ["adjacency", "laplacian"]
    assert kind in kinds, f"kind not in {kinds}"
    adj = adjacency_from_interactions(
        data_df, users_num=dataset.users_num, items_num=dataset.items_num
    )
    P = get_laplacian(adj) if kind == "laplacian" else adj

    if normalised:
        # user_degree=data_df.groupby(DEFAULT_USER_COL).count().values.squeeze()
        # item_degree=data_df.groupby(DEFAULT_ITEM_COL).count().values.squeeze()
        # popt_user, popt_item=dataset.user_item_degree_distr(data_df, plot=False)
        #
        # user_degree=np.power(user_degree,-popt_user[0])
        # item_degree=np.power(item_degree,-popt_item[0])
        #
        # degree=np.concatenate((user_degree, item_degree), axis=0)

        degree = np.array(adj.sum(axis=0)).squeeze() ** (-0.5)
        # degree=np.array(adj.sum(axis=0)).squeeze() ** (-1)
        D = sps.diags(degree, format="csr")
        P = D * P * D
    return P
