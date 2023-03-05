#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from time import time
from functools import wraps
import numpy as np
import scipy.sparse as sps
import logging
import pandas as pd
from irec.constants import *
import irec.utils.pandas_utils as pu
import pickle
import os

logger = logging.getLogger(__name__)


def create_directory(path):
    """create a directory in the specified path"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(set_color(f"Directory: {path} created", "cyan"))


def save_pickle(obj, path):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path + ".pkl", "rb") as f:
        return pickle.load(f)


def set_color(log, color, highlight=True):
    color_set = ["white", "red", "green", "yellow", "blue", "pink", "cyan", "black"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print("func:{}\n took: {} sec".format(f.__name__, te - ts))
        return result

    return wrap


@timing
def get_top_k(scores, top_k, sort_top_k=True):
    """Extract top K element from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (np.array): score matrix (users x items).
        top_k (int): number of top items to recommend.
        sort_top_k (bool): flag to sort top k results.

    Returns:
        np.array, np.array: indices into score matrix for each users top items, scores corresponding to top items.
    """

    logger.info(set_color(f"Sort_top_k:{sort_top_k}", "cyan"))
    # ensure we're working with a dense ndarray
    if isinstance(scores, sps.spmatrix):
        logger.warning(
            set_color("Scores are in a sparse format, densify them", "white")
        )
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            set_color(
                "Number of items is less than top_k, limiting top_k to number of items",
                "white",
            )
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    return np.array(top_items), np.array(top_scores)


def truncate_top_k(x, k):
    """Keep top_k highest values elements for each row of a numpy array

    Args:
        x (np.Array): numpy array
        k (int): number of elements to keep for each row

    Returns:
        np.Array: processed array
    """
    s = x.shape
    ind = np.argpartition(x, -k, axis=1)[:, :-k]
    rows = np.arange(s[0])[:, None]
    x[rows, ind] = 0
    return x


def interactions_to_sparse_matrix(interactions, users_num=None, items_num=None):
    """Convert interactions df into a sparse matrix

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

    user_ids_mapping_dict = None
    item_ids_mapping_dict = None

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

    row_data = interactions[DEFAULT_USER_COL].values
    col_data = interactions[DEFAULT_ITEM_COL].values
    data = np.ones(len(row_data))

    sp_m = sps.csr_matrix((data, (row_data, col_data)), shape=(row_num, col_num))
    return sp_m, user_ids_mapping_dict, item_ids_mapping_dict
