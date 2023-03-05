#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging
import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
import tensorflow.keras as keras

from irec.constants import *
from irec.data.dataset import Dataset
from irec.utils.utils import (
    set_color,
    timing,
    get_top_k,
    interactions_to_sparse_matrix,
    create_directory,
    save_pickle,
    load_pickle,
)

logger = logging.getLogger(__name__)


class AbstractRecommender(ABC):
    """Interface for recommender system algorithms

    Attributes:
        dataset (Dataset)
        train_data (pd.DataFrame): dataframe containing user-item interactions

    Note:
        For each user-item pair is expected a row in the dataframe, additional columns are allowed

    Example:
        >>> train_data = pd.DataFrame({"userID":[0, 0, 1], "itemID":[0, 1, 2]})
    """

    name = "Abstract Recommender"

    def __init__(self, dataset, train_data):
        for c in [DEFAULT_USER_COL, DEFAULT_ITEM_COL]:
            assert c in train_data.columns, f"column {c} not present in train_data"
        assert (
            dataset.dataset is not None
        ), "Dataset passed has not been loaded ! Call load() method"
        self.train_data = train_data
        self.dataset = dataset

    @classmethod
    def _get_model_save_path(cls, dataset, split_name):
        base_path = os.path.expanduser(os.path.join("~", MODELS_SAVE_PATH))
        dataset, kcore = dataset.name, dataset.k_core
        model_name = cls.name
        save_path = base_path + "/{}/k_core_{}/{}/{}/".format(
            dataset, kcore, split_name, model_name
        )
        create_directory(save_path)
        return save_path

    @classmethod
    def _load_state_dict(cls, load_path, name):
        sd_path = load_path + f"{name}/" + "state_dict"
        sd = load_pickle(sd_path)
        logger.info(set_color("Loaded state dict with name: {}".format(name), "yellow"))
        logger.info(set_color("model: {} name: {}".format(cls.name, name), "cyan"))
        return sd

    @abstractmethod
    def predict(self, interactions):
        """Compute items scores for each user inside interactions
        Args:
            interactions (pd.DataFrame): user interactions
        Returns:
            pd.DataFrame: items scores for each user
        """
        pass

    @abstractmethod
    def state_dict(self):
        """Return dictionary containing hyperparameters of the model"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        """Load model"""
        pass

    def save(self, split_name, run_name, *args, **kwargs):
        """Save the model

        For every model we save the state dict and depending on the model type other important parameters
        e.g. tensorflow model we save also the weights

        We save the model in the following path:
        MODELS_SAVE_PATH
            dataset
                kcore
                    split_name
                        model_name
                            run_name
                                state_dict
                                other_params
        """
        base_save_path = self._get_model_save_path(
            dataset=self.dataset, split_name=split_name
        )
        save_path = base_save_path + run_name
        create_directory(save_path)
        save_name = save_path + "/state_dict"
        save_pickle(self.state_dict(), save_name)
        logger.info(
            set_color(
                "Saved model state dict: {} | state dict: {} -> Path: {}".format(
                    self.name, run_name, save_name
                ),
                "yellow",
            )
        )

    @staticmethod
    @timing
    def remove_seen_items(scores, interactions):
        """Methods to set scores of items used at training time to `-np.inf`

        Args:
            scores (pd.DataFrame): items scores for each user, indexed by user id
            interactions (pd.DataFrane): interactions of the users for which retrieve predictions

        Returns:
            pd.DataFrame: dataframe of scores for each user indexed by user id
        """

        logger.info(set_color(f"Removing seen items", "cyan"))
        user_list = interactions[DEFAULT_USER_COL].values
        item_list = interactions[DEFAULT_ITEM_COL].values

        scores_array = scores.values

        user_index = scores.index.values
        arange = np.arange(len(user_index))
        mapping_dict = dict(zip(user_index, arange))
        user_list_mapped = np.array([mapping_dict.get(u) for u in user_list])

        scores_array[user_list_mapped, item_list] = -np.inf
        scores = pd.DataFrame(scores_array, index=user_index)

        return scores

    def recommend(self, cutoff, interactions):
        """
        Give recommendations up to a given cutoff to users inside `user_idxs` list

        Args:
            cutoff (int): cutoff used to retrieve the recommendations
            interactions (pd.DataFrame): interactions of the users for which retrieve predictions

        Returns:
            pd.DataFrame: DataFrame with predictions for users

        Note:
            predictions are in the following format | userID | itemID | prediction | item_rank
        """

        logger.info(set_color(f"Recommending items", "cyan"))
        # compute scores
        scores = self.predict(interactions)

        # set the score of the items used during the training to -inf
        scores_df = AbstractRecommender.remove_seen_items(scores, interactions)

        array_scores = scores_df.to_numpy()
        user_ids = scores_df.index.values

        # todo we can use tensorflow here
        items, scores = get_top_k(scores=array_scores, top_k=cutoff, sort_top_k=True)
        # create user array to match shape of retrievied items
        users = np.repeat(user_ids, cutoff).reshape(len(user_ids), -1)

        recs_df = pd.DataFrame(
            zip(users.flatten(), items.flatten(), scores.flatten()),
            columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL],
        )

        # add item rank
        recs_df["item_rank"] = np.tile(np.arange(1, cutoff + 1), len(user_ids))
        return recs_df


class ItemSimilarityRecommender(AbstractRecommender, ABC):
    """Item similarity matrix recommender interface

    Each recommender extending this class has to implement compute_similarity_matrix() method
    """

    def __init__(self, dataset, train_data):
        super().__init__(dataset=dataset, train_data=train_data)
        self.similarity_matrix = None

    @abstractmethod
    def compute_similarity_matrix(self):
        """Compute similarity matrix and assign it to self.similarity_matrix"""
        pass

    def predict(self, interactions):
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        sp_int, u_md, _ = interactions_to_sparse_matrix(
            interactions, items_num=self.dataset.items_num
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        scores = sp_int @ self.similarity_matrix
        scores_df = pd.DataFrame(scores, index=list(u_md.keys()))
        return scores_df

    def save(self, split_name, run_name, *args, **kwargs):
        logger.warning(
            set_color(
                "for ItemSimilarityRecommender we DO NOT save the similarity matrix for space efficiency"
                " only the state dict is saved",
                "white",
            )
        )
        super(ItemSimilarityRecommender, self).save(
            split_name=split_name, run_name=run_name
        )

    def load(self, split_name, run_name, *args, **kwargs):
        logger.warning(
            set_color(
                "for ItemSimilarityRecommender we retrain the model due to space efficiency",
                "white",
            )
        )
        base_load_path = self._get_model_save_path(
            dataset=self.dataset, split_name=split_name
        )
        self._load_state_dict(base_load_path, run_name)
        self.compute_similarity_matrix()


class UserSimilarityRecommender(AbstractRecommender, ABC):
    """Item similarity matrix recommender interface

    Each recommender extending this class has to implement compute_similarity_matrix() method
    """

    def __init__(self, dataset, train_data):
        super().__init__(dataset=dataset, train_data=train_data)
        self.similarity_matrix = None

    @abstractmethod
    def compute_similarity_matrix(self):
        """Compute similarity matrix and assign it to self.similarity_matrix"""
        pass

    def predict(self, interactions):
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        sp_int, u_md, _ = interactions_to_sparse_matrix(
            interactions, items_num=self.dataset.items_num
        )
        # compute scores as the dot product between user interactions and the similarity matrix
        scores = self.similarity_matrix @ sp_int
        scores_df = pd.DataFrame(scores, index=list(u_md.keys()))
        return scores_df

    def save(self, split_name, run_name, *args, **kwargs):
        logger.warning(
            set_color(
                "for ItemSimilarityRecommender we DO NOT save the similarity matrix for space efficiency"
                " only the state dict is saved",
                "white",
            )
        )
        super(UserSimilarityRecommender, self).save(
            split_name=split_name, run_name=run_name
        )

    def load(self, split_name, run_name, *args, **kwargs):
        logger.warning(
            set_color(
                "for ItemSimilarityRecommender we retrain the model due to space efficiency",
                "white",
            )
        )
        base_load_path = self._get_model_save_path(
            dataset=self.dataset, split_name=split_name
        )
        self._load_state_dict(base_load_path, run_name)
        self.compute_similarity_matrix()


class RepresentationsBasedRecommender(AbstractRecommender, ABC):
    """Representation based algorithm interface

    Interface for recommendation system algorithms which learn users and items embeddings to retrieve recommendation

    We use `pandas` dataframe to store the representations for both user and item, the dataframes have to be indexed by
    the user and item idxs

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
    """

    def __init__(self, train_data, dataset):
        super().__init__(dataset=dataset, train_data=train_data)

    @abstractmethod
    def compute_representations(self, interactions):
        """Compute users and items representations

        Args:
            interactions (pd.Dataframe): interactions of the users for which
                retrieve predictions stored inside a pd.DataFrame

        Returns:
            pd.DataFrame, pd.DataFrame: user representations, item representations
        """
        pass

    def predict(self, interactions):
        # we user the dor product between user and item embeddings to predict the user preference scores
        users_repr_df, items_repr_df = self.compute_representations(interactions)

        assert isinstance(users_repr_df, pd.DataFrame) and isinstance(
            items_repr_df, pd.DataFrame
        ), "Representations have to be stored inside pd.DataFrane objects!\n user: {}, item: {}".format(
            type(users_repr_df), type(items_repr_df)
        )
        assert (
            users_repr_df.shape[1] == items_repr_df.shape[1]
        ), "Users and Items representations have not the same shape!\n user: {}, item: {}".format(
            users_repr_df.shape[1], items_repr_df.shape[1]
        )

        # sort items representations
        items_repr_df.sort_index(inplace=True)

        # compute the scores as dot product between users and items representations
        arr_scores = users_repr_df.to_numpy().dot(items_repr_df.to_numpy().T)
        scores = pd.DataFrame(arr_scores, index=users_repr_df.index)
        return scores


class TensorflowRecommender(RepresentationsBasedRecommender, keras.Model, ABC):
    """Representation based algorithm interface

    Interface for recommendation system algorithms which learn users and items embeddings to retrieve recommendation

    We use `pandas` dataframe to store the representations for both user and item, the dataframes have to be indexed by
    the user and item idxs

    Attributes:
        train_data (pd.DataFrame): dataframe containing user-item interactions
    """

    def __init__(self, train_data, dataset, *args, **kwargs):
        RepresentationsBasedRecommender.__init__(
            self, dataset=dataset, train_data=train_data
        )
        keras.Model.__init__(self)

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Compute the loss function for a given batch of data"""
        pass

    def save(self, split_name, run_name, *args, **kwargs):
        super(TensorflowRecommender, self).save(
            split_name=split_name, run_name=run_name
        )
        save_path = self._get_model_save_path(
            dataset=self.dataset, split_name=split_name
        )
        save_name = save_path + "{}/weights".format(run_name)
        self.save_weights(save_name)
        logger.info(
            set_color(
                "Saved model: {} | weights: {} -> Path: {}".format(
                    self.name, run_name, save_name
                ),
                "yellow",
            )
        )

    @classmethod
    def load(cls, dataset, train_data, split_name, run_name, *args, **kwargs):
        base_load_path = cls._get_model_save_path(
            dataset=dataset, split_name=split_name
        )
        load_path = base_load_path + f"{run_name}/weights"
        state_dict = cls._load_state_dict(base_load_path, run_name)
        model = cls(dataset=dataset, train_data=train_data, **state_dict)
        model.load_weights(load_path)
        logger.info(
            set_color(
                "Loaded weights with run_name: {}".format(run_name),
                "yellow",
            )
        )
        return model
