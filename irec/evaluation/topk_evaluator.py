#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging

from irec.evaluation.python_evaluation import *
from irec.model.recommender_interfaces import AbstractRecommender
from irec.utils.utils import set_color

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Top-K recommendation task Evaluator

    Evaluate a recommender algorithm according to specified metrics and cutoff

    Attributes:
        cutoff_list (list): list of cutoff used to retrieve the recommendations
        metrics (list): list of metrics to evaluate
        test_data (pd.DataFrame): ground truth data
    """

    def __init__(self, cutoff_list, metrics, test_data):
        """
        Top-K recommendation task Evaluator

        Evaluate a recommender algorithm according to specified metrics and cutoff

        Attributes:
            cutoff_list (list): list of cutoff used to retrieve the recommendations
            metrics (list): list of metrics to evaluate
            test_data (pd.DataFrame): ground truth data
        """
        self.test_data = self._check_test_data(test_data)

        self._check_metrics(metrics)
        self.metrics = metrics

        self.cutoff_list = cutoff_list

        # set by _check_cutoff
        self.max_cutoff = max(cutoff_list)
        self.test_data = test_data

        self.result_dict = {}
        self.recommender_name = None

    @staticmethod
    def _check_test_data(test_data):
        if DEFAULT_RATING_COL not in test_data:
            logger.warning(
                set_color(
                    "Adding rating = 1, considering implicit recommendation task",
                    "white",
                ),
            )
            test_data[DEFAULT_RATING_COL] = 1
        return test_data

    @staticmethod
    def _check_metrics(metrics):
        for m in metrics:
            if m not in metrics_dict:
                raise ValueError(
                    f"metric: {m} not available \n Available metrics: {metrics_dict.keys()}"
                )

    def evaluate_recommender(self, recommender, interactions):
        """Evaluate a recommendation system algorithm

        Args:
            recommender: algorithm to evaulate
            interactions (pd.DataFrame): user interactions
        """
        assert issubclass(
            recommender.__class__, AbstractRecommender
        ), "recommender passed is not extending class: {}".format(AbstractRecommender)

        # todo: check kwargs for recommend arguments
        # retrieve the users idxs for which retrieve predictions
        recommendation = recommender.recommend(
            cutoff=self.max_cutoff, interactions=interactions
        )

        for m in self.metrics:
            for c in self.cutoff_list:
                # keep recommendations up to c
                recs = recommendation[recommendation["item_rank"] <= c]

                print(m)
                print(c)
                metric_value = metrics_dict[m](
                    rating_pred=recs,
                    rating_true=self.test_data,
                    relevancy_method=None,
                )

                # update result dict
                self.result_dict[f"{m}@{c}"] = metric_value

    def print_evaluation_results(self):
        """Print evaluation results"""
        print(set_color("=== RESULTS ===", "white"))
        for k, v in self.result_dict.items():
            print(set_color("{}: {}".format(k, v), "pink"))

    def get_results_df(self):
        """Return results as a df
        | cutoff | metric_name | metric_score |
        """
        cutoff = []
        metric_score = []
        metric_name = []
        for k, v in self.result_dict.items():
            m_name, c = k.split("@")
            cutoff.append(int(c))
            metric_score.append(v)
            metric_name.append(m_name)
        res_df = pd.DataFrame(
            zip(cutoff, metric_name, metric_score),
            columns=["cutoff", "metric_name", "metric_score"],
        )
        return res_df
