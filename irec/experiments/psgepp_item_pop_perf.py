#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.data.implemented_datasets import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.experiments.best_models import load_model
from irec.model.bpr.bpr import BPR
from irec.model.ease.ease import EASE
from irec.model.lightgcn.lightgcn import LightGCN
from irec.model.psgepp.psgepp import PSGE
from irec.model.puresvd.puresvd import PureSVD
from irec.model.sgmc.sgmc import SGMC
from irec.model.glpgcn.glpgcn import GLPGCN
from irec.evaluation.python_evaluation import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging

from irec.utils.utils import set_color

logger = logging.getLogger(__name__)

DATASET = "Gowalla"
KCORE = 10
DATASET_SPLIT = "stratified_0.8_0.1_0.1"
MODELS = [BPR, EASE, LightGCN, SGMC, PureSVD]
CUTOFFS = [20]
METRIC = ["Recall"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
N = 4


def item_pop_splitter(train_df, test_df, n):
    df_item_pop = (
        train_df.groupby(DEFAULT_ITEM_COL)
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )

    item_id = df_item_pop[DEFAULT_ITEM_COL].values
    count = df_item_pop["count"].values

    cum_sum = np.cumsum(count)
    partition_dim = cum_sum[-1] / n

    split_ids = []
    split_interactions = []
    for i in range(1, n):
        split_dim = partition_dim * i
        idx = (np.abs(cum_sum - split_dim)).argmin()
        split_ids.append(idx)
        split_interactions.append(count[idx])

    split_interactions.append(count[-1])
    splits = np.split(item_id, split_ids)

    split_item_count = []
    pop_test_dfs = []
    for split in splits:
        split_item_count.append(len(split))
        pop_test_dfs.append(test_df[test_df[DEFAULT_ITEM_COL].isin(split)])

    return pop_test_dfs, split_interactions, split_item_count


if __name__ == "__main__":
    # load data
    dataset = eval(DATASET)()
    dataset.load(k_core=KCORE)
    split_dict = dataset.load_split(k_core=KCORE, split_name=DATASET_SPLIT)
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # merge train and val
    train = pd.concat([train, val], axis=0)
    pop_test_dfs, split_max_counts, split_item_counts = item_pop_splitter(
        train, test, N
    )

    res_df_model_list = []

    for m in MODELS:
        for pop_test_df, split_max_count, split_item_count in zip(
            pop_test_dfs, split_max_counts, split_item_counts
        ):
            # setup test eval
            test_evaluator = Evaluator(
                cutoff_list=CUTOFFS, metrics=METRIC, test_data=pop_test_df
            )
            # load model
            model = load_model(m, dataset, train, merge_val=True, val=val)

            # retrieve result for popularity partition test
            test_evaluator.evaluate_recommender(model, train)
            res_df = test_evaluator.get_results_df()

            # adding required cols
            res_df["split_max_count"] = split_max_count
            res_df["split_item_count"] = split_item_count
            res_df["alg"] = model.name

            # append
            res_df_model_list.append(res_df)

    res_df_spectral_model_list = []

    for pop_test_df, split_max_count, split_item_count in zip(
        pop_test_dfs, split_max_counts, split_item_counts
    ):
        for alpha in ALPHAS:
            # setup test eval
            test_evaluator = Evaluator(
                cutoff_list=CUTOFFS, metrics=METRIC, test_data=pop_test_df
            )
            # load model
            model = load_model(PSGE, dataset, train, merge_val=True, val=val)
            model.alpha = alpha

            # retrieve result for popularity partition test
            test_evaluator.evaluate_recommender(model, train)
            res_df = test_evaluator.get_results_df()

            # adding required cols
            res_df["split_max_count"] = split_max_count
            res_df["split_item_count"] = split_item_count
            res_df["alg"] = model.name
            res_df["alpha"] = alpha

            # append
            res_df_spectral_model_list.append(res_df)

    final_res_df = pd.concat(res_df_model_list, axis=0)
    final_spectral_res_df = pd.concat(res_df_spectral_model_list, axis=0)

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    Path(k_core_dir).mkdir(exist_ok=True)
    for sn, df in zip(
        ["pop_res_df", "pop_psgepp_res_df"], [final_res_df, final_spectral_res_df]
    ):
        save_name = os.path.join(k_core_dir, sn)
        df.to_csv(save_name, index=False)
        logger.info(
            set_color(
                f"Pop rec saved in {sn}",
                "white",
            )
        )
