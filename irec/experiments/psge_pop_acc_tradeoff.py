#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from irec.data.implemented_datasets import *
from irec.evaluation.python_evaluation import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.experiments.best_models import load_model
from irec.model.bpr.bpr import BPR
from irec.model.ease.ease import EASE
from irec.model.glpgcn.glpgcn import GLPGCN
from irec.model.lightgcn.lightgcn import LightGCN
from irec.model.psge.psge import PSGE
from irec.model.puresvd.puresvd import PureSVD
from irec.model.sgmc.sgmc import SGMC
from irec.utils.utils import set_color

logger = logging.getLogger(__name__)

DATASET = "Gowalla"
KCORE = 10
DATASET_SPLIT = "stratified_0.8_0.1_0.1"

MODELS = [BPR, LightGCN, PureSVD, EASE, SGMC]
# MODELS = [EASE]
CUTOFFS = [20]
METRIC = ["NDCG"]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def get_item_pop_df(train_df):
    item_pop_df = train_df.groupby(DEFAULT_ITEM_COL).size().reset_index(name="item_pop")
    item_pop_df["item_pop"] = item_pop_df["item_pop"] / len(
        train_df[DEFAULT_USER_COL].unique()
    )
    return item_pop_df


def get_pop_rec(recs):
    cutoffs = []
    item_pop = []
    rel_item_pop = []
    for c in CUTOFFS:
        f_recs = recs[recs["item_rank"] <= c]
        merged = pd.merge(f_recs, item_pop_df, on=DEFAULT_ITEM_COL)

        relevant_recs = pd.merge(f_recs, test, on=DEFAULT_ITEM_COL, how="right")
        merged_relevant = pd.merge(relevant_recs, item_pop_df, on=DEFAULT_ITEM_COL)

        rel_avg_pop = merged_relevant["item_pop"].mean()
        avg_pop = merged["item_pop"].mean()
        cutoffs.append(c)
        item_pop.append(avg_pop)
        rel_item_pop.append(rel_avg_pop)

    avg_pop_df = pd.DataFrame(
        zip(cutoffs, item_pop, rel_item_pop),
        columns=["cutoff", "avg_pop", "rel_avg_pop"],
    )

    test_evaluator.evaluate_recommender(model, m_train)
    res_df = test_evaluator.get_results_df()
    print(model.name)
    print(res_df)
    merged_res = pd.merge(res_df, avg_pop_df, on="cutoff")
    merged_res["alg"] = model.name

    return merged_res


if __name__ == "__main__":
    # load data
    dataset = eval(DATASET)()
    dataset.load(k_core=KCORE)
    split_dict = dataset.load_split(k_core=KCORE, split_name=DATASET_SPLIT)
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    test_evaluator = Evaluator(cutoff_list=CUTOFFS, metrics=METRIC, test_data=test)
    # merge train and val
    m_train = pd.concat([train, val], axis=0)
    item_pop_df = get_item_pop_df(m_train)

    res_df_model_list = []

    for m in MODELS:
        model = load_model(m, dataset, train, merge_val=True, val=val)
        # retrive recs
        recs = model.recommend(cutoff=max(CUTOFFS), interactions=m_train)
        print(ndcg_at_k(rating_true=test, rating_pred=recs, relevancy_method=None))
        merged_res = get_pop_rec(recs)
        print(merged_res)
        res_df_model_list.append(merged_res)

    res_df_spectral_model_list = []

    for alpha in ALPHAS:
        model = load_model(PSGE, dataset, train, merge_val=True, val=val)
        model.alpha = alpha
        recs = model.recommend(cutoff=max(CUTOFFS), interactions=m_train)
        merged_res = get_pop_rec(recs)
        merged_res["alpha"] = alpha
        res_df_spectral_model_list.append(merged_res)

    final_res_df = pd.concat(res_df_model_list, axis=0)
    final_spectral_res_df = pd.concat(res_df_spectral_model_list, axis=0)

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    Path(k_core_dir).mkdir(exist_ok=True)
    for sn, df in zip(["res_df", "psge_res_df"], [final_res_df, final_spectral_res_df]):
        save_name = os.path.join(k_core_dir, sn)
        df.to_csv(save_name, index=False)
        logger.info(
            set_color(
                f"Pop rec saved in {sn}",
                "white",
            )
        )
