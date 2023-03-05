#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.data.implemented_datasets import *
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from irec.model.psis.psis import PSIS
from irec.model.psus.psus import PSUS
from irec.model.psge_svd.psge_svd import PSGE


KCORE = 10
MODELS = ["PSGESVD", "PSUS", "PSIS"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser("performance")
    parser.add_argument("--dataset", type=str, default="AmazonElectronics")
    parser.add_argument("--metric", type=str, default="Recall")
    parser.add_argument("--cutoff", type=int, default=20)
    args = vars(parser.parse_args())

    metrics = ["Recall", "Precision", "NDCG"]
    cutoffs = [5, 20, 50]
    datasets = ["Movielens1M", "AmazonElectronics", "Gowalla"]
    assert args["dataset"] in datasets, f"Dataset should be in {datasets}"
    assert args["metric"] in metrics, f"Metric should be in {metrics}"
    assert args["cutoff"] in cutoffs, f"Cutoff should be in {metrics}"
    m_k = args["metric"] + "@" + str(args["cutoff"])
    dataset = eval(args["dataset"])()

    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{KCORE}")
    save_path = os.path.join(k_core_dir, "psge_results")
    dfs = []
    for model in MODELS:
        save_name = "{}.csv".format(eval(model).name)
        final_path = os.path.join(save_path, save_name)
        res_df = pd.read_csv(final_path, index_col=0)
        res_df["alg"] = eval(model).name
        dfs.append(res_df)

    res_df = pd.concat(dfs, axis=0).reset_index()

    # get best perf
    for alg in MODELS:
        id_max = res_df[
            (res_df["split"] == "validation") & (res_df["alg"] == eval(alg).name)
        ][m_k].idxmax()
        k_best = res_df.loc[id_max]["k"]
        res_row = res_df[
            (res_df["split"] == "test")
            & (res_df["k"] == k_best)
            & (res_df["alg"] == eval(alg).name)
        ]
        res_dict = res_row.to_dict()
        print("model: {}".format(eval(alg).name))
        for key, value in res_dict.items():
            print(key, " : ", value)
        print("")

    sns.lineplot(x="k", y=m_k, hue="alg", style="split", data=res_df)
    # r = get_result_dict(dataset)
    # baselines
    # plt.axhline(y=r["BPR"][METRIC], linestyle="--", label="BPR", color="red")
    # plt.axhline(y=r["LightGCN"][m_k], linestyle="--", label="LightGCN", color="m")
    # plt.axhline(y=r["EASE"][m_k], linestyle="--", label="EASE", color="r")
    # plt.axhline(y=r["ItemKNN"][METRIC], linestyle="--", label="ItemKNN")
    plt.legend()
    plt.title(dataset.name)
    plt.show()
