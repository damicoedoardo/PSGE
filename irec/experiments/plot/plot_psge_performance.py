#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from irec.data.implemented_datasets import *
from irec.experiments.best_models import get_result_dict

dataset = Movielens1M()
KCORE = 10
METRIC = "Recall@20"
kind = "adjacency"
if __name__ == "__main__":
    dfs = []
    for normalised in [True]:  # , False]:
        preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
        k_core_dir = os.path.join(preprocessed_dir, f"k_core_{KCORE}")
        save_path = os.path.join(k_core_dir, "psge_results")
        save_name = f"{kind}_normalised.csv" if normalised else f"{kind}.csv"
        final_path = os.path.join(save_path, save_name)
        df = pd.read_csv(final_path, index_col=0)
        df["normalised"] = normalised
        dfs.append(df)
    res_df = pd.concat(dfs, axis=0).reset_index()

    # get best perf
    id_max = res_df[(res_df["split"] == "validation") & (res_df["normalised"] == 1)][
        "Recall@20"
    ].idxmax()
    k_best = res_df.loc[id_max]["k"]
    res_row = res_df[
        (res_df["split"] == "test")
        & (res_df["normalised"] == 1)
        & (res_df["k"] == k_best)
    ]
    res_dict = res_row.to_dict()
    for key, value in res_dict.items():
        print(key, " : ", value)

    sns.lineplot(x="k", y=METRIC, hue="normalised", style="split", data=res_df)
    # r = get_result_dict(dataset)
    # baselines
    # plt.axhline(y=r["BPR"][METRIC], linestyle="--", label="BPR", color="red")
    # plt.axhline(y=r["LightGCN"][METRIC], linestyle="--", label="LightGCN", color="m")
    # plt.axhline(y=r["EASE"][METRIC], linestyle="--", label="EASE", color="r")
    # plt.axhline(y=r["ItemKNN"][METRIC], linestyle="--", label="ItemKNN")
    # plt.getax.legend_.get_frame().set_linewidth(0.5)
    plt.title(dataset.name)
    plt.show()
