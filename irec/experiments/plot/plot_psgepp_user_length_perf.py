#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.experiments.plot.matplotlib_utils.setup_plot import setup_plot
from pathlib import Path
from irec.data.implemented_datasets import *
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DATASET = "Movielens1M"

if __name__ == "__main__":
    setup_plot(243, fig_ratio=0.6, style_sheet="base", font_size=8)

    dataset = eval(DATASET)()
    split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")

    # load dfs
    save_name = os.path.join(k_core_dir, "ul_res_df")
    res_df = pd.read_csv(save_name)
    save_name = os.path.join(k_core_dir, "ul_psgepp_res_df")
    sp_res_df = pd.read_csv(save_name)

    # filter on cutoff of interest
    filtered_res_df = res_df[res_df["cutoff"] == 20]
    filtered_res_df = filtered_res_df[filtered_res_df["metric_name"] == "Recall"]

    filtered_sp_res_df = sp_res_df[sp_res_df["cutoff"] == 20]
    filtered_sp_res_df = filtered_sp_res_df[
        filtered_sp_res_df["metric_name"] == "Recall"
    ]

    ### BARPLOT DISTRIBUTION ###
    filtered_res_df["User length cluster"] = list(
        map(lambda x: "$< {}$".format(x), filtered_res_df["split_max_count"])
    )
    filtered_sp_res_df["User length cluster"] = list(
        map(lambda x: "$< {}$".format(x), filtered_sp_res_df["split_max_count"])
    )

    ax = plt.gca()
    ax2 = ax.twinx()

    ax2.bar(
        x="User length cluster",
        height="split_user_count",
        data=filtered_res_df[filtered_res_df["alg"] == "BPR"],
        color="slateblue",
        alpha=0.2,
    )
    # set visibility to False
    ax2.get_yaxis().set_visible(False)

    sns.lineplot(
        data=filtered_sp_res_df,
        x="User length cluster",
        y="metric_score",
        hue="alpha",
        ax=ax,
        markers=True,
    )

    ax.set_ylabel("Recall@20")

    ### FIX LEGEND ###
    ax.legend_.remove()
    lines, labels = ax.get_legend_handles_labels()
    labels = ["$\\alpha=$" + f"{l}" for l in labels]
    legend = plt.legend(lines, labels, loc="best")
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(
        "{}/egcn_plot/{}.pdf".format(
            os.environ["HOME"], f"{DATASET}_alpha_user_length"
        ),
    )
    plt.show()

    # BPR = filtered_res_df[filtered_res_df["alg"] == "BPR"]
    # ax.plot("Item popularity cluster", "metric_score", data=BPR, ls="--", color="red", marker=".")
    #
    # LightGCN=filtered_res_df[filtered_res_df["alg"] == "LightGCN"]
    # ax.plot("Item popularity cluster", "metric_score", data=LightGCN, ls="-.", color="blue", marker="v")
    #
    # PureSVD=filtered_res_df[filtered_res_df["alg"] == "PureSVD"]
    # ax.plot("Item popularity cluster", "metric_score", data=PureSVD, ls="-.", color="green", marker="^")
    #
    # EASE=filtered_res_df[filtered_res_df["alg"] == "EASE"]
    # ax.plot("Item popularity cluster", "metric_score", data=EASE, ls="-.", color="purple", marker="P")
    #
    # SGMC=filtered_res_df[filtered_res_df["alg"] == "SGMC"]
    # ax.plot("Item popularity cluster", "metric_score", data=SGMC, ls="-.", color="orange", marker="X")

    # BPR=filtered_res_df[filtered_res_df["alg"] == "BPR"]
    # ax.scatter("Item popularity cluster", "metric_score", data=BPR, color="red", marker=".")
    #
    # LightGCN=filtered_res_df[filtered_res_df["alg"] == "LightGCN"]
    # ax.scatter("Item popularity cluster", "metric_score", data=LightGCN, color="blue", marker="v")
    #
    # PureSVD=filtered_res_df[filtered_res_df["alg"] == "PureSVD"]
    # ax.scatter("Item popularity cluster", "metric_score", data=PureSVD, color="green", marker="^")
    #
    # EASE=filtered_res_df[filtered_res_df["alg"] == "EASE"]
    # ax.scatter("Item popularity cluster", "metric_score", data=EASE, color="purple", marker="P")
    #
    # SGMC=filtered_res_df[filtered_res_df["alg"] == "SGMC"]
    # ax.scatter("Item popularity cluster", "metric_score", data=SGMC, color="orange", marker="X")

    # sns.swarmplot(
    #     data=filtered_res_df,
    #     x="Item popularity cluster",
    #     y="metric_score",
    #     hue="alg",
    #     style="alg",
    #     ax=ax,
    # )

    ### PLOT BASELINES ###
    # ax = sns.lineplot(
    #     data=filtered_res_df,
    #     x="Item popularity cluster",
    #     y="metric_score",
    #     hue="alg",
    #     style="alg",
    #     markers=True,
    #     ax=ax
    # )

    # sns.barplot(
    #     x="Item popularity cluster",
    #     y="split_item_count",
    #     color="royalblue",
    #     alpha=0.2,
    #     data=filtered_res_df[filtered_res_df["alg"] == "BPR"],
    #     ax = ax.twinx()
    # )

    # plt.bar(
    #     x="item popularity cluster",
    #     height="split_item_count",
    #     data=filtered_res_df[filtered_res_df["alg"] == "BPR"],
    #     color="royalblue",
    #     alpha=0.2,
    #     ax = ax.twinx()
    # )

    # plt.ylabel("Recall@20")

    ### FIX LEGEND ###
    # ax.legend_.set_title(None)
    # ax.legend_.get_frame().set_linewidth(0.5)

    # plt.show()
