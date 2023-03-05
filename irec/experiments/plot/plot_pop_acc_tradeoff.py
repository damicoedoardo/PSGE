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

DATASET = "Gowalla"
POP_KIND = "REL"

if __name__ == "__main__":
    if POP_KIND == "REL":
        pop = "rel_avg_pop"
        pop_name = "Relative Average Popularity"
    else:
        pop = "avg_pop"
        pop_name = "Average Popularity"

    setup_plot(162, fig_ratio=0.8, style_sheet="base", font_size=8)

    dataset = eval(DATASET)()
    split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")

    # load dfs
    save_name = os.path.join(k_core_dir, "res_df")
    res_df = pd.read_csv(save_name)
    save_name = os.path.join(k_core_dir, "sp_res_df")
    sp_res_df = pd.read_csv(save_name)

    # filter on cutoff of interest
    filtered_res_df = res_df[res_df["cutoff"] == 20]
    filtered_sp_res_df = sp_res_df[sp_res_df["cutoff"] == 20]

    # plot spectral model
    ax = sns.lineplot(
        data=filtered_sp_res_df,
        x=pop,
        y="metric_score",
        hue="alg",
        style="alg",
        palette=sns.color_palette("cool", 3),
        linewidth=1.5,
    )
    # plot on top scatter normal models
    sns.scatterplot(
        data=filtered_res_df,
        x=pop,
        y="metric_score",
        hue="alg",
        style="alg",
        ax=ax,
        # markers=['o', 'v', '^', "s"]
    )

    # labels
    if dataset.name == "MovieLens1M":
        plt.ylabel("Recall@20")
        # legend
        ax.legend_.set_title(None)
        ax.legend_.get_frame().set_linewidth(0.5)
    else:
        # plt.ylabel("")
        plt.ylabel("Recall@20")
        ax.legend_.remove()

    plt.xlabel(pop_name)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(
        "{}/egcn_plot/{}.pdf".format(os.environ["HOME"], f"{DATASET}_{pop_name}"),
    )
    plt.show()
