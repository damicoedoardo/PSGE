from pathlib import Path

import pandas as pd
import numpy as np
import os
from irec.data.implemented_datasets import *
import seaborn as sns
import matplotlib.pyplot as plt

from irec.experiments.plot.matplotlib_utils.setup_plot import setup_plot

DATASET = "Movielens1M"

if __name__ == "__main__":
    setup_plot(234, fig_ratio=0.4, style_sheet="base", font_size=8)

    dataset = eval(DATASET)()
    split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    Path(k_core_dir).mkdir(exist_ok=True)
    save_name = os.path.join(k_core_dir, "span_eig_through_epoch.csv")
    df = pd.read_csv(save_name)

    palette=sns.color_palette("viridis", 4)[::-1]
    colors = {
        "LightGCN_1": palette[0],
        "LightGCN_2": palette[1],
        "LightGCN_3": palette[2],
        "LightGCN_4": palette[3],
        "BPR": "black",
    }

    sizes = {
        "LightGCN_1": 1,
        "LightGCN_2": 1.4,
        "LightGCN_3": 1.8,
        "LightGCN_4": 2.2,
        "BPR": 1,
    }

    line_style = {
        "LightGCN_1":[1, 0],
        "LightGCN_2":[1, 0],
        "LightGCN_3":[1, 0],
        "LightGCN_4":[1, 0],
        "BPR": [2, 2],
    }

    g = sns.lineplot(
        data=df,
        x="epoch",
        y="corr",
        hue="alg",
        palette=colors,
        #linewidth=3,
        sizes=sizes,
        size="alg",
        style="alg",
        dashes=line_style,
    )

    #labels
    plt.ylabel("$\\rho(\\mathrm{X})$")
    plt.xlabel("Epoch")

    legend=plt.legend(labels=["$k=1$", "$k=2$", "$k=3$", "$k=4$", "BPR"])
    legend.get_frame().set_linewidth(0.5)

    #plt.tight_layout(pad=0.5)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(
        "{}/egcn_plot/{}.pdf".format(os.environ["HOME"], "span_corr_through_epoch"),
        #bbox_inches="tight",
    )
    plt.show()
