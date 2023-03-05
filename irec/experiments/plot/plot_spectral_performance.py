import glob
import os

from irec.experiments.plot.matplotlib_utils.setup_plot import setup_plot
from irec.experiments.svd_dataset import load_svd
from irec.model.psge_svd.psge_svd import PSGE
from irec.model.psis.psis import PSIS
from irec.model.psus.psus import PSUS
import pandas as pd
from irec.data.implemented_datasets import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DATASET = "Gowalla"
if __name__ == "__main__":
    setup_plot(162, fig_ratio=0.7, style_sheet="base", font_size=8)
    # load data
    dataset = eval(DATASET)()
    dataset.load(k_core=10)
    split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # get save folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")

    models_df_list = []
    for model in [PSGE, PSIS, PSUS]:
        save_path = os.path.join(k_core_dir, f"{model.name}")
        li = []
        all_files = glob.glob(os.path.join(save_path, "*.csv"))
        for fn in all_files:
            df = pd.read_csv(fn, index_col=None, header=0)
            li.append(df)
        frame = pd.concat(li, axis=0, ignore_index=True)
        models_df_list.append(frame)

    res_df = pd.concat(models_df_list, axis=0, ignore_index=True)
    # load eigenvalues
    _, _, s = load_svd(dataset, merge_val=True)

    cumsum_eig = s[:-1]
    ks = [a for a in range(1, len(cumsum_eig + 1))]
    eigs_df = pd.DataFrame(zip(ks, cumsum_eig), columns=["k", "eig"])
    f_res_df = pd.merge(res_df, eigs_df, on="k")

    f_res_df = f_res_df.sort_values(by="k")

    # f_res_df = f_res_df[f_res_df["k"]<=50]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sns.lineplot(
        data=f_res_df, x="k", y="Recall@20", hue="alg", linewidth=1, ax=ax
    )  # , style="alg")
    ax.tick_params(direction="in", length=1.5, width=0.5)

    # labels
    plt.xlabel("$\ell$")
    ax.legend_.remove()

    ax2 = ax.twinx()
    ax2.tick_params(direction="in", length=1.5, width=0.5)
    ax2.plot(f_res_df["k"], f_res_df["eig"], color="red", linestyle="--")
    # sns.lineplot(data=f_res_df, x="k", y="eig", ax=ax2, color="red", linestyle="--")
    plt.ylabel("$\lambda$")

    if dataset.name == "MovieLens1M":
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        labels2 = ["$\lambda$"]
        legend = plt.legend(lines + lines2, labels + labels2, loc="best", bbox_to_anchor=(0.5, 0., 0.5, 0.55))
        legend.get_frame().set_linewidth(0.5)

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig("{}/egcn_plot/{}.pdf".format(os.environ["HOME"], f"{DATASET}_sp_perf"),)
    plt.show()
