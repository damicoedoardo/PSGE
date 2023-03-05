#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.experiments.eigendecomposition_dataset import load_eigendecomposition
from irec.data.implemented_datasets import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

KCORE = 10
DATASETS = [Movielens100k(), Movielens1M(), LastFM(), AmazonElectronics()]
MODE = "train"

if __name__ == "__main__":
    df_list = []
    for dataset in DATASETS:
        dataset.load(k_core=KCORE)
        _, eig = load_eigendecomposition(dataset, MODE, trgt="eigenvalues")
        d = pd.DataFrame(eig, columns=["eigenvalues"])
        d["dataset"] = dataset.name
        df_list.append(d)
    # plot
    sns.set_theme(style="whitegrid")
    for df in df_list:
        ax = sns.histplot(data=df, x="eigenvalues")
        sns.ecdfplot(data=df, x="eigenvalues", ax=ax.twinx(), color="orange")
        plt.title(df["dataset"].values[0])
        plt.show()
