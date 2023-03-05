#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import os

import matplotlib.pyplot as plt
import seaborn as sns
from irec.data.implemented_datasets import *
from irec.evaluation.python_evaluation import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.experiments.best_models import load_model, load_model_span_eigv
from irec.experiments.eigendecomposition_dataset import load_eigendecomposition
from irec.experiments.plot.matplotlib_utils.setup_plot import setup_plot
from irec.experiments.span_eigenv_through_epoch import compute_corr
from irec.model.bpr.bpr import BPR
from irec.model.ease.ease import EASE
from irec.model.glpgcn.glpgcn import GLPGCN
from irec.model.lightgcn.lightgcn import LightGCN
from irec.model.psge_svd.psge_svd import PSGE
from irec.model.sgmc.sgmc import SGMC
from tqdm import tqdm

DATASET = "Movielens1M"
KCORE = 10
DATASET_SPLIT = "stratified_0.8_0.1_0.1"
MODELS = [LightGCN, BPR]
MAX_K = 20

if __name__ == "__main__":
    setup_plot(234, fig_ratio=0.4, style_sheet="base")
    # load data
    dataset = eval(DATASET)()
    dataset.load(k_core=KCORE)
    split_dict = dataset.load_split(k_core=KCORE, split_name=DATASET_SPLIT)
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # load eigenvectors
    eig, _ = load_eigendecomposition(dataset=dataset, kind="adjacency")

    corr_list = []
    k_list = []
    alg_name = []
    for m in MODELS:
        model = load_model(m, dataset, train)
        for i in tqdm(range(1, MAX_K)):
            f_eig = eig[:, -i:]
            # compute spectral coefficient
            c = compute_corr(f_eig, model)
            print(i)
            print(c)
            corr_list.append(c)
            k_list.append(i)
            alg_name.append(m.name)

    r_df = pd.DataFrame(zip(corr_list, alg_name, k_list), columns=["corr", "alg", "k"])

    line_style = {
        "LightGCN": [1, 0],
        "BPR": [2, 2],
    }

    colors = {
        "LightGCN": "black",
        "BPR": "black",
    }

    sns.lineplot(
        data=r_df,
        x="k",
        y="corr",
        style="alg",
        dashes=line_style,
        palette=colors,
        hue="alg",
    )

    legend = plt.legend(labels=["LightGCN", "BPR"])
    legend.get_frame().set_linewidth(0.5)

    # labels
    plt.ylabel("$\\rho(\\mathrm{X})$")
    plt.xlabel("$\ell$")

    plt.axhline(y=1, color="r", linestyle="dotted")
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(
        "{}/egcn_plot/{}.pdf".format(os.environ["HOME"], "progressive_span"),
        # bbox_inches="tight",
    )
    plt.show()
