#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from irec.data.implemented_datasets import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.experiments.best_models import load_model, load_model_span_eigv
from irec.experiments.eigendecomposition_dataset import load_eigendecomposition
from irec.experiments.span_eigenv_through_epoch import compute_corr
from irec.model.psge_svd.psge_svd import PSGE
from irec.model.bpr.bpr import BPR
from irec.model.ease.ease import EASE
from irec.model.lightgcn.lightgcn import LightGCN
from irec.model.psis.psis import PSIS
from irec.model.psus.psus import PSUS
from irec.model.psge_svd.psge_svd import PSGE
from irec.model.sgmc.sgmc import SGMC
from irec.model.glpgcn.glpgcn import GLPGCN
from irec.evaluation.python_evaluation import *
import seaborn as sns
import matplotlib.pyplot as plt

DATASET = "Movielens1M"
KCORE = 10
DATASET_SPLIT = "stratified_0.8_0.1_0.1"
MODELS = ["LightGCN_1", "LightGCN_2", "LightGCN_3", "LightGCN_4", "BPR"]

if __name__ == "__main__":
    # load data
    dataset = eval(DATASET)()
    dataset.load(k_core=KCORE)
    split_dict = dataset.load_split(k_core=KCORE, split_name=DATASET_SPLIT)
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # load eigenvectors
    eig, _ = load_eigendecomposition(dataset=dataset)
    eig = eig[:, :5]

    corr_list = []
    alg_name = []
    for m in MODELS:
        model = load_model_span_eigv(m, dataset, train)
        # compute spectral coefficient
        c = compute_corr(eig, model)
        corr_list.append(c)
        alg_name.append(m)

        # sp_coeff = np.dot(np.transpose(eig), model().numpy()/np.linalg.norm(model().numpy()))
        # abs_sp_coeff =abs(sp_coeff)
        # norm_abs_sp_coeff = np.sum(abs_sp_coeff, axis=1)/np.sum(abs_sp_coeff)
        # cumsum = np.cumsum(norm_abs_sp_coeff)
        # corr_df = pd.DataFrame(zip(cumsum, np.arange(1, len(cumsum)+1)), columns=["corr", "eig"])
        # corr_df["alg"] = model.name
        # corr_dfs.append(corr_df)

    r_df = pd.DataFrame(zip(corr_list, alg_name), columns=["corr", "alg"])
    sns.barplot(data=r_df, x="alg", y="corr")
    plt.show()
