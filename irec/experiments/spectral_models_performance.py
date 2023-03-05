#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import wandb
from irec.data.implemented_datasets import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.model.psge.psge import PSGE
from irec.train_models import dummy_train
from irec.trainer import DummyTrainer
from irec.utils.parser_utils import (
    add_dataset_parameters,
    add_early_stopping_parameters,
    add_evaluation_parameter,
    add_train_parameters,
)
from irec.utils.utils import set_color
from tqdm import tqdm

logger = logging.getLogger(__name__)

best_param = {
    "Movielens1M": {
        "alpha": 0.4,
        "beta": 0.3,
    },
    "AmazonElectronics": {
        "alpha": 0.5,
        "beta": 0.3,
    },
    "Gowalla": {
        "alpha": 0.4,
        "beta": 0.3,
    },
}

K_MAX = 3000
K_STEP = 50
DATASET = "Gowalla"

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train PSGE")
    # Train parameters
    add_dataset_parameters(parser)
    # Early stopping parameters
    add_evaluation_parameter(parser)
    args = vars(parser.parse_args())
    print(args.keys())

    k_max = K_MAX
    dataset = eval(DATASET)()

    args["dataset"] = DATASET

    val_series, test_series = [], []
    k_list = [1]
    temp = list(range(0, k_max, K_STEP))[1:]
    k_list.extend(temp)

    for k in tqdm(k_list):
        # load data
        split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
        train, val, test = (
            split_dict["train"],
            split_dict["val"],
            split_dict["test"],
        )

        train = pd.concat([train, val], axis=0)

        # initialise the model
        alpha = best_param[DATASET]["alpha"]
        beta = best_param[DATASET]["beta"]

        model = PSGE(
            dataset=dataset,
            train_data=train,
            k=k,
            precomputed_svd=True,
            alpha=alpha,
            beta=beta,
        )

        _, test_res = dummy_train(model=model, test=test, args=args)
        test_res["k"] = k
        test_ser = pd.Series(test_res)
        test_series.append(test_ser)

    test_res_df = pd.concat(test_series, axis=1).T
    test_res_df["split"] = "test"
    res_df = pd.concat([test_res_df], axis=0)

    # create the save folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    save_path = os.path.join(k_core_dir, "psge_results")
    Path(save_path).mkdir(exist_ok=True)

    save_name = f"{model.name}.csv"
    final_path = os.path.join(save_path, save_name)
    res_df.to_csv(final_path)
