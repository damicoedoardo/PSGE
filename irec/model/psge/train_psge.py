#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import argparse
import logging
from pickle import TRUE

import pandas as pd
import wandb
from irec.constants import *
from irec.data.implemented_datasets import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.experiments.svd_dataset import load_svd
from irec.model.psge.psge import PSGE
from irec.train_models import dummy_train
from irec.trainer import DummyTrainer
from irec.utils.parser_utils import add_dataset_parameters, add_evaluation_parameter

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train PSGE")
    # Model parameters
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=1500)
    # Dataset parameters
    add_dataset_parameters(parser)
    # Recommendation cutoff used for evaluation
    add_evaluation_parameter(parser)
    # wandb parameter
    parser.add_argument("--wandb", type=bool, default=True)
    args = vars(parser.parse_args())
    # load data
    dataset = eval(args["dataset"])()
    split_dict = dataset.load_split(
        k_core=args["k_core"], split_name=args["dataset_split"]
    )
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    if args["merge_val"]:
        # merge train and val data
        train = pd.concat([train, val], axis=0)
        val = None

    # initialise the model
    model = PSGE(
        dataset=dataset,
        train_data=train,
        alpha=args["alpha"],
        beta=args["beta"],
        k=args["k"],
        precomputed_svd=False,
    )
    dummy_train(model=model, val=val, test=test, args=args)
