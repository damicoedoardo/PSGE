#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import argparse
import logging

import pandas as pd
from irec.constants import *
from irec.data.implemented_datasets import *
from irec.evaluation.topk_evaluator import Evaluator
from irec.model.ease.ease import EASE
from irec.model.itemKNN.itemKNN import ItemKNN
from irec.model.puresvd.puresvd import PureSVD
from irec.train_models import dummy_train
from irec.trainer import DummyTrainer
from irec.utils.parser_utils import add_dataset_parameters, add_evaluation_parameter

import wandb

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train PureSVD")
    # Model parameters
    parser.add_argument("--n_components", type=int, default=1500)
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
    model = PureSVD(
        dataset=dataset, train_data=train, n_components=args["n_components"]
    )
    dummy_train(model=model, val=val, test=test, args=args)
