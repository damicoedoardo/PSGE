#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import argparse
import logging
import os

import pandas as pd
from irec.data.implemented_datasets import *
from irec.model.bpr.bpr import BPR
from irec.train_models import train_bpr_loss
from irec.utils import gpu_utils
from irec.utils.parser_utils import (
    add_train_parameters,
    add_early_stopping_parameters,
    add_dataset_parameters,
    add_evaluation_parameter,
)

logger = logging.getLogger(__name__)
# tf.config.run_functions_eagerly(True)
logging.basicConfig(level=LOGGING_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train BPR matrix factorization")

    # select free gpu if available
    if gpu_utils.list_available_gpus() is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_utils.pick_gpu_lowest_memory())

    # Model parameters
    parser.add_argument("--embedding_size", type=int, default=64)
    # Dataset parameters
    add_dataset_parameters(parser)
    # Recommendation cutoff used for evaluation
    add_evaluation_parameter(parser)
    # Train parameters
    add_train_parameters(parser)
    # Early stopping parameters
    add_early_stopping_parameters(parser)
    # wandb parameter
    parser.add_argument("--wandb", type=bool, default=False)
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
    model = BPR(
        dataset=dataset, train_data=train, embedding_size=args["embedding_size"]
    )

    train_bpr_loss(
        model=model, dataset=dataset, train=train, val=val, test=test, args=args
    )
