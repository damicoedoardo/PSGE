#!/usr/bin/env python

import argparse
import logging

import pandas as pd
import wandb
from irec.constants import *
from irec.data.implemented_datasets import *
from irec.evaluation.python_evaluation import ndcg_at_k
from irec.evaluation.topk_evaluator import Evaluator
from irec.model.ease.ease import EASE
from irec.train_models import dummy_train
from irec.trainer import DummyTrainer
from irec.utils.parser_utils import (add_dataset_parameters,
                                     add_evaluation_parameter)

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOGGING_LEVEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("train Ease")
    # Model parameters
    parser.add_argument("--l2", type=float, default=0.1)
    # Dataset parameters
    add_dataset_parameters(parser)
    # Recommendation cutoff used for evaluation
    add_evaluation_parameter(parser)
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
        print(len(train))
        train = pd.concat([train, val], axis=0)
        val = None
        print(len(train))

    # initialise the model
    model = EASE(dataset=dataset, train_data=train, l2=args["l2"])
    recs = model.recommend(cutoff=20, interactions=train)
    test[DEFAULT_RATING_COL] = 1
    print(ndcg_at_k(rating_true=test, rating_pred=recs, relevancy_method=None))
    dummy_train(model=model, val=val, test=test, args=args)
