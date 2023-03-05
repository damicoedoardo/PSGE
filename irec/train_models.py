#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import os
from datetime import datetime

import wandb
from irec.data.implemented_datasets.movielens_100k import Movielens100k
from irec.early_stopping import EarlyStoppingHandlerTensorFlow
from irec.evaluation.topk_evaluator import Evaluator
from irec.sampler import TripletsBPRGenerator
from irec.trainer import DummyTrainer, TensorflowTrainer
from irec.utils import gpu_utils


def train_bpr_loss(model, dataset, train, test, args, val=None):
    # create data sampler
    data_generator = TripletsBPRGenerator(
        dataset=dataset,
        train_data=train,
        batch_size=args["batch_size"],
        items_after_users_ids=True,
    )

    # add the model name inside args
    args.update({"recommender_name": model.name})

    run_name = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    # initialize wandb
    if args["wandb"]:
        wandb.init(config=args)
        run_name = wandb.run.name + "_" + run_name

    early_stopping_handler = None
    val_evaluator = None
    if val is not None:
        val_evaluator = Evaluator(
            cutoff_list=args["cutoff"], metrics=args["metrics"], test_data=val
        )

        if args["early_stopping"]:
            early_stopping_handler = EarlyStoppingHandlerTensorFlow(
                patience=args["es_patience"],
                metric_name=args["es_metric"],
                save_best_model=True,
                split_name=args["dataset_split"],
                run_name=run_name,
            )

    trainer = TensorflowTrainer(
        model=model,
        train_generator=data_generator,
        epochs=args["epochs"],
        l2_reg=args["l2_reg"],
        learning_rate=args["learning_rate"],
        val_evaluator=val_evaluator,
        val_every=args["val_every"],
        early_stopping_handler=early_stopping_handler,
        wandb_log=args["wandb"],
    )

    # initialize wandb if needed
    if args["wandb"]:
        wandb.init(args)

    trainer.fit()
    if val is None:
        model.save(split_name=args["dataset_split"], run_name=run_name)

    # evaluate on test data
    model = model.__class__.load(
        dataset=dataset,
        train_data=train,
        split_name=args["dataset_split"],
        run_name=run_name,
    )

    test_evaluator = Evaluator(
        cutoff_list=args["cutoff"], metrics=args["metrics"], test_data=test
    )
    print("test")
    test_evaluator.evaluate_recommender(model, interactions=model.train_data)
    test_evaluator.print_evaluation_results()

    if args["wandb"]:
        test_result_dict = {}
        for k, v in test_evaluator.result_dict.items():
            new_key = "test_{}".format(k)
            test_result_dict[new_key] = test_evaluator.result_dict[k]
        wandb.log(test_result_dict)


def dummy_train(model, test, args, val=None, save_model=False):
    # add the model name inside args
    args.update({"recommender_name": model.name})

    run_name = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    # initialize wandb
    if "wandb" in args:
        if args["wandb"]:
            wandb.init(config=args)
            run_name = wandb.run.name
    else:
        args["wandb"] = False

    if val is not None:
        val_evaluator = Evaluator(
            cutoff_list=args["cutoff"], metrics=args["metrics"], test_data=val
        )

        trainer = DummyTrainer(
            model=model,
            val_evaluator=val_evaluator,
            wandb_log=args["wandb"],
        )

        trainer.fit()

    if save_model:
        model.save(split_name=args["dataset_split"], run_name=run_name)

    test_evaluator = Evaluator(
        cutoff_list=args["cutoff"], metrics=args["metrics"], test_data=test
    )

    test_evaluator.evaluate_recommender(model, interactions=model.train_data)
    test_evaluator.print_evaluation_results()

    if args["wandb"]:
        test_result_dict = {}
        for k, v in test_evaluator.result_dict.items():
            new_key = "test_{}".format(k)
            test_result_dict[new_key] = test_evaluator.result_dict[k]
        wandb.log(test_result_dict)

    if val is not None:
        return val_evaluator.result_dict, test_evaluator.result_dict
    else:
        return None, test_evaluator.result_dict
