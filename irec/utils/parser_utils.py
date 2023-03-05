#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"


def add_train_parameters(parser):
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--l2_reg", type=float, default=1e-5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=390)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--models_to_save", type=int, default=5)


def add_early_stopping_parameters(parser):
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_metric", type=str, default="Recall@20")


def add_dataset_parameters(parser):
    parser.add_argument("--dataset", type=str, default="Movielens1M")
    parser.add_argument("--k_core", type=int, default=10)
    parser.add_argument("--dataset_split", type=str, default="stratified_0.8_0.1_0.1")


def add_evaluation_parameter(parser):
    parser.add_argument("--cutoff", type=list, default=[5, 20, 50])
    parser.add_argument("--metrics", type=list, default=["Recall", "NDCG", "Precision"])
    parser.add_argument("--merge_val", type=bool, default=True)
