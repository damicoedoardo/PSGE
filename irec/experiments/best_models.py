#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import pandas as pd
from irec.model.bpr.bpr import BPR
from irec.model.lightgcn.lightgcn import LightGCN


def load_model_span_eigv(model_name, dataset, train):
    DATASET_SPLIT = "stratified_0.8_0.1_0.1"
    run_names_dict = get_span_model_run_name()
    run_name = run_names_dict[dataset.name][model_name]
    model_class = None
    if "LightGCN" in model_name:
        model_class = LightGCN
    else:
        model_class = BPR
    m = model_class.load(
        dataset=dataset,
        train_data=train,
        split_name=DATASET_SPLIT,
        run_name=run_name,
    )
    return m


def get_span_model_run_name():
    run_name_dict = {
        "MovieLens1M": {
            "BPR": "elated-sweep-3",
            "LightGCN_1": "sweet-sweep-2",
            "LightGCN_2": "worthy-sweep-11",
            "LightGCN_3": "different-sweep-20",
            "LightGCN_4": "electric-sweep-29",
        }
    }
    return run_name_dict


def load_model(model_class, dataset, train, merge_val=False, val=None):
    DATASET_SPLIT = "stratified_0.8_0.1_0.1"
    print(len(train))
    if merge_val:
        train = pd.concat([train, val], axis=0)
    if model_class.name in ["BPR", "LightGCN", "GLP-GCN"]:
        if merge_val:
            run_names_dict = get_best_model_trainval_run_name()
        else:
            run_names_dict = get_best_model_run_name()
        run_name = run_names_dict[dataset.name][model_class.name]
        m = model_class.load(
            dataset=dataset,
            train_data=train,
            split_name=DATASET_SPLIT,
            run_name=run_name,
        )
        return m
    else:
        best_params_dicts = get_best_params_dict()
        print(best_params_dicts)
        p_dict = best_params_dicts[dataset.name][model_class.name]
        p_dict["dataset"] = dataset
        print(len(train))
        p_dict["train_data"] = train
        m = model_class(**p_dict)
        return m


def get_best_model_trainval_run_name():
    run_name_dict = {
        "MovieLens1M": {
            "BPR": "chocolate-hill-5_12_01_2022__19_57_41",
            "LightGCN": "noble-durian-4_12_01_2022__19_56_21",
        },
        "Amazon Electronics": {
            "BPR": "generous-jazz-6_12_01_2022__20_36_17",
            "LightGCN": "treasured-donkey-7_12_01_2022__20_46_57",
        },
        "Gowalla": {
            "BPR": "fresh-planet-7_12_01_2022__20_48_10",
            "LightGCN": "zesty-music-10_12_01_2022__21_27_25",
        },
    }
    return run_name_dict


def get_best_model_run_name():
    run_name_dict = {
        "MovieLens1M": {
            "BPR": "elated-sweep-3",
            "LightGCN": "different-sweep-20",
            "GLP-GCN": "prime-sweep-26",
            "SGMC": "fanciful-sweep-4",
            "PSGE": "lemon-sweep-15",
            "PSIS": "devoted-sweep-6",
            "PSUS": "dark-sweep-5",
        },
        "Amazon Electronics": {
            "BPR": "silvery-sweep-7",
            "LightGCN": "helpful-sweep-8",
            "GLP-GCN": "lively-sweep-17",
            "SGMC": "vivid-sweep-18",
            "PSGE": "playful-sweep-14",
            "PSIS": "silvery-sweep-16",
            "PSUS": "lively-sweep-15",
        },
        "Gowalla": {
            "BPR": "playful-sweep-2",
            "LightGCN": "resilient-sweep-29",
            "GLP-GCN": "proud-sweep-36",
            "SGMC": "firm-sweep-5",
            "PSGE": "stoic-sweep-15",
            "PSIS": "rich-sweep-10",
            "PSUS": "eager-sweep-5",
        },
    }
    return run_name_dict


def get_best_params_dict():
    best_params_dict = {
        "MovieLens1M": {
            "SGMC": {"k": 80},
            "EASE": {"l2": 0.1},
            "PureSVD": {"n_components": 20},
            "PSGE": {"k": 80, "alpha": 0.4, "beta": 0.3},
        },
        "Amazon Electronics": {
            "SGMC": {"k": 40},
            "PSIS": {"k": 80, "beta": 0.5},
            "PSUS": {"k": 40, "beta": 0.5},
            "EASE": {"l2": 0.1},
            "PSGE++": {"k": 40, "alpha": 0.4},
            "PureSVD": {"n_components": 10},
            "PSGE": {"k": 80, "alpha": 0.5, "beta": 0.3},
        },
        "Gowalla": {
            "SGMC": {"k": 1500},
            "EASE": {"l2": 0.01},
            "PureSVD": {"n_components": 1500},
            "PSGE": {"k": 1500, "alpha": 0.4, "beta": 0.3, "precomputed_svd": True},
        },
    }
    return best_params_dict


def get_result_dict(dataset):
    r = {
        "MovieLens1M": {
            "BPR": {"Recall@5": 0.1028, "Recall@20": 0.2521},
            "LightGCN": {"Recall@5": 0.1079, "Recall@20": 0.2660},
            "EASE": {"Recall@5": 0.1194, "Recall@20": 0.2816},
        },
        "Amazon Electronics": {
            "BPR": {"Recall@5": 0.0346},
            "LightGCN": {"Recall@5": 0.0337},
            "EASE": {"Recall@5": 0.0378},
        },
    }
    return r[dataset.name]
