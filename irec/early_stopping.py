#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil
from tensorflow import keras

import logging

from irec.utils.utils import set_color

logger = logging.getLogger(__name__)


class EarlyStoppingHandlerTensorFlow:
    """
    TensorFlow Early Stopping Handler

    Early stopping handler stop training when the watched metric is not improving and save the best
    model weights

    Attributes:
        patience (int): number of epoch without improvement to wait before stopping the training
            procedure
        save_path (str): path where to save the weights of the best model found
    """

    def __init__(
        self,
        patience,
        metric_name,
        save_best_model=False,
        split_name=None,
        run_name=None,
    ):
        """
        TensorFlow Early Stopping Handler

        Early stopping handler stop training when the watched metric is not improving and save the best
        model weights

        Args:
            patience (int): number of epoch without improvement to wait before stopping the training
                procedure
            metric_name (str): metric to check for improvement
            save_best_model (bool): whether to save the best model
            split_name (str): used to retrieve the save path of the model
        """
        if save_best_model:
            assert (
                split_name is not None and run_name is not None
            ), "To save the best model split_name and run_name parameters have to be passed!"

        self.run_name = run_name

        self.split_name = split_name
        self.save_best_model = save_best_model

        self.patience = patience
        self.metric_name = metric_name

        # initialize best result dict
        best_result_dict = {"epoch_best_result": 0, "best_result": -np.inf}
        self.best_result_dict = best_result_dict
        self.es_counter = 0

    def update(self, epoch, metric, model):
        """
        Update EarlyStopping Handler with training results

        Args:
            epoch (int): current training epoch
            metric (float): current value of tracked metric
            model (keras.Model): trained model
        """
        if metric > self.best_result_dict["best_result"]:
            print("New best model found!\n")
            self.best_result_dict["epoch_best_result"] = epoch
            self.best_result_dict["best_result"] = metric
            self.es_counter = 0

            if self.split_name is not None:
                model.save(split_name=self.split_name, run_name=self.run_name)
        else:
            self.es_counter += 1

    def stop_training(self):
        """Function to call to know whether to stop or not the training procedure"""
        return True if self.es_counter == self.patience else False
