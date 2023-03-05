#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from abc import ABC, abstractmethod
import time
import logging
import wandb
from irec.evaluation.topk_evaluator import Evaluator
from irec.losses import l2_reg
from irec.utils.utils import set_color
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AbstractTrainer(ABC):
    """Abstract class to train recommender systems model
    Attributes:
        model: recommender system model
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def fit(self):
        pass


class DummyTrainer(AbstractTrainer):
    """Dummy trainer for algorithm which doesn't require a learning process

    Attributes:
        model: recommender system model
        val_evaluator: evaluator used to test model performance on validation data
    """

    def __init__(self, model, val_evaluator=None, wandb_log=False):
        super().__init__(model=model)
        self.model = model
        self.val_evaluator = val_evaluator
        self.wandb_log = wandb_log

    def fit(self):
        if self.val_evaluator is not None:
            self.val_evaluator.evaluate_recommender(
                recommender=self.model, interactions=self.model.train_data
            )
            self.val_evaluator.print_evaluation_results()
            if self.wandb_log:
                res_dict = self.val_evaluator.result_dict
                wandb.log(res_dict)


class RepresentationsBasedRecommenderTrainer(AbstractTrainer):
    """Dummy trainer for algorithm which doesn't require a learning process

    Attributes:
        model: recommender system model
        val_evaluator: evaluator used to test model performance on validation data
    """

    def __init__(self, model, val_evaluator=None, wandb_log=False):
        super().__init__(model=model)
        self.model = model
        self.val_evaluator = val_evaluator
        self.wandb_log = wandb_log

    def fit(self):
        self.model.compute_similarity_matrix()
        if self.val_evaluator is not None:
            self.val_evaluator.evaluate_recommender(
                recommender=self.model, interactions=self.model.train_data
            )
            self.val_evaluator.print_evaluation_results()
        if self.wandb_log:
            res_dict = self.val_evaluator.result_dict
            wandb.log(res_dict)


class TensorflowTrainer(AbstractTrainer):
    """Trainer for TensorflowRecommender models

    Attributes:
        model: recommender system model
        train_generator: train data generator
        epochs (int): number of training epochs
        l2_reg (float): l2 used for regularization
        learning_rate (float): learning rate used for optimization
        val_evaluator: evaluator used to test model performance on validation data
        val_every: frequency in epoch of the evaluation on validation data
        test_evaluator: evaluator used to test model performance on test data
        early_stopping_handler: handler used for early stopping
        wandb_log (bool): if log the training on wandb
    """

    def __init__(
        self,
        model,
        train_generator,
        epochs,
        l2_reg=0.0,
        learning_rate=1e-3,
        val_evaluator=None,
        val_every=None,
        early_stopping_handler=None,
        wandb_log=False,
    ):
        AbstractTrainer.__init__(self, model=model)

        self.es_handler = early_stopping_handler
        self.train_generator = train_generator
        self.epochs = epochs
        self.val_evaluator = val_evaluator
        self.val_every = val_every
        self.wandb_log = wandb_log

        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.validation = True if val_evaluator is not None else False
        self.early_stopping = True if early_stopping_handler is not None else False

    @tf.function
    def _train_on_batch(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.model.train_step(inputs)
            loss += l2_reg(self.model, alpha=self.l2_reg)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self):
        for epoch in range(1, self.epochs):
            cum_loss = 0
            t1 = time.time()
            for _ in range(self.train_generator.num_batches):
                inputs = self.train_generator.sample()

                loss = self._train_on_batch(inputs)

                cum_loss += loss

            cum_loss /= self.train_generator.num_batches
            log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
            print(set_color(log.format(epoch, cum_loss, time.time() - t1), "red"))

            # validate the model
            if self.validation and epoch % self.val_every == 0:
                self.val_evaluator.evaluate_recommender(
                    recommender=self.model, interactions=self.model.train_data
                )
                self.val_evaluator.print_evaluation_results()

                if self.wandb_log:
                    res_dict = self.val_evaluator.result_dict
                    wandb.log(res_dict, step=epoch)

                if self.early_stopping:
                    es_metric = self.val_evaluator.result_dict[
                        self.es_handler.metric_name
                    ]
                    self.es_handler.update(epoch, es_metric, self.model)
                    if self.es_handler.stop_training():
                        break
        if self.wandb_log and self.early_stopping:
            wandb.log(self.es_handler.best_result_dict)
