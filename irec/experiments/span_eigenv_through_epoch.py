#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX@XXX"

from pathlib import Path

from irec.experiments.eigendecomposition_dataset import load_eigendecomposition
from irec.losses import l2_reg
from irec.model.bpr.bpr import BPR
from irec.model.lightgcn.lightgcn import LightGCN
from irec.data.implemented_datasets import *
from irec.sampler import TripletsBPRGenerator, logging, set_color
import pandas as pd
import tensorflow as tf
import time
import numpy as np
import os

logger = logging.getLogger(__name__)


def compute_corr(filtered_eig, model):
    num = np.linalg.norm(
        np.dot(np.transpose(filtered_eig[:, -20:]), model().numpy()), ord="fro"
    )
    den = np.linalg.norm(model().numpy(), ord="fro")
    # coeff_norm = np.linalg.norm(sp_coeff, ord=2)
    return num / den


def span_corr_through_epochs(_model):
    # create data sampler
    train_generator = TripletsBPRGenerator(
        dataset=dataset,
        train_data=train,
        batch_size=2048,
        items_after_users_ids=True,
    )

    # build optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    @tf.function
    def _train_on_batch(model, optimizer, inputs):
        with tf.GradientTape() as tape:
            loss = model.train_step(inputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # initialise the models
    norm_per_epoch = []
    # compute correlation random initialisation
    coeff_norm = compute_corr(eig, _model)
    norm_per_epoch.append(coeff_norm)

    for epoch in range(1, EPOCHS):
        cum_loss = 0
        t1 = time.time()
        for _ in range(train_generator.num_batches):
            inputs = train_generator.sample()

            loss = _train_on_batch(_model, optimizer, inputs)

            cum_loss += loss

        cum_loss /= train_generator.num_batches
        log = "Epoch: {:03d}, Loss: {:.4f}, Time: {:.4f}s"
        print(log.format(epoch, cum_loss, time.time() - t1))

        coeff_norm = compute_corr(eig, _model)
        norm_per_epoch.append(coeff_norm)
    return norm_per_epoch


DATASET = "Movielens1M"
K = [1, 2, 3, 4]
EPOCHS = 20
NUM_EIGENV = 20

if __name__ == "__main__":
    dataset = eval(DATASET)()
    split_dict = dataset.load_split(k_core=10, split_name="stratified_0.8_0.1_0.1")
    train, val, test = split_dict["train"], split_dict["val"], split_dict["test"]

    # load eigenvectors
    eig, _ = load_eigendecomposition(dataset=dataset, kind="adjacency")
    # filtered_eig = eig[:, :20]

    norm_dfs = []
    for k in K:
        m = LightGCN(dataset=dataset, train_data=train, embedding_size=64, k=k)
        norm_per_epoch = span_corr_through_epochs(m)

        norm_df = pd.DataFrame(
            zip(norm_per_epoch, np.arange(0, len(norm_per_epoch))),
            columns=["corr", "epoch"],
        )
        norm_df["alg"] = m.name + f"_{k}"
        norm_dfs.append(norm_df)

    m = BPR(dataset=dataset, train_data=train, embedding_size=64)
    bpr_norm_per_epoch = span_corr_through_epochs(m)
    bpr_norm_df = pd.DataFrame(
        zip(bpr_norm_per_epoch, np.arange(0, len(bpr_norm_per_epoch))),
        columns=["corr", "epoch"],
    )
    bpr_norm_df["alg"] = m.name
    norm_dfs.append(bpr_norm_df)

    concat_df = pd.concat(norm_dfs, axis=0).reset_index()

    # create the splits folder
    preprocessed_dir = os.path.join(dataset.data_directory, "preprocessed")
    k_core_dir = os.path.join(preprocessed_dir, f"k_core_{dataset.k_core}")
    Path(k_core_dir).mkdir(exist_ok=True)
    save_name = os.path.join(k_core_dir, "span_eig_through_epoch.csv")
    concat_df.to_csv(save_name, index=False)

    logger.info(
        set_color(
            f"Eigenvalues and eigenvectors saved in {save_name}",
            "white",
        )
    )
