# %%
from utils.experiment_utils import prepare_data
from torch.utils.data import DataLoader
import logging
import numpy as np
import pandas as pd
import os
import datetime
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from tsai.all import computer_setup
from lightning.pytorch import Trainer, seed_everything

computer_setup()

# parameters for testing/dev
parameters = {
    "experiment_dir": "EXPERIMENTS/TESTS",  # output directory for the experiment
    "experiment_model": "EEG_PatchTST",  # output subdirectory for the experiment (for the specific model)
    "model_name": "EEG_PatchTST",
    "datetime": datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"),
    "freq": "20L",
    "prediction_length": 96,
    "context_length": 336,
    "train_data_path": "DATA_EEG/EEG_PREP_ROBUST_TRAIN",
    "test_data_path": "DATA_EEG/TEST_50",
    "val_data_path": "DATA_EEG/VAL_70",
    "batch_size_train": 32,
    "batch_size_test": 1,
    "batch_size_val": 32,
    "fast_dev_run": False,
    "early_stopping": 15,
    "val_check_interval": 50,
    "limit_val_batches": 100,  # must be within the maximum NO USE IN INFORMER
    "limit_test_batches": 32,  # PatchTST
    "limit_train_batches": 40000,
    "max_epochs": 5,
    # "max_val_files": 10,
    # "max_test_files": 10,
    # "max_train_files": 50,
    # "num_batches_per_epoch_train": 100, # this number of batches * number of dataloaders(train)
    # "num_batches_per_epoch_val": 10, # number of batches * number of dataloaders(val)
    # "num_batches_per_epoch_test": 20, # number of batches * number of dataloaders(test)
    "max_samples_test": 5000,
    "max_samples_val": 9600,
    "num_workers": 8,
    "lags_sequence": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
    ],  # lags by max auto-correlation
    "matmul_precision": "medium",
}

import argparse

parser = argparse.ArgumentParser(description="Runs the  model with certain parameters")
parser.add_argument(
    "--experiment_dir",
    type=str,
    default=parameters["experiment_dir"],
    help="Directory to save the experiment results",
)
parser.add_argument(
    "--experiment_model",
    type=str,
    default=parameters["experiment_model"],
    help="Subdirectory to save the experiment results",
)
parser.add_argument(
    "--model_name", type=str, default=parameters["model_name"], help="Name of the model"
)
parser.add_argument(
    "--prediction_length",
    type=int,
    default=parameters["prediction_length"],
    help="Prediction length of the model",
)
parser.add_argument(
    "--context_length",
    type=int,
    default=parameters["context_length"],
    help="Context length of the model",
)
# data paths
parser.add_argument(
    "--train_data_path",
    type=str,
    default=parameters["train_data_path"],
    help="Path to the training data",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default=parameters["test_data_path"],
    help="Path to the test data",
)
parser.add_argument(
    "--val_data_path",
    type=str,
    default=parameters["val_data_path"],
    help="Path to the validation data",
)
# batch sizes
parser.add_argument(
    "--batch_size_train",
    type=int,
    default=parameters["batch_size_train"],
    help="Batch size for training",
)
parser.add_argument(
    "--batch_size_test",
    type=int,
    default=parameters["batch_size_test"],
    help="Batch size for testing",
)
parser.add_argument(
    "--batch_size_val",
    type=int,
    default=parameters["batch_size_val"],
    help="Batch size for validation",
)
# fast dev run for testing#
parser.add_argument(
    "--fast_dev_run",
    type=bool,
    default=parameters["fast_dev_run"],
    help="Run the model on a small dataset",
)

# early stopping and check interval
parser.add_argument(
    "--early_stopping",
    type=int,
    default=parameters["early_stopping"],
    help="Early stopping parameter",
)
parser.add_argument(
    "--val_check_interval",
    type=int,
    default=parameters["val_check_interval"],
    help="Validation check interval",
)
# limit max number of batches --> for timeseries useful. USED IN THE LIGHTNING TRAINER
parser.add_argument(
    "--limit_val_batches",
    type=int,
    default=parameters["limit_val_batches"],
    help="Limit validation batches",
)
parser.add_argument(
    "--limit_test_batches",
    type=int,
    default=parameters["limit_test_batches"],
    help="Limit test batches",
)
parser.add_argument(
    "--limit_train_batches",
    type=float,
    default=parameters["limit_train_batches"],
    help="Limit train batches",
)
# limit max number of epochs
parser.add_argument(
    "--max_epochs",
    type=int,
    default=parameters["max_epochs"],
    help="Maximum number of epochs",
)

# limit max number of samples. USED IN THE DATASET LOADER
parser.add_argument(
    "--max_samples_test",
    type=int,
    default=parameters["max_samples_test"],
    help="Maximum number of samples to use!",
)
parser.add_argument(
    "--max_samples_val",
    type=int,
    default=parameters["max_samples_val"],
    help="Maximum number of samples to use (val)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=parameters["num_workers"],
    help="Number of workers for the dataloaders",
)
args = parser.parse_args()
parameters.update(vars(args))


# %%
# create the experiment directory
if not os.path.exists(parameters["experiment_dir"]):
    os.makedirs(parameters["experiment_dir"])

parameters["experiment_path"] = (
    f"{parameters['experiment_dir']}/{parameters['experiment_model']}__{parameters['context_length']}_{parameters['prediction_length']}__{parameters['datetime']}"
)
if not os.path.exists(parameters["experiment_path"]):
    os.makedirs(parameters["experiment_path"])

if not os.path.exists(f"{parameters['experiment_path']}/lightning_logs"):
    os.makedirs(f"{parameters['experiment_path']}/lightning_logs")

# set up the logger
logging.basicConfig(
    level=logging.INFO,
    force=True,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"{parameters['experiment_path']}/{parameters['experiment_model']}_{parameters['context_length']}_{parameters['prediction_length']}.log",
)
logging.info(f"START OF EXPERIMENT")
logging.info(f"Parameters: {parameters}")
logging.info(
    f"EXPERIMENT {parameters['context_length']} - {parameters['prediction_length']}"
)
print(f"EXPERIMENT {parameters['context_length']} - {parameters['prediction_length']}")
################################################################################################

######   ##   ##   #####         #####    ######  ########  ##   ##  ######
##       ###  ##   ##   #       #         ##         ##     ##   ##  ##    #
####     ## # ##   ##   #        #####    ####       ##     ##   ##  ######
##       ##  ###   ##   #             #   ##         ##     ##   ##  ##
######   ##   ##   #####         #####    ######     ##      #####   ##

###############################################################################################
# %%
######################################################################################

#####       ########    ########    ########
##   ##     ##    ##       ##       ##    ##
##   ##     ########       ##       ########
##   ##     ##    ##       ##       ##    ##
#####       ##    ##       ##       ##    ##

######################################################################################
models_no_timeembedding = ["tsmixer", "patchtst"]
models_with_timeembedding = ["transformer", "informer", "autoformer"]
if any(
    [model in parameters["model_name"].lower() for model in models_no_timeembedding]
):
    # load data without time embeddings
    logging.info(f"Model {parameters['model_name']} with no time embeddings")
    logging.info(f"Loading data")
    training_dataset = prepare_data(
        path=parameters["train_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
    )

    test_dataset = prepare_data(
        path=parameters["test_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
        max_samples=parameters["max_samples_test"],
        test=True,
    )

    val_dataset = prepare_data(
        path=parameters["val_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
        max_samples=parameters["max_samples_val"],
        test=True,
    )
elif any(
    [model in parameters["model_name"].lower() for model in models_with_timeembedding]
):
    # load data with time embeddings
    logging.info(f"Model {parameters['model_name']} with time embeddings")
    logging.info(f"Loading data")
    training_dataset = prepare_data(
        path=parameters["train_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
        lags_sequence=parameters["lags_sequence"],
        time_embedding=True,
    )

    test_dataset = prepare_data(
        path=parameters["test_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
        max_samples=parameters["max_samples_test"],
        test=True,  # test/val
        time_embedding=True,
        lags_sequence=parameters["lags_sequence"],
    )

    val_dataset = prepare_data(
        path=parameters["val_data_path"],
        fcst_horizon=parameters["prediction_length"],
        fcst_history=parameters["context_length"],
        local_min_max="scale_all",
        verbose=True,
        max_datasets=None,
        max_samples=parameters["max_samples_val"],
        test=True,  # test/val
        time_embedding=True,
        lags_sequence=parameters["lags_sequence"],
    )
else:
    logging.error(f"Model {parameters['model_name']} not found.")
    logging.error(
        f"Available models (Name must include the word from one of these models): {models_no_timeembedding + models_with_timeembedding}"
    )
    raise ValueError("Model not found.")
# DataLoader

train_dataloader = DataLoader(
    training_dataset,
    batch_size=parameters["batch_size_train"],
    shuffle=True,
    num_workers=parameters["num_workers"],
)
sample = next(iter(train_dataloader))
logging.info(f"Train Dataloader\t{sample[0].shape}, {sample[1].shape}")
print(len(train_dataloader))

val_dataloader = DataLoader(
    val_dataset,
    batch_size=parameters["batch_size_val"],
    shuffle=False,
    num_workers=parameters["num_workers"],
)
sample = next(iter(val_dataloader))
logging.info(f"Val Dataloader: \t{sample[0].shape}, {sample[1].shape}")
print(len(val_dataloader))

test_dataloader = DataLoader(
    test_dataset,
    batch_size=parameters["batch_size_test"],
    shuffle=False,
    num_workers=parameters["num_workers"],
)
print(len(test_dataloader))
sample = next(iter(test_dataloader))
logging.info(f"Test Dataloader: \t{sample[0].shape}, {sample[1].shape}")
for s in sample:
    print(s.shape, s.dtype)

############################################################

#####  #####   ######   #####     ########   ##
##   ##   ##  ##    ##  ##   ##   ##         ##
##   ##   ##  ##    ##  ##   ##   #####      ##
##        ##  ##    ##  ##   ##   #####      ##
##        ##  ##    ##  #    ##   ##         ##
##        ##   ######   #####     ########   ########

############################################################
from utils.model_configs import get_model

model = get_model(parameters["model_name"], parameters=parameters)
print(model.config)
print(model)
logging.info(f"Model config: {model.config}")
logging.info(f"Model: {model}")

#############################################################################

##          ##   #######    ##     ##   ##########     ##      ##
##          ##  ##      #   ##     ##       ##         ####    ##
##          ##  ##          #########       ##         ##  ##  ##
##          ##  ##    ###   ##     ##       ##         ##    ####
##          ##  ##     ##   ##     ##       ##         ##      ##
#########   ##   #######    ##     ##       ##         ##      ##

#############################################################################
if any(
    [model in parameters["model_name"].lower() for model in models_no_timeembedding]
):
    logging.info("Loading Lightning Module with no time embeddings")
    from utils.lightning_configs import LitModel

    model_lit = LitModel(model)
elif any(
    [model in parameters["model_name"].lower() for model in models_with_timeembedding]
):
    logging.info("Loading Lightning Module with time embeddings")
    from utils.lightning_configs import LitModelTimeEmbed as LitModel

    model_lit = LitModel(model)
else:
    raise ValueError("Model not found")


from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# TRAINER
seed_everything(42, workers=True)
trainer = Trainer(
    default_root_dir=f"{parameters['experiment_path']}",
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min", patience=parameters["early_stopping"]
        ),  # early stopping
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            dirpath=f"{parameters['experiment_path']}/checkpoints",
            filename=f"best_model_{parameters['experiment_model']}__{parameters['context_length']}_{parameters['prediction_length']}",
        ),
    ],  # save the best model
    val_check_interval=parameters[
        "val_check_interval"
    ],  # check validation after XXX training batches
    fast_dev_run=parameters[
        "fast_dev_run"
    ],  # run through the training loop quickly to check if there are any bugs
    # can be an int to how many batches to run through
    limit_train_batches=parameters[
        "limit_train_batches"
    ],  # parameters["limit_train_batches"], # limit the number of training batches 0-1
    # limit_val_batches=parameters["limit_val_batches"], # limit the number of validation batches 0-1
    limit_test_batches=parameters[
        "limit_test_batches"
    ],  # 32 limit the number of test batches 0-1
    num_sanity_val_steps=4,  # checks before training that validation is working
    max_epochs=parameters[
        "max_epochs"
    ],  # maximum number of epochs to train if not stopped by early stopping
    min_epochs=3,
)


#############################################################################

##########  ########        ##########      ##    ##      ##
##      ##      ##      ##      ##      ##    ####    ##
##      ##      ##      ##      ##      ##    ##  ##  ##
##      ########        ##########      ##    ##    ####
##      ##     ##       ##      ##      ##    ##      ##
##      ##      ##      ##      ##      ##    ##      ##

#############################################################################

torch.set_float32_matmul_precision(parameters["matmul_precision"])
logging.info(f"Matmul precision: {parameters['matmul_precision']}")
# %%
# TRAIN
print(parameters["experiment_path"])
trainer.fit(
    model=model_lit,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


# %%
logging.info(f"BEST MODEL: {trainer.checkpoint_callback.best_model_path}")
print(f"BEST MODEL: {trainer.checkpoint_callback.best_model_path}")
best_model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

# %%
# TEST on all TEST Recordings
trainer.test(
    model=best_model,
    dataloaders=[
        DataLoader(
            dataset=dataset,
            batch_size=parameters["batch_size_test"],
            shuffle=True,
            num_workers=parameters["num_workers"],
        )
        for dataset in test_dataset.datasets
    ],
)


# %%
logging.info(
    f"Experiment exited successfully. Check the logs and the tensorboard for more information."
)
logging.info(f"END OF EXPERIMENT")
print(
    "Experiment exited successfully. Check the logs and the tensorboard for more information."
)
print("END OF EXPERIMENT")
