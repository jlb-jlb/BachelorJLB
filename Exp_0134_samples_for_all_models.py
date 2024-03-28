# %%
from utils.experiment_utils import prepare_data
import logging
import numpy as np
import pandas as pd
import os
import datetime
import argparse
from tsai.all import computer_setup

computer_setup()


# %%
parameters = {
    "context_length": 1024,
    "prediction_length": 96,
    "test_data_path": "DATA_EEG/TEST_50",
    "batch_size_test": 1,
    "limit_test_batches": 64,
    "max_samples_test": 5000,
    "num_workers": 8,
}
# %%
test_dataset = prepare_data(
    path=parameters["test_data_path"],
    fcst_horizon=parameters["prediction_length"],
    fcst_history=parameters["context_length"],
    local_min_max=None,
    verbose=True,
    max_datasets=None,
    time_embedding=True,
    max_samples=parameters["max_samples_test"],
    test=True,
)
from torch.utils.data import DataLoader

test_dataloader = DataLoader(
    test_dataset,
    batch_size=parameters["batch_size_test"],
    shuffle=False,
    num_workers=parameters["num_workers"],
)
print(len(test_dataloader))
sample = next(iter(test_dataloader))
logging.info(f"Test Dataloader: \t{sample[0].shape}, {sample[1].shape}")
# %%
print(sample[0].shape)

import pickle as pkl

save_dir = "PLOTS/samples"
for i, dl in enumerate(
    [
        DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            num_workers=parameters["num_workers"],
        )
        for dataset in test_dataset.datasets
    ]
):
    batch = next(iter(dl))
    print(batch[0].shape, batch[1].shape)
    # save the batch in save_dir with the name sample_dl_{i}  as pickle file
    batch.append(dict())
    with open(os.path.join(save_dir, f"sample_dl_{i}.pkl"), "wb") as f:
        pkl.dump(batch, f)
