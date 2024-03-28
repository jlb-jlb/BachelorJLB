parameters = {
    "samples_path": "PLOTS/samples",
    "transformer": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGTransformer__336_96__2024_03_13_09_28/checkpoints/best_model_Exp_EEGTransformer__336_96-v1.ckpt",
    "informer": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGInformer__336_96__2024_03_12_18_16/checkpoints/best_model_Exp_EEGInformer__336_96.ckpt",
    "autoformer": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGAutoformer__336_96__2024_03_12_23_56/checkpoints/best_model_Exp_EEGAutoformer__336_96-v1.ckpt",
    "patchTST_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__336_96__2024_03_14_23_15/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__336_96-v2.ckpt",
    "patchTST_512": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__512_96__2024_03_14_23_16/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__512_96-v2.ckpt",
    "patchTST_720": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__720_96__2024_03_15_14_10/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__720_96-v1.ckpt",
    "patchTST_1024": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__1024_96__2024_03_15_13_24/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__1024_96.ckpt",
    "TSMixer_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__336_96__2024_03_11_20_50/checkpoints/best_model_EXP_EEGpatchTSMixer__336_96-v1.ckpt",
    "TSMixer_512": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__512_96__2024_03_11_20_51/checkpoints/best_model_EXP_EEGpatchTSMixer__512_96-v2.ckpt",
    "TSMixer_720": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__720_96__2024_03_17_20_10/checkpoints/best_model_EXP_EEGpatchTSMixer__720_96.ckpt",
    "TSMixer_1024": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixerMIN3__1024_96__2024_03_17_20_11/checkpoints/best_model_EXP_EEGpatchTSMixerMIN3__1024_96.ckpt",
    "experiment_path": "PLOTS/predictions",
}

# for transformer, autoformer, informer (336)
from utils.lightning_configs import LitModelTimeEmbed, LitModel
from utils.experiment_utils import scaler_local_min_max
from lightning.pytorch import Trainer, seed_everything

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

samples = os.listdir(parameters["samples_path"])
samples.sort()


seed_everything(42, workers=True)
os.makedirs(parameters["experiment_path"], exist_ok=True)
os.makedirs(parameters["samples_path"], exist_ok=True)
models = [
    "transformer",
    "autoformer",
    "informer",
    "patchTST_336",
    "TSMixer_336",
    "patchTST_512",
    "TSMixer_512",
    "patchTST_720",
    "TSMixer_720",
    "patchTST_1024",
    "TSMixer_1024",
]


# check if cuda available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for model in models:
    # check if model transformer, autoformer, or informer
    if model in ["transformer", "autoformer", "informer"]:
        LitM = LitModelTimeEmbed
    else:
        LitM = LitModel
    best_model = LitM.load_from_checkpoint(checkpoint_path=parameters[model])
    best_model.eval()
    for idx, sample in enumerate(samples):
        with open(f"{parameters['samples_path']}/{sample}", "rb") as f:
            data = pickle.load(f)
        print(data)
        print(len(data))

        if model in ["transformer", "autoformer", "informer"]:
            batch = [
                item[:, -(336 + 20) :, :] if item.shape[1] == 1031 else item
                for item in data[:-1]
            ]
            batch[0], batch[1] = scaler_local_min_max(batch[0], batch[1])
            for item in batch:
                print(item.shape)
            print(idx)
            predictions = best_model.pred_for_plot(batch)
            forecasts = np.median(predictions, 1)
            print(forecasts.shape)
            data[-1][model] = forecasts
        else:
            context_length = int(model.plit("_")[-1])
            batch = [
                item[:, -(context_length):, :] if item.shape[1] == 1031 else item
                for item in data[:-1]
            ]
            batch[0], batch[1] = scaler_local_min_max(batch[0], batch[1])
            for item in batch:
                print(item.shape)
            predictions = best_model.pred_for_plot(
                (batch[0].to(device), batch[1].to(device))
            )
            data[-1][model] = predictions

        with open(f"{parameters['samples_path']}/{sample}", "wb") as f:
            pickle.dump(data, f)
