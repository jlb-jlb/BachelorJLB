import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "visuals_path": "PLOTS/visuals",
    "smape_vis_path": "PLOTS/smape_visuals_336",
    "colors_336": ["#ffc857", "#e9724c", "#c5283d", "#036666", "#67B99A"],
    "colors": ["#036666", "#1C7B73", "#358f80", "#67b99a", "#99e2b4"],
}

# %%

from utils.visualization_utils import plot_all

samples = os.listdir(parameters["samples_path"])
samples.sort()

histories = [336, 512, 720, 1024]
for hist in histories:
    if hist == 336:
        selected_predictions = [
            "transformer",
            "informer",
            "autoformer",
            "patchTST_336",
            "TSMixer_336",
        ]
        colors = ["#ffc857", "#e9724c", "#c5283d", "#036666", "#67B99A"]
    else:
        selected_predictions = [f"patchTST_{hist}", f"TSMixer_{hist}"]
        colors = ["#036666", "#67B99A"]
    #
    print(selected_predictions)
    parameters["visuals_path"] = f"PLOTS/visuals_{hist}"
    # make dir, if exists is okay
    if not os.path.exists(parameters["visuals_path"]):
        os.makedirs(parameters["visuals_path"], exist_ok=True)
    #
    plot_all(samples, parameters, colors, hist, selected_predictions)


# %%
# plot the smape and mae for each model considering the accuracy with increased prediction length. i.e. calculate the smape starting with one timestep in the future and ending with all 96 timesteps
# The result should be an array of shape (samples, 96, channels) put in a dict for all models

# Calculate SMAPE and MAE for each model and prediction length
from utils.experiment_utils import scaler_local_min_max
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError


selected_predictions = [
    "transformer",
    "informer",
    "autoformer",
    "patchTST_336",
    "TSMixer_336",
]
history_len = 336
smape_results = {}
mae_results = {}
for sample in samples:
    with open(f"{parameters['samples_path']}/{sample}", "rb") as f:
        data = pickle.load(f)
    #
    smape_results[sample] = {}
    mae_results[sample] = {}
    #
    past_values = data[0][:, -history_len:, :]
    future_values = data[1]
    past_values, future_values = scaler_local_min_max(past_values, future_values)
    predictions = data[-1]
    for model in selected_predictions:
        #
        smape_results[sample][model] = []
        mae_results[sample][model] = []
        #
        model_predictions = predictions[model]
        for prediction_length in range(1, 97):
            model_prediction_with_pred_len = torch.from_numpy(
                model_predictions[:, :prediction_length, :]
            )
            fut_vals_with_pred_len = future_values[:, :prediction_length, :]
            smape_value = SymmetricMeanAbsolutePercentageError()(
                model_prediction_with_pred_len, fut_vals_with_pred_len
            )
            mae_value = torch.nn.L1Loss()(
                model_prediction_with_pred_len, fut_vals_with_pred_len
            )
            #
            smape_results[sample][model].append(smape_value.item())
            mae_results[sample][model].append(mae_value.item())
            #
        print(
            f"Model: {model}, Prediction Length: {prediction_length}, \nSMAPE: {smape_results[sample][model]}, \nMAE: {mae_results[sample][model]}"
        )

# %%
for idx, sample in enumerate(smape_results.keys()):
    print(idx)
    smapes = smape_results[sample]
    # print(smapes.keys())
    # break
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    # Create a figure object
    for row, model in enumerate(smapes.keys()):
        axs.plot(
            range(1, 97),
            smapes[model],
            color=parameters["colors_336"][row],
            linewidth=2,
        )
        axs.set_xlabel("Prediction Length", fontsize=20)
        axs.tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()

    if not "17" in sample and not "26" in sample:
        plt.legend(smapes.keys(), fontsize=24)
    #
    plt.savefig(f"{parameters['smape_vis_path']}/{sample}_smape.png")
    # delete the plot
    plt.close()

# %%
# ALL SMAPES

import numpy as np

smape_models = {}

for idx, sample in enumerate(smape_results.keys()):
    print(idx)
    smapes = smape_results[sample]
    for model in smapes.keys():
        if model not in smape_models.keys():
            smape_models[model] = []
        smape_models[model].append(smapes[model])


avg_smape_models = {}
for model in smape_models.keys():
    avg_smape_models[model] = np.mean(smape_models[model], axis=0)


# %%
# plot all 5 average smapes
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
# Create a figure object
for row, model in enumerate(avg_smape_models.keys()):
    axs.plot(range(1, 97), avg_smape_models[model], color=parameters["colors_336"][row])
    # axs.set_title(f"{model}")
    axs.set_xlabel(
        "Prediction Length",
    )
    axs.set_ylabel("SMAPE")
    axs.set_title("Average SMAPE for all models")

    plt.tight_layout()
    plt.legend(smapes.keys())
    plt.savefig(f"{parameters['smape_vis_path']}/AVERAGE_SMAPE.png")


# smape and mae results for TSMIXER AND PATCHTST with increased context lengths
selected_predictions = ["patchTST_", "TSMixer_"]
history_lengths = [336, 512, 720, 1024]
smape_results = {}
mae_results = {}

from itertools import product


for sample in samples:
    with open(f"{parameters['samples_path']}/{sample}", "rb") as f:
        data = pickle.load(f)
    #
    smape_results[sample] = {}
    mae_results[sample] = {}
    for history_len in history_lengths:
        #
        past_values = data[0][:, -history_len:, :]
        future_values = data[1]
        # Scale the data
        shape_0 = past_values.shape
        shape_1 = future_values.shape
        start_index_y = shape_0[1]
        all_data = np.concatenate((past_values, future_values), axis=1)
        data_flattened = all_data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data_flattened).reshape(all_data.shape)
        past_values = torch.from_numpy(scaled_data[:, :start_index_y, :])
        future_values = torch.from_numpy(scaled_data[:, start_index_y:, :])
        #
        predictions = data[-1]
        for model_ in selected_predictions:
            model = f"{model_}{history_len}"
            #
            smape_results[sample][model] = []
            mae_results[sample][model] = []
            #
            model_predictions = predictions[model]
            for prediction_length in range(1, 97):
                model_prediction_with_pred_len = torch.from_numpy(
                    model_predictions[:, :prediction_length, :]
                )
                fut_vals_with_pred_len = future_values[:, :prediction_length, :]
                smape_value = SymmetricMeanAbsolutePercentageError()(
                    model_prediction_with_pred_len, fut_vals_with_pred_len
                )
                mae_value = torch.nn.L1Loss()(
                    model_prediction_with_pred_len, fut_vals_with_pred_len
                )
                #
                smape_results[sample][model].append(smape_value.item())
                mae_results[sample][model].append(mae_value.item())
                #
            print(
                f"Model: {model}, Prediction Length: {prediction_length}, \nSMAPE: {smape_results[sample][model]}, \nMAE: {mae_results[sample][model]}"
            )
# %%
import numpy as np

smape_models = {}

for idx, sample in enumerate(smape_results.keys()):
    print(idx)
    smapes = smape_results[sample]
    for model in smapes.keys():
        if model not in smape_models.keys():
            smape_models[model] = []
        smape_models[model].append(smapes[model])


avg_smape_models = {}
for model in smape_models.keys():
    avg_smape_models[model] = np.mean(smape_models[model], axis=0)
# plot all 8 average smapes
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
# Create a figure object
for row, model in enumerate(avg_smape_models.keys()):
    axs.plot(range(1, 97), avg_smape_models[model], color=parameters["colors_336"][row])
    # axs.set_title(f"{model}")
    axs.set_xlabel(
        "Prediction Length",
    )
    axs.set_ylabel("SMAPE")
    axs.set_title("Average SMAPE for all models")

    plt.tight_layout()
    plt.legend(smapes.keys())
    plt.savefig(f"{parameters['smape_vis_path']}/336_1024_AVERAGE_SMAPE.png")

# plot the average smapes x prediction lengths for patchTST and TSMixer. Total of 2 plots one for all context lengths of the corresponding mdoel
# plot the average smapes x prediction lengths for patchTST and TSMixer

# %%
import json

with open("PLOTS/smapes_pred_len/smapes_per_model.json", "r") as f:
    avg_smape_models = json.load(f)

print(avg_smape_models.keys())

# %%

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# Plot for patchTST
patchTST_smapes = {
    key: avg_smape_models[key]
    for key in avg_smape_models.keys()
    if "patchtst" in key.lower()
}
TSMixer_smapes = {
    key: avg_smape_models[key]
    for key in avg_smape_models.keys()
    if "tsmixer" in key.lower()
}

for i, history_len in enumerate(history_lengths):
    axs[0].plot(
        range(1, 96),
        patchTST_smapes[f"PatchTST_{history_len}"],
        color=parameters["colors"][i],
    )
axs[0].set_xlabel("Prediction Length")
axs[0].set_ylabel("SMAPE")
axs[0].set_title("PatchTST")
axs[0].legend(history_lengths)
# grid
axs[0].grid(color="lightgrey", linestyle="--")

# Plot for TSMixer

for i, history_len in enumerate(history_lengths):
    axs[1].plot(
        range(1, 96),
        TSMixer_smapes[f"TSMixer_{history_len}"],
        color=parameters["colors"][i],
    )
axs[1].set_xlabel("Prediction Length")
axs[1].set_ylabel("SMAPE")
axs[1].set_title("TSMixer")
axs[1].legend(history_lengths)
# grid
axs[1].grid(color="lightgrey", linestyle="--")

plt.tight_layout()
plt.show()
plt.savefig(f"{parameters['smape_vis_path']}/patchTST_TSMixer_AVERAGE_SMAPE.png")


# plot
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

history_len = 336


parameters = {
    "26_1": "PLOTS/samples/sample_dl_26.pkl",
    "selected_predictions": ["autoformer"],
}

with open(parameters["26_1"], "rb") as f:
    data = pickle.load(f)

# %%
past_values = data[0][:, -history_len:, :]
future_values = data[1]
# scale the data
shape_0 = past_values.shape
shape_1 = future_values.shape
start_index_y = shape_0[1]
all_data = np.concatenate((past_values, future_values), axis=1)
data_flattened = all_data.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data_flattened).reshape(all_data.shape)
past_values = scaled_data[:, :start_index_y, :]
future_values = scaled_data[:, start_index_y:, :]
predictions = data[-1]
sub_predictions = {key: predictions[key] for key in parameters["selected_predictions"]}

# %%
import matplotlib.pyplot as plt

channel_1_past = past_values[0, :, 0]
channel_1_future = future_values[0, :, 0]
index_for_plot = np.arange(-history_len, 96) * 20

# concatenate past and future array
all_values = np.concatenate((channel_1_past, channel_1_future))
# mean of all values
mean_all_values = np.mean(all_values)

window = 25
# Calculate simple moving average
all_data_df = pd.DataFrame({"ch1": all_values})
all_data_df["SMA"] = all_data_df["ch1"].rolling(window=window).mean()
print(all_data_df["SMA"].shape)
# calculate the Exponential Moving Average (EMA) with smoothing factor 0.5
all_data_df["EMA"] = all_data_df["ch1"].ewm(span=35, adjust=False).mean()
# calculate the Cumulative Moving Average
all_data_df["CMA"] = all_data_df["ch1"].expanding().mean()
print(f"CMA: {all_data_df['CMA'].shape}")

# calculate the CMA, starting at datapoint 150

cma_xx_len = 96 + 336

cma_xx = all_data_df["ch1"][-(cma_xx_len):].expanding().mean()

# standard deviation
all_data_df["std"] = all_data_df["ch1"].std()
# variance
all_data_df["var"] = all_data_df["ch1"].var()

cutoff_past = 100

################################
plt.figure(figsize=(17, 4))


plt.plot(
    index_for_plot[-(cutoff_past + 96) : -96],
    channel_1_past[-cutoff_past:],
    color="grey",
    label="Past & Future",
)
last_point = channel_1_past[-1]
plt.plot(
    index_for_plot[-97:],
    np.insert(channel_1_future, 0, last_point),
    color="grey",
)
# plot the prediction from "autoformer"
plt.plot(
    index_for_plot[-96:],
    predictions["autoformer"][0, -cutoff_past:, 0],
    color="#c5283d",
    label="Autoformer",
)
# plot the moving average
plt.plot(
    index_for_plot[-(cutoff_past + 96) :],
    all_data_df["SMA"][-(cutoff_past + 96) :],
    color="#1C7B73",
    label=f"Moving Average, window={window}",
)
# plot the Exponential Moving Average
# plt.plot(index_for_plot[-(cutoff_past+96):], all_data_df["EMA"][-(cutoff_past+96):], label="EMA")
# plt.fill_between(index_for_plot[-(cutoff_past+96):], all_data_df["SMA"][-(cutoff_past+96):] - all_data_df["std"][-(cutoff_past+96):], all_data_df["SMA"][-(cutoff_past+96):] + all_data_df["std"][-(cutoff_past+96):], color='b', alpha=0.2)
# plot the std deviation from the mean_all_values
# plt.fill_between(index_for_plot[-(cutoff_past+96):], mean_all_values+all_data_df["std"][-(cutoff_past+96):], mean_all_values-all_data_df["std"][-(cutoff_past+96):], color='b', alpha=0.2)
# plot the variance
# plt.fill_between(index_for_plot[-(cutoff_past+96):], mean_all_values+all_data_df["var"][-(cutoff_past+96):], mean_all_values-all_data_df["var"][-(cutoff_past+96):], color='r', alpha=0.2)
# plot cma
# plt.plot(index_for_plot[-(cutoff_past+96):], all_data_df["CMA"][-(cutoff_past+96):],color="#036666",linestyle='--', label="Comulative Moving Average")


# plot cma_xx
plt.plot(
    index_for_plot[-(cma_xx_len):],
    cma_xx,
    color="#036666",
    linestyle="--",
    label="CMA_xx",
)

plt.plot()

# mean line on plot
# name y axis scaled amplitude
plt.ylabel("Scaled Amplitude")
plt.xlabel("Time in ms")

# grid
plt.grid(color="lightgrey", linestyle="--")
plt.axhline(y=mean_all_values, color="#036666", linestyle="--", label="Mean Total")
plt.legend()
plt.show()

# clear plots
plt.close()


######################################################
# %%
plt.figure(figsize=(15, 8))
plt.plot(all_values)
plt.show()
plt.close()
