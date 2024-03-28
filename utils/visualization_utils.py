import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError


def plototot(
    past_values,
    future_values,
    predictions: dict,
    past_length=96,
    models=["transformer", "informer", "autoformer", "patchTST_336", "TSMixer_336"],
    colors=[
        "#036666",
        "#1C7B73",
        "#358f80",
        "#67b99a",
        "#99e2b4",
    ],  # Colors for the models' plots
    max_num_channels=22,
    figsize=(20, 120),
    suptitle="",
    show=True,
    save_path=None,
    predictions_index=2,
    label_y_axis="Scaled Amplitude",
    label_x_axis="Time in ms",
    actual_values_color="grey",
    grid_color="lightgrey",
    index_freq="20L",
    index_multiplier=20,
):
    # Plotting setup
    mae_predictions = {}
    smape_predictions = {}
    num_channels = past_values.shape[2]  # Number of channels
    if num_channels > max_num_channels:
        num_channels = max_num_channels  # Limit the number of channels to plot
    for model_name in predictions.keys():
        prediction_data = predictions[model_name]
        mae = F.l1_loss(torch.tensor(prediction_data), torch.tensor(future_values))
        mae_predictions[model_name] = mae
        smape = SymmetricMeanAbsolutePercentageError()(
            torch.tensor(prediction_data), torch.tensor(future_values)
        )
        smape_predictions[model_name] = smape
    print(mae_predictions)
    print(smape_predictions)
    # Generate plots for each channel
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle, fontsize=24)
    for channel_index in range(num_channels):
        plt.subplot(num_channels, 1, channel_index + 1)
        # Select the last 96 time steps from the past values and all future values for the current channel
        past_values_channel = past_values[0, -past_length:, channel_index]
        last_past_value = np.array([past_values[0, -1, channel_index]])
        future_values_channel = future_values[0, :, channel_index]
        time_index_past = np.arange(-(past_length - 1), 1) * index_multiplier
        time_index_future = (
            np.arange(0, future_values_channel.shape[0] + 1) * index_multiplier
        )
        # Plot past and future values
        plt.plot(
            time_index_past,
            past_values_channel,
            label="Past and Future Values",
            color=actual_values_color,
        )
        plt.plot(
            time_index_future,
            np.concatenate(([past_values_channel[-1]], future_values_channel)),
            color=actual_values_color,
        )
        # Plot the mean predictions for each model
        model_index = 0
        for model_name in predictions.keys():
            prediction_data = predictions[model_name][0, :, channel_index]
            plt.plot(
                time_index_future,
                np.concatenate(([past_values_channel[-1]], prediction_data)),
                #  linestyle='--',
                color=colors[model_index],
                label=f"{model_name}, MAE: {mae_predictions[model_name]:.2f}, SMAPE: {smape_predictions[model_name]:.2f}",
            )
            model_index += 1
        # vertical yellow line
        plt.axvline(x=0, color=grid_color, linestyle="--")
        # Add plot title and legend
        plt.title(f"Channel {channel_index + 1}")
        plt.xlabel(label_x_axis)
        plt.ylabel(label_y_axis)
        # plt.ylim(-1,1)
        plt.grid(color=grid_color, linestyle="--")
        # if channel_index == 0:
        plt.legend(loc="lower left")
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()


from utils.experiment_utils import scaler_local_min_max


def plot_all(samples, parameters, colors, history_len, selected_predictions):
    for sample in samples:
        with open(f"{parameters['samples_path']}/{sample}", "rb") as f:
            data = pickle.load(f)
        past_values = data[0][:, -history_len:, :]
        future_values = data[1]
        # Scale the past and future values
        past_values, future_values = scaler_local_min_max(past_values, future_values)
        predictions = data[-1]
        sub_predictions = {key: predictions[key] for key in selected_predictions}
        #
        print(past_values.shape, future_values.shape, sub_predictions.keys())
        plototot(
            past_values,
            future_values,
            sub_predictions,
            suptitle=sample,
            #  colors=["#ffc857", "#e9724c", "#c5283d", "#481d24", "#255f85"],
            # colors = ["#ffc857", "#e9724c", "#c5283d", "#036666", "#67B99A"]
            colors=colors,
            save_path=f"{parameters['visuals_path']}/{sample}.png",
            past_length=100,
            show=False,
        )
