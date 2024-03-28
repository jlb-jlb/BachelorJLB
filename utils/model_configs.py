import torch
from transformers import (
    InformerConfig,
    InformerForPrediction,
    AutoformerConfig,
    AutoformerForPrediction,
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
)


def get_model(model_name: str, parameters: dict) -> torch.nn.Module:
    """
    model_name: str, Available models: "Informer", "Autoformer", "TimeSeriesTransformer"

    parameters: dict, model parameters
        must contain the following:
        - prediction_length: int, number of steps to predict
        - context_length: int, number of steps to use as context
        - lags_sequence: list, list of lags to use as input requiered for TimeSeriesTransformer, Autoformer, Informer
    """
    if "informer" in model_name.lower():
        config = InformerConfig(  # config from the original paper
            # in the multivariate setting, input_size is the number of variates in the time series per time step
            input_size=22,  # num_of_variates,
            # prediction length:
            prediction_length=parameters["prediction_length"],
            # context length:
            context_length=parameters["context_length"],
            # lags values to use:
            lags_sequence=parameters["lags_sequence"],
            # time features to use + age feature:
            num_time_features=5,
            # paper params
            encoder_layers=4,  # prev 4
            encoder_attention_heads=16,
            encoder_layerdrop=0.1,
            # dropout=0.1,
            encoder_ffn_dim=2048,
            decoder_layers=2,  # prev 2
            decoder_attention_heads=8,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.1,
            attention_type="prop",
            # project input from num_of_variates*len(lags_sequence)+num_time_features to:
            d_model=512,
        )
        model = InformerForPrediction(config)
    elif "autoformer" in model_name.lower():
        config = AutoformerConfig(
            prediction_length=parameters["prediction_length"],
            context_length=parameters["context_length"],
            input_size=22,  # num_of_variates,
            lags_sequence=parameters["lags_sequence"],
            num_time_features=5,
            # params to match informer paper
            encoder_layers=4,
            d_model=512,  # 512 in paper
            encoder_attention_heads=16,
            encoder_layerdrop=0.1,
            encoder_ffn_dim=2048,
            decoder_layers=2,
            decoder_attention_heads=8,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.1,
            moving_average=25,  # same as paper
        )
        model = AutoformerForPrediction(config)
    elif "transformer" in model_name.lower():
        config = TimeSeriesTransformerConfig(
            prediction_length=parameters["prediction_length"],
            context_length=parameters["context_length"],
            input_size=22,  # num_of_variates,
            lags_sequence=parameters["lags_sequence"],
            num_time_features=5,
            # params to match informer paper
            encoder_layers=4,
            d_model=512,  # 512 in paper
            encoder_attention_heads=16,
            encoder_layerdrop=0.1,
            encoder_ffn_dim=2048,
            decoder_layers=2,
            decoder_attention_heads=8,
            decoder_ffn_dim=2048,
            decoder_layerdrop=0.1,
        )
        model = TimeSeriesTransformerForPrediction(config)
    elif "patchtst" in model_name.lower():
        config = PatchTSTConfig(
            num_input_channels=22,
            context_length=parameters["context_length"],
            patch_length=16,
            patch_stride=8,
            prediction_length=parameters["prediction_length"],
            random_mask_ratio=0.4,
            d_model=512,
            num_attention_heads=16,
            num_hidden_layers=6,
            ffn_dim=2048,
            dropout=0.2,
            head_dropout=0.2,
            pooling_type=None,
            channel_attention=False,
            scaling="std",
            loss="mse",
            pre_norm=True,
            norm_type="batchnorm",
        )
        model = PatchTSTForPrediction(config)
    elif "tsmixer" in model_name.lower():
        config = PatchTSMixerConfig(
            context_length=parameters["context_length"],
            prediction_length=parameters["prediction_length"],
            num_input_channels=22,
            patch_stride=16,
            patch_length=16,
            d_model=512,
            num_layers=8,
            expansion_factor=5,
            dropout=0.2,
            head_dropout=0.2,
            # mode="common_channel",
            mode="mix_channel",
            scaling="std",
        )
        model = PatchTSMixerForPrediction(config)
    else:
        raise ValueError(f"Model {model_name} not available.")
    return model
