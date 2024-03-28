# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
from utils.lightning_configs import LitModel

parameters = {
    "TimeSeriesTransformer_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGTransformer__336_96__2024_03_13_09_28/checkpoints/best_model_Exp_EEGTransformer__336_96-v1.ckpt",
    "Informer_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGInformer__336_96__2024_03_12_18_16/checkpoints/best_model_Exp_EEGInformer__336_96.ckpt",
    "Autoformer_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/Exp_EEGAutoformer__336_96__2024_03_12_23_56/checkpoints/best_model_Exp_EEGAutoformer__336_96-v1.ckpt",
    "PatchTST_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__336_96__2024_03_14_23_15/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__336_96-v2.ckpt",
    "PatchTST_512": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__512_96__2024_03_14_23_16/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__512_96-v2.ckpt",
    "PatchTST_720": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__720_96__2024_03_15_14_10/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__720_96-v1.ckpt",
    "PatchTST_1024": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_100KTrain_EEGPatchTST__1024_96__2024_03_15_13_24/checkpoints/best_model_EXP_100KTrain_EEGPatchTST__1024_96.ckpt",
    "TSMixer_336": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__336_96__2024_03_11_20_50/checkpoints/best_model_EXP_EEGpatchTSMixer__336_96-v1.ckpt",
    "TSMixer_512": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__512_96__2024_03_11_20_51/checkpoints/best_model_EXP_EEGpatchTSMixer__512_96-v2.ckpt",
    "TSMixer_720": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixer__720_96__2024_03_17_20_10/checkpoints/best_model_EXP_EEGpatchTSMixer__720_96.ckpt",
    "TSMixer_1024": "/home/users/j/joscha.bisping/PROJECTS/Bachelor-Thesis-JLB/EXPERIMENTS/EXP_0411_JLB/EXP_EEGpatchTSMixerMIN3__1024_96__2024_03_17_20_11/checkpoints/best_model_EXP_EEGpatchTSMixerMIN3__1024_96.ckpt",
}


for key, model_path in parameters.items():
    model = LitModel.load_from_checkpoint(model_path)
    hugg_model = model.model
    model_name = f"EEG_{key.split('_')[0]}_{key.split('_')[1]}_history_96_horizon"
    kwargs = {
        "tasks": "forecasting",
        "dataset": "Temple University Data Corpus",
        "tags": ["EEG", "Forecasting", "Time Series", key.split("_")[0]],
    }
    hugg_model.push_to_hub(f"JLB-JLB/{model_name}", **kwargs)
