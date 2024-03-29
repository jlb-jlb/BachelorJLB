# from IPython.display import display, clear_output

import os
import gc
import shutil
import mne
import datetime
import logging
import pathlib

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

from pyprep.prep_pipeline import PrepPipeline
from utils.tuh_eeg_utils import (
    tuh_eeg_load_data_with_annotations,
    tuh_eeg_apply_bipolar_montage,
    tuh_eeg_change_channel_names,
    check_if_only_bckg_in_eeg_recording,
    MONTAGE_ALL,
)
from utils.tuh_eeg_recording_loader import DataLoader as TUH_DataLoader


dataset_config = {
    "extensions": [".edf", ".csv_bi"],
    "l_freq": 0.5,
    "h_freq": 50,
    # "resample_freq": 50,
    # "window_size": 5,  # sec
    # "window_overlap": 1,  # sec
    # "number_channels": 20,
    # "prediction_length": 96,
    # # "lags_seq": lags_sequence,
    # "start_time": datetime.datetime(2015, 1, 1, 0, 0, 0, 0),
    # "freq": "2H",
    # "number_edf_files": 20,
}

CHANNEL_NAMES_19 = [
    "T6",
    "C3",
    "O2",
    "T5",
    "Fp2",
    "Cz",
    "F7",
    "Fp1",
    "O1",
    "F8",
    "A1",
    "T3",
    "P3",
    "F3",
    "C4",
    "P4",
    "F4",
    "A2",
    "T4",
]


print(len(CHANNEL_NAMES_19))
print(CHANNEL_NAMES_19)


def get_data_eeg(
    data_paths,
    verbose=True,
    start_idx=0,
    seiz_or_bckg="seiz",
):
    assert seiz_or_bckg in [
        "seiz",
        "bckg",
    ], "seiz_or_bckg must be either 'seiz' or 'bckg'"

    for i, dataset_path in enumerate(data_paths):
        if i < start_idx:
            continue

        # print(dataset_path[0], dataset_path[1])
        logging.info(f"Current file: {i}/{len(data_paths)}")
        logging.info(f"File name: {dataset_path[0]}")
        if seiz_or_bckg == "seiz":
            if check_if_only_bckg_in_eeg_recording(  # removed the not
                dataset_path[1]
            ):  # check that we only use recordings with seizures
                if verbose:
                    logging.info(f"No Seiz")
                continue
        elif seiz_or_bckg == "bckg":
            if not check_if_only_bckg_in_eeg_recording(dataset_path[1]):
                if verbose:
                    logging.info(f"Seiz")
                continue
        else:
            raise ValueError("seiz_or_bckg must be either 'seiz' or 'bckg'")

        try:
            raw = tuh_eeg_load_data_with_annotations(dataset_path[0], dataset_path[1])

            raw_copy = tuh_eeg_change_channel_names(raw, drop_unknown_channels=True)

            # raw_copy = raw_copy.pick(MONTAGE_ALL)

            if len(raw_copy.ch_names) < 19:  # check if at least 19 channels are present
                if verbose:
                    logging.info(f"Too few channels")
                continue
            else:  # check if the required channels for the montage are present!
                for ch in raw_copy.ch_names:
                    if ch not in CHANNEL_NAMES_19:
                        if verbose:
                            logging.info(f"Too few channels or not the right channels")
                            logging.info(f"Channel: {ch} is missing")
                        continue

            yield i, raw_copy, dataset_path

        except ValueError as e:
            logging.error(dataset_path[0][-100:])
            logging.error("Couldn't load data")
            continue


def data_PREP(dataloader, plots=False):
    """
    Perform data preprocessing using the PyPREP library.

    Args:
        dataloader: An iterator that yields raw data.

    Yields:
        Preprocessed raw data.

    Raises:
        ValueError: If the dataset is too short and a value error occurs during plotting.
        OSError: If too many noisy channels are present and an OSError occurs during fitting.

    Returns:
        None
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    plot_start = 100

    number_files = 0

    for idx, raw, data_path in dataloader:
        # clear_output(wait=True)

        sample_rate = raw.info["sfreq"]

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(60, sample_rate / 2, 60),  # set the freq to 60 Hz
        }

        sample_rate = raw.info["sfreq"]
        raw_copy_ = raw  # removed the .copy() to save memory

        prep = PrepPipeline(raw_copy_, prep_params, montage)

        try:
            logging.info("Fitting prep")
            # display(prep.fit())  #  can get an OSError here if too many noisy channels
            prep.fit()
        except OSError as e:
            logging.error(e)
            continue
        except ValueError as e:
            logging.error(e)
            continue

        logging.info("Bad channels: {}".format(prep.interpolated_channels))
        logging.info(
            "Bad channels original: {}".format(prep.noisy_channels_original["bad_all"])
        )
        logging.info(
            "Bad channels after interpolation: {}".format(prep.still_noisy_channels)
        )

        if len(prep.still_noisy_channels) > 0:
            logging.error("Interpolating didn't work for all channels")
            continue

        # try:
        #     if plots:
        #         raw_copy_.plot(
        #             duration=10, scalings="auto", n_channels=20, start=plot_start
        #         )
        # except ValueError as e:
        #     logging.error(e)
        #     continue

        number_files += 1
        logging.info(f"Number of files: {number_files}")

        yield idx, prep.raw, data_path


def main(path, data_paths, start_idx=0, seiz_or_bckg="seiz"):
    l_freq = dataset_config["l_freq"]
    h_freq = dataset_config["h_freq"]
    unit = "s"
    data_paths
    dataloader = get_data_eeg(
        data_paths, start_idx=start_idx, seiz_or_bckg=seiz_or_bckg
    )
    dataPREPed = data_PREP(dataloader, plots=False)

    logging.info("Starting to save data")
    for idx, raw, data_path in dataPREPed:
        gc.collect()
        # break
        logging.info(f"File name: {data_paths[idx][0]}")
        logging.info(f"Number of channels in file: {len(raw.ch_names)}")

        raw = tuh_eeg_apply_bipolar_montage(
            raw, data_paths[idx][1], only_return_bipolar=True
        )  #  unsure if I should apply the montage or not tbh
        raw = raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design="firwin",
            method="fir",
            verbose=False,
        )
        raw = raw.resample(50)
        df_eeg = raw.to_data_frame()

        df_eeg["time"] = pd.to_datetime(df_eeg["time"], unit=unit)

        data_name = data_path[0].split("/")[-1]
        data_name = data_name[:-4]

        index_number = f"{idx}".zfill(7)

        df_eeg.to_csv(f"{path}/{data_name}_{index_number}.csv", index=False)
        # get the csv_bi file and save copy it to the target folder
        csv_bi_file = data_path[1].split("/")[-1]
        csv_bi_file = csv_bi_file[:-7]
        shutil.copy(data_path[1], f"{path}/{data_name}_{index_number}.csv_bi")
        logging.info(f"Saved file {index_number}")


if __name__ == "__main__":
    # create LOGS folder ok, if exists
    os.makedirs("LOGS", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        filename="./LOGS/logs_100_EEG_PREP_DATA_SEIZURES_{}.log".format(
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y:%m:%d:%H:%M:%S",
    )
    mne.set_log_level("WARNING")
    parameters = {
        "data_paths": [
            "TUH_EEG_SEIZ/edf/dev",
            "TUH_EEG_SEIZ/edf/train",
        ],  # path of recordings parent folder (train or dev)
        "target_paths": ["DATA_EEG/EEG_TEST", "DATA_EEG/EEG_TRAIN"],  # target folder
        "extensions": [".edf", ".csv_bi"],
    }

    for source_path, target_path in zip(
        parameters["data_paths"], parameters["target_paths"]
    ):
        os.makedirs(target_path, exist_ok=True)

        data_paths = TUH_DataLoader(
            source_path, extensions=parameters["extensions"]
        ).file_tuples

        main(target_path, data_paths, start_idx=0, seiz_or_bckg="bckg")

    print("DONE")
