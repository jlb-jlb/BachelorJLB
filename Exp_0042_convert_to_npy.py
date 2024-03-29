"""
If you get the _bz2 error, try executing the file from the singularity container.
"""

import os
import numpy as np
from utils.experiment_utils import convert_csv_to_npy


def main(data_path, target_path):
    os.makedirs(target_path, exist_ok=True)
    files = os.listdir(data_path)
    files = [file for file in files if file.endswith(".csv")]
    sanity_file = None
    for file in files:
        print(file)
        try:
            convert_csv_to_npy(os.path.join(data_path, file), target_path)
            sanity_file = file
        except AssertionError as e:
            print(e)
            continue

    if sanity_file is None:
        raise ValueError("No files were converted to .npy")
    return os.path.join(target_path, sanity_file.replace(".csv", "_data.npy"))


if __name__ == "__main__":
    parameters = {
        "data_paths": ["DATA_EEG/EEG_TEST", "DATA_EEG/EEG_TRAIN"],
        "target_paths": [
            "DATA_EEG/EEG_PREP_ROBUST_TEST",
            "DATA_EEG/EEG_PREP_ROBUST_TRAIN",
        ],
    }

    for data_path, target_path in zip(
        parameters["data_paths"], parameters["target_paths"]
    ):
        sanity_check = main(data_path, target_path)

        print("Sanity check")

        npy_file = np.load(sanity_check)
        print(npy_file.shape)

        print("DONE")
