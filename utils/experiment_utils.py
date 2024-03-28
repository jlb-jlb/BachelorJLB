import numpy as np
import pandas as pd

from tsai.all import *

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple

import gc
import os
import sys
import logging
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler, StandardScaler


def scaler_local_min_max(
    X: np.ndarray,
    y: np.ndarray,
    new_min: int = -1,
    new_max: int = 1,
    include_future=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scale input data (X) and optional future values (y) to a specified range.

    This function uses the MinMaxScaler from scikit-learn to scale the input array (X)
    and an optional future values array (y) to a range between new_min and new_max.
    If include_future is True, it combines X and y before scaling to ensure consistent
    scaling across both datasets. It is particularly useful in time series forecasting
    tasks where future data points (y) are considered during training.

    Parameters
    ----------
    X : numpy.ndarray
        The input data to scale. It should be a 2D array where each row represents a data point and each column represents a feature.
    y : numpy.ndarray
        The future values to scale along with the input data. It is only used if include_future is True. It should have the same number of columns as X.
    new_min : int, optional
        The minimum value of the transformed data. Default is -1.
    new_max : int, optional
        The maximum value of the transformed data. Default is 1.
    include_future : bool, optional
        Flag to include future values (y) in scaling. If True, X and y are combined before scaling. Default is False.

    Returns
    -------
    tuple
        A tuple containing two elements. The first element is a tensor of the scaled past values (X). The second element is a tensor of the scaled future values (y) if include_future is True; otherwise, it's the scaled version of y using the scale of X.

    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[7, 8, 9]])
    >>> scaled_X, scaled_y = scaler_local_min_max(X, y, include_future=False)
    >>> print(scaled_X)
    >>> print(scaled_y)

    Note
    ----
    The function converts the scaled data to PyTorch tensors before returning, making it suitable for machine learning models in PyTorch.
    """
    if include_future:
        past_values = X
        future_values = y
        shape_0 = past_values.shape
        start_index_y = shape_0[0]
        all_data = np.concatenate((past_values, future_values), axis=0)
        data_flattened = all_data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data_flattened).reshape(all_data.shape)
        past_values = torch.from_numpy(scaled_data[:start_index_y, :])
        future_values = torch.from_numpy(scaled_data[start_index_y:, :])
        return past_values, future_values
    else:
        X_flattened = X.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_flattened)
        past_values = torch.from_numpy(scaler.transform(X_flattened).reshape(X.shape))
        future_values = torch.from_numpy(
            scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        )
        return past_values, future_values


# Dataset Classes
class MultiTSDatasetWithTimeEmbedding(Dataset):
    """
    A PyTorch Dataset for time series forecasting that includes time embeddings.

    This dataset is designed for loading and preprocessing time series data for
    machine learning models, especially for forecasting tasks. It supports loading
    data from numpy files, generating time-based features, and scaling the data
    using local min-max scaling.

    For the TimeEmbedding this Dataset Class uses implementations of the gluonts library.

    Parameters
    ----------
    path : str
        The directory path where the dataset file is located.
    filename : str
        The name of the dataset file to load, which should be a `.npy` file.
    timestamp : str, optional
        The starting timestamp for the time series data, used to generate time-based features.
    freq : str, optional
        The frequency of the time series data, used to generate the time index.
    fcst_history : int, optional
        The number of past time steps to include for each sample in the dataset.
    fcst_horizon : int, optional
        The number of future time steps to predict for each sample in the dataset.
    local_min_max : str, optional
        The scaling method to apply. Currently, only "scale_all" is supported, which scales each sample.
    memory_map : bool, optional
        Whether to load the dataset into memory or use memory mapping (useful for large datasets).
    lags_sequence : List[int], optional
        The sequence of lag values to consider for creating historical features.

    Attributes
    ----------
    data : numpy.ndarray
        The loaded time series data.
    time_features : numpy.ndarray
        The generated time-based features for the dataset.

    Methods
    -------
    __len__():
        Returns the number of samples in the dataset.
    __getitem__(idx):
        Returns a sample from the dataset at the specified index, including both data and time features.
    create_time_embedding(timestamp, freq):
        Generates time-based features based on the specified timestamp and frequency.
    describe():
        Returns a string description of the dataset.
    """

    def __init__(
        self,
        path: str,
        filename: str,
        timestamp: str = "2020-01-01 00:00:00",
        freq: str = "20L",
        fcst_history: int = 336,
        fcst_horizon: int = 96,
        local_min_max: str = "scale_all",
        memory_map: bool = False,
        lags_sequence: List[int] = [1, 2, 3, 4, 5, 6, 7],
    ):
        assert filename.endswith(".npy"), "Only .npy files are supported"
        if memory_map:
            self.data = np.load(f"{path}/{filename}", mmap_mode="c")
        else:
            self.data = np.load(f"{path}/{filename}")
        # assert that file is big enough
        assert self.data.shape[1] == 22, "df.shape[1] != 23"

        self.fcst_history = fcst_history
        if lags_sequence:
            self.fcst_history += max(lags_sequence)

        self.fcst_horizon = fcst_horizon
        assert (
            self.data.shape[0] >= self.fcst_history + self.fcst_horizon
        ), "df.shape[0] < fcst_history + fcst_horizon"

        self.timestamp = timestamp
        self.freq = freq
        self.local_min_max = local_min_max
        self.filename = filename
        self.path = path
        self.past_observed_mask = np.ones(
            (self.fcst_history, self.data.shape[1]), dtype=np.float32
        )  # we need to provide observed values for the past and future
        self.future_observed_mask = np.ones(
            (self.fcst_horizon, self.data.shape[1]), dtype=np.float32
        )  #

        self.time_features = self.create_time_embedding(self.timestamp, self.freq)

        # set length
        self.length = int(self.data.shape[0] - self.fcst_history - self.fcst_horizon)
        logging.debug(self.length)  # debug
        logging.debug(self.filename)  # debug

    def __len__(self):
        return self.length

    def describe(self):
        return f"Data: {self.filename}, Shape: {self.data.shape}, Time Features: {self.time_features.shape}"

    def _normalize(self, xs, num: float):
        """Scale values of ``xs`` to [-0.5, 0.5]."""

        return np.asarray(xs) / (num - 1) - 0.5

    def microsecond_of_second(self, index: pd.PeriodIndex) -> np.ndarray:
        """
        Microsecond of second encoded as value between [-0.5, 0.5]
        """
        return self._normalize(index.microsecond, num=1_000_000)

    def second_of_minute(self, index: pd.PeriodIndex) -> np.ndarray:
        """
        Second of minute encoded as value between [-0.5, 0.5]
        """
        return self._normalize(index.second, num=60)

    def minute_of_hour(self, index: pd.PeriodIndex) -> np.ndarray:
        """
        Minute of hour encoded as value between [-0.5, 0.5]
        """
        return self._normalize(index.minute, num=60)

    def hour_of_day(self, index: pd.PeriodIndex) -> np.ndarray:
        """
        Hour of day encoded as value between [-0.5, 0.5]
        """

        return self._normalize(index.hour, num=24)

    def create_time_embedding(self, timestamp, freq):
        # create time embedding
        self.date_range = pd.date_range(
            start=timestamp, periods=self.data.shape[0], freq=freq
        )
        microsecond_of_second = self.microsecond_of_second(self.date_range)
        second_of_minute = self.second_of_minute(self.date_range)
        minute_of_hour = self.minute_of_hour(self.date_range)
        hour_of_day = self.hour_of_day(self.date_range)
        age = np.log1p(range(self.data.shape[0]))

        return np.stack(
            [microsecond_of_second, second_of_minute, minute_of_hour, hour_of_day, age],
            axis=1,
            dtype=np.float32,
        )

    def __getitem__(self, idx):

        data_all = self.data[idx : idx + self.fcst_history + self.fcst_horizon]

        time_features_all = self.time_features[
            idx : idx + self.fcst_history + self.fcst_horizon
        ]

        past_values = data_all[: self.fcst_history]
        past_time_features = time_features_all[: self.fcst_history]

        future_values = data_all[self.fcst_history :]
        future_time_features = time_features_all[self.fcst_history :]

        if self.local_min_max == "scale_all":
            past_values, future_values = scaler_local_min_max(
                past_values, future_values, new_min=-1, new_max=1, include_future=False
            )

        return (
            past_values,
            future_values,
            past_time_features,
            future_time_features,
            self.past_observed_mask,
            self.future_observed_mask,
        )


class MultiTSDataset(Dataset):
    """
    A dataset class for loading and preprocessing time series data for machine learning models, particularly for forecasting tasks.

    This class supports reading data from .csv and .npy files, applying various scaling techniques, and handling data in chunks to manage memory usage efficiently. It's designed to work with forecasting models, providing functionality to generate training samples based on specified forecast history and horizon.

    Parameters
    ----------
    path : str
        Directory path where the dataset file is located.
    filename : str
        Name of the dataset file. Supported formats are .csv and .npy.
    timestamp : str, optional
        Starting timestamp for the time series data, used with frequency to generate datetime indices. Only relevant for .csv files.
    freq : str, optional
        Frequency of the time series data, used to generate the time index. Default is "2ms".
    fcst_history : int, optional
        Number of past timesteps to include for each forecasting sample. Default is 96.
    fcst_horizon : int, optional
        Number of future timesteps to predict for each forecasting sample. Default is 24.
    valid_size : int, optional
        Size of the validation set. Default is 0.
    test_size : int, optional
        Size of the test set. Default is 0.
    stride : int, optional
        Stride length when generating samples from the time series. Default is 1.
    datetime_col : str, optional
        Column name for datetime information in the .csv file. Default is "time".
    scaling : str, optional
        Type of scaling to apply to the data. Options are "robust", "standard", or "minmax". Default is "robust".
    local_min_max : bool, optional
        Flag to indicate whether local min-max scaling should be applied. Default is False.
    save_path : str, optional
        Path to save processed data files. If None, uses `path`. Only relevant for .csv files.
    verbose : bool, optional
        If True, prints additional information during processing. Default is True.
    memory_map : bool, optional
        If True, uses memory mapping for .npy files to reduce memory usage. Default is False.

    Methods
    -------
    memory_usage():
        Returns a string describing the memory usage of the loaded data.
    describe():
        Returns a string summary of the dataset, including the filename and shape of the data.
    __len__():
        Returns the total number of samples in the dataset.
    __getitem__(idx):
        Returns the data for a specific index, including past and future values based on the forecast history and horizon.
    cleanup():
        Removes any temporary files created by the dataset, applicable to .csv data processing.

    Example
    -------
    >>> dataset = MultiTSDataset(path="data/", filename="timeseries.csv", fcst_history=96, fcst_horizon=24, scaling="robust")
    >>> print(dataset.describe())
    Data: timeseries.csv
    X shape: (1000, 22)
    """

    def __init__(
        self,
        path: str,
        filename: str,
        timestamp: str = None,
        freq: str = "20L",
        fcst_history: int = 336,
        fcst_horizon: int = 96,
        local_min_max: bool = False,
        memory_map: bool = False,
    ):
        if filename.endswith(".csv"):
            raise NotImplementedError("Not implemented, File must be .npy")

        elif filename.endswith(".npy"):
            # raise Exception("Not implemented yet")
            if memory_map:
                self.X = np.load(f"{path}/{filename}", mmap_mode="c")
            else:
                self.X = np.load(f"{path}/{filename}")

            assert self.X.shape[1] == 22, "df.shape[1] != 22"
            assert (
                self.X.shape[0] >= fcst_history + fcst_horizon
            ), "df.shape[0] < fcst_history + fcst_horizon"

            self.fcst_history = fcst_history
            self.fcst_horizon = fcst_horizon
            self.timestamp = timestamp
            self.freq = freq

        else:
            raise Exception("File format not supported")

        self.filename = filename
        self.path = path
        self.local_min_max = local_min_max

    def memory_usage(self):
        return f"Memory Usage X: {sys.getsizeof(self.X):12} bytes ({bytes2str(sys.getsizeof(self.X))})"  # \nMemory Usage y: {sys.getsizeof(self.y):12} bytes ({bytes2str(sys.getsizeof(self.y))})'

    def describe(self):
        return f"Data: {self.filename}\nX shape: {self.X.shape}"  # \ny shape: {self.y.shape}\n{self.memory_usage()}"

    def __len__(self):
        return int(self.X.shape[0] - self.fcst_history - self.fcst_horizon)

    def __getitem__(self, idx):
        data_all = self.X[idx : idx + self.fcst_history + self.fcst_horizon]
        past_values = data_all[: self.fcst_history]
        future_values = data_all[self.fcst_history :]

        if self.local_min_max == "scale_all":
            past_values, future_values = scaler_local_min_max(
                past_values, future_values, new_min=-1, new_max=1, include_future=False
            )

        return past_values, future_values

    def cleanup(self):
        # delete both memmap files
        os.remove(f"{self.path}/{self.filename}_data.npy")


#  create npy file
def convert_csv_to_npy(
    file_path: str, save_path: str, scaling: str = "robust", verbose: bool = True
):
    """
    Converts a CSV file to an NPY file, with optional data scaling.

    This function reads data from a specified CSV file, optionally applies scaling (robust, standard, or min-max),
    and saves the processed data as an NPY file for efficient storage and access.

    Parameters
    ----------
    file_path : str
        The path to the input CSV file. The file should have a shape of (n_samples, 23) where the first column is expected
        to be an index or identifier column, not used in scaling.
    save_path : str
        The directory path where the output NPY file will be saved. The output file's name is derived from the input file's name.
    scaling : str, optional
        The type of scaling to apply to the data. Options include "robust", "standard", and "minmax". Default is "robust".
    verbose : bool, optional
        If True, prints additional information during processing. Default is True.

    Raises
    ------
    AssertionError
        If the input file is not a CSV file, if the CSV does not have 23 columns, or if save_path is not provided.
    NotImplementedError
        If "minmax" scaling is selected, which is not implemented in this example.

    Examples
    --------
    >>> convert_csv_to_npy("data/input_data.csv", "data/processed", scaling="robust", verbose=True)
    """
    assert file_path.endswith(".csv"), "Only .csv files are supported"
    assert save_path is not None, "save_path must be provided"
    df = pd.read_csv(file_path)
    fname = file_path.split("/")[-1].replace(".csv", "")
    if verbose:
        logging.debug(df.shape)
    assert df.shape[1] == 23, "df.shape[1] != 23"
    x_vars = df.columns[1:]
    if verbose:
        logging.debug(f"x_vars: {x_vars}")

    # Apply scaling based on the specified method
    if scaling == "robust":
        scaler = RobustScaler()
        df[x_vars] = scaler.fit_transform(df[x_vars])
    elif scaling == "standard":
        scaler = StandardScaler()
        df[x_vars] = scaler.fit_transform(df[x_vars])
    elif scaling == "minmax":
        raise NotImplementedError("Min-max scaling has not been implemented.")
    else:
        if verbose:
            logging.info("No scaling applied")

    shape_ = (df.shape[0], df.shape[1] - 1)
    if verbose:
        logging.debug(f"shape_: {shape_}")

    np.save(f"{save_path}/{fname}_data.npy", df.values[:, 1:].astype(np.float32))


class ConcatDataset(Dataset):
    """
    A dataset class that concatenates multiple PyTorch datasets for sequential access.

    This class is designed to handle multiple datasets and provide an interface to access them as a single dataset.
    It keeps track of the lengths and cumulative lengths of the individual datasets to correctly handle indexing across them.

    Parameters
    ----------
    *datasets : tuple of Dataset
        Variable number of dataset instances to be concatenated. These datasets should implement the `__len__` and
        `__getitem__` methods as expected by PyTorch datasets.

    Attributes
    ----------
    datasets : tuple of Dataset
        The datasets passed to the constructor.
    seed : int
        The seed for the random number generator. This is currently set to a fixed value (42) and is not used within the class.
    rng : np.random.Generator
        An instance of numpy's random number generator, seeded with `seed`. Currently not used in the class but can be utilized for randomized operations on datasets.
    length : int
        The total combined length of all datasets.
    lengths : np.ndarray
        An array containing the lengths of each dataset.
    cum_lengths : np.ndarray
        The cumulative lengths of the datasets, used to determine dataset boundaries within the concatenated dataset.
    selected_indices : list of set
        Currently not used in the class but can be utilized to keep track of accessed indices or samples for each dataset.

    Methods
    -------
    __getitem__(idx):
        Retrieves an item from the concatenated dataset based on the global index `idx`.
    __len__():
        Returns the total length of the concatenated dataset.
    describe():
        Returns a string describing the concatenated dataset, including the number of datasets it contains.
    cleanup():
        Calls the `cleanup` method on all constituent datasets, if they implement it.

    Example
    -------
    >>> dataset_list = [CustomDataset1(), CustomDataset2()]
    >>> concat_dataset = ConcatDataset(*dataset_list)
    >>> print(concat_dataset.describe())
    ConcatDataset with 2 datasets
    >>> len(concat_dataset)  # Combined length of dataset1 and dataset2
    >>> data_item = concat_dataset[5]  # Accesses 5th item from the combined dataset
    """

    def __init__(self, *datasets):
        self.datasets = datasets
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.length = sum(len(d) for d in datasets)
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_lengths = np.cumsum(self.lengths)
        self.selected_indices = [set() for _ in datasets]

    def __getitem__(self, idx):
        """List of datasets that gets iterated over we don't mix them, because we can do this before"""
        # chose dataset that has the cumsum larger than idx
        ds_id = np.argmax(self.cum_lengths > idx)
        ds = self.datasets[ds_id]
        # get the data from the dataset
        if ds_id != 0:
            lowest_data_idx = self.cum_lengths[ds_id - 1]
        else:
            lowest_data_idx = 0
        return ds[idx - lowest_data_idx]

    def __len__(self):
        return self.length

    def describe(self):
        return f"ConcatDataset with {len(self.datasets)} datasets"

    def cleanup(self):
        for ds in self.datasets:
            ds.cleanup()


class ConcatDatasetTest(Dataset):
    """
    A PyTorch Dataset for concatenating multiple datasets with an option to limit the total number of samples.

    This dataset allows for the sequential access of multiple datasets as if they were a single dataset. It provides
    the functionality to limit the maximum number of samples that can be accessed, making it suitable for scenarios
    where working with a smaller subset of the combined datasets is desired.

    Parameters
    ----------
    *datasets : Dataset
        A variable number of dataset instances to be concatenated.
    max_samples : int, optional
        The maximum number of samples to be accessed from the concatenated dataset. If None, all samples are accessible.
    seed : int, optional
        The random seed used for generating the mapping of indices when max_samples is specified. This ensures consistent
        access patterns across runs.

    Attributes
    ----------
    datasets : tuple of Dataset
        The datasets to be concatenated.
    seed : int
        The seed for random operations, ensuring reproducibility.
    lengths : np.ndarray
        An array of the lengths of each individual dataset.
    cum_lengths : np.ndarray
        The cumulative lengths of the datasets, aiding in index mapping across the concatenated dataset.
    length : int
        The total length of the concatenated dataset or the maximum number of samples if max_samples is specified.
    mapping : dict
        A dictionary mapping the accessible indices to the original indices of the concatenated dataset, used when max_samples is specified.

    Methods
    -------
    __getitem__(idx):
        Retrieves an item from the concatenated dataset based on a possibly remapped index `idx`.
    __len__():
        Returns the effective length of the concatenated dataset, respecting the max_samples constraint.
    describe():
        Provides a brief description of the dataset, including the number of concatenated datasets and the effective maximum length.
    cleanup():
        Calls the cleanup method on all contained datasets, if available.

    Example
    -------
    >>> dataset1 = CustomDataset1()
    >>> dataset2 = CustomDataset2()
    >>> concat_dataset_test = ConcatDatasetTest(dataset1, dataset2, max_samples=100, seed=42)
    >>> print(concat_dataset_test.describe())
    ConcatDatasetTest with 2 datasets and maximum length of 100
    """

    def __init__(self, *datasets, max_samples: int = None, seed: int = 42):
        self.datasets = datasets
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_lengths = np.cumsum(self.lengths)
        self.length = sum(self.lengths)

        self.max_samples = max_samples
        self.mapping = {}
        if max_samples:
            self._generate_mapping()
            self.length = min(max_samples, self.length)

    def _generate_mapping(self):
        assert self.max_samples <= self.length, "max_samples > length"

        # create output values
        output_values = list(range(self.length))
        # shuffle
        self.rng.shuffle(output_values)
        # create mapping of max val samples to output values
        self.mapping = {i: output_values[i] for i in range(self.max_samples)}

    def __getitem__(self, idx):
        """List of datasets that gets iterated over we don't mix them, because we can do this before"""
        # chose dataset that has the cumsum larger than idx
        assert idx < self.length, "INDEX ERROR idx >= length"
        if self.max_samples:
            idx = self.mapping[idx]

        ds_id = np.argmax(self.cum_lengths > idx)
        ds = self.datasets[ds_id]
        # get the data from the dataset
        # get the data from the dataset
        if ds_id != 0:
            lowest_data_idx = self.cum_lengths[ds_id - 1]
        else:
            lowest_data_idx = 0

        return ds[idx - lowest_data_idx]

    def __len__(self):
        return self.length

    def describe(self):
        return f"ConcatDatasetTest with {len(self.datasets)} datasets and maximum length of {self.max_samples}"

    def cleanup(self):
        for ds in self.datasets:
            if hasattr(ds, "cleanup"):
                ds.cleanup()


def npy_files(
    path: str,
    max_datasets: int = None,
    fcst_history: int = 336,
    fcst_horizon: int = 96,
    local_min_max: str = None,
    test: bool = False,
    max_samples: int = None,
    verbose: bool = False,
    time_embedding: bool = False,
    lags_sequence: list = [1, 2, 3, 4, 5, 6, 7],
    memory_map: bool = False,
) -> Dataset:
    """
    Loads time series datasets from a specified path and returns a concatenated dataset.

    This function reads .npy files, creates dataset instances
    for each file, and optionally concatenates them into a single dataset object. It supports the inclusion of time embeddings
    and can limit the number of datasets or samples for testing purposes.

    Parameters
    ----------
    path : str
        The directory path containing the dataset files.
    max_datasets : int, optional
        The maximum number of dataset files to load. If None, loads all available datasets in the path.
    fcst_history : int, required
        The number of historical time steps to include for forecasting. Default is 336.
    fcst_horizon : int, required
        The forecast horizon, i.e., the number of future time steps to predict. Default is 96.
    local_min_max : str, optional
        Flag indicating whether to apply local min-max scaling. Default is False.
    test : bool, optional
        If True, uses a testing setup that can limit the number of samples. Default is False.
    max_samples : int, optional
        The maximum number of samples for the testing dataset. Only relevant if `test` is True.
    verbose : bool, optional
        If True, prints additional information during dataset loading. Default is False.
    time_embedding : bool, optional
        If True, includes time embeddings in the dataset. Default is False.
    lags_sequence : list, optional
        A list of lag values to include as features. Default is [1, 2, 3, 4, 5, 6, 7].
    memory_map : bool, optional
        If True, loads the datasets using memory mapping to reduce memory usage. Default is False.

    Returns
    -------
    Dataset
        A concatenated dataset object containing all loaded datasets or a subset if `max_datasets` or `max_samples` is specified.

    Raises
    ------
    AssertionError
        If any assertions in the dataset loading process fail.

    Example
    -------
    >>> concat_dataset = npy_files(
            path="/data/time_series",
            max_datasets=10,
            test=True,
            max_samples=1000,
            time_embedding=True,
            verbose=True
        )
    """
    all_files = os.listdir(path)
    all_files.sort()
    datasets = []

    for file_n in all_files[:max_datasets]:
        try:
            if not time_embedding:
                dataset = MultiTSDataset(
                    path=path,
                    filename=file_n,
                    fcst_history=fcst_history,
                    fcst_horizon=fcst_horizon,
                    local_min_max=local_min_max,
                    memory_map=memory_map,
                )
            else:
                dataset = MultiTSDatasetWithTimeEmbedding(
                    path=path,
                    filename=file_n,
                    fcst_history=fcst_history,
                    fcst_horizon=fcst_horizon,
                    local_min_max=local_min_max,
                    lags_sequence=lags_sequence,
                    memory_map=memory_map,
                )
        except AssertionError as e:
            logging.error(f"ERROR in npy_files dataset: {e}")
            continue
        datasets.append(dataset)
        logging.info(dataset.describe())

    #  Concatenate all datasets
    if not test:
        concat_dataset = ConcatDataset(*datasets)
    else:
        concat_dataset = ConcatDatasetTest(*datasets, max_samples=max_samples)
    logging.info(f"Concatenated dataset: {concat_dataset.length}")
    return concat_dataset


def prepare_data(
    path: str,
    max_datasets: int = None,
    fcst_history: int = 336,
    fcst_horizon: int = 96,
    local_min_max: str = None,
    test: bool = False,
    max_samples: int = None,
    verbose: bool = False,
    time_embedding: bool = False,
    lags_sequence: list = [1, 2, 3, 4, 5, 6, 7],
    memory_map: bool = False,
):
    """
    Prepares data from a specified path, handling both .npy and potentially .csv files.

    This function identifies the file types present in the specified directory and
    calls the appropriate handling function based on the file type detected. Currently,
    it processes .npy files and includes a structure for extending support to .csv files.

    Parameters
    ----------
    path : str
        The directory path containing the dataset files.
    max_datasets : int, optional
        The maximum number of dataset files to load.
    fcst_history : int, optional
        The number of past time steps included for forecasting.
    fcst_horizon : int, optional
        The forecast horizon in future time steps.
    local_min_max : str, optional
        Specifies if local min-max scaling should be applied. Pass "scale_all" to activate.
    test : bool, optional
        If True, operates in a test mode that can limit the number of samples.
    max_samples : int, optional
        The maximum number of samples to use in test mode.
    verbose : bool, optional
        If True, prints additional information during data loading.
    time_embedding : bool, optional
        If True, includes time embeddings in the dataset.
    lags_sequence : list, optional
        A list of lags to include as features.
    memory_map : bool, optional
        If True, uses memory mapping to load .npy files, reducing memory usage.

    Returns
    -------
    Dataset
        A dataset object ready for training or testing, depending on the parameters.

    Raises
    ------
    ValueError
        If no suitable files (.npy or .csv) are found in the specified path.
    """
    all_files = os.listdir(path)
    # check if there is a file in the all_files list that ends with .csv
    # if any([file.endswith(".csv") for file in all_files]):
    #     return csv_files(
    #         path,
    #         max_datasets=max_datasets,
    #         fcst_history=fcst_history,
    #         fcst_horizon=fcst_horizon,
    #         verbose=verbose,
    #     )
    # check if all files end with .npy
    if all([file.endswith(".npy") for file in all_files]):
        return npy_files(
            path,
            max_datasets=max_datasets,
            fcst_history=fcst_history,
            fcst_horizon=fcst_horizon,
            local_min_max=local_min_max,
            verbose=verbose,
            test=test,
            max_samples=max_samples,
            time_embedding=time_embedding,
            lags_sequence=lags_sequence,
            memory_map=memory_map,
        )
    # Raise error if no supported file types are found
    raise ValueError("No supported files (.npy or .csv) found in the specified path.")


############################################################################################################
#  visualizations etc
def plot_prediction_TST(
    past_values: np.ndarray,
    future_values: np.ndarray,
    future_predictions: np.ndarray,
    batch_idx: int = 0,
    channel_idx: int = 0,
    show: bool = True,
    save_path: Optional[str] = None,
    offset: Optional[int] = None,
    y_axis_limit: Tuple[int, int] = (-2, 2),
    figsize: Tuple[int, int] = (15, 10),
    title: Optional[str] = None,
):
    """
    Plots the past values, actual future values, and predicted future values of a time series for a given batch and channel.

    Parameters
    ----------
    past_values : np.ndarray
        A 3D numpy array of shape (batch_size, sequence_length, num_channels) containing the past values of the time series.
    future_values : np.ndarray
        A 3D numpy array of the same shape as `past_values` containing the actual future values of the time series.
    future_predictions : np.ndarray
        A 3D numpy array of the same shape as `past_values` containing the model's predictions for the future values of the time series.
    batch_idx : int, optional
        The index of the batch to plot. Defaults to 0.
    channel_idx : int, optional
        The index of the channel to plot. Defaults to 0.
    show : bool, optional
        If True, displays the plot. Defaults to True.
    save_path : Optional[str], optional
        The path to save the plot image. If None, the plot is not saved. Defaults to None.
    offset : Optional[int], optional
        An optional offset to start plotting the past values from a certain index. Defaults to None.
    y_axis_limit : Tuple[int, int], optional
        The limits for the Y-axis as a tuple (min, max). Defaults to (-2, 2).
    figsize : Tuple[int, int], optional
        The size of the figure as a tuple (width, height) in inches. Defaults to (15, 10).
    title : Optional[str], optional
        The title of the plot. If None, no title is set. Defaults to None.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    past_vals = past_values[batch_idx, :, channel_idx]
    future_vals = future_values[batch_idx, :, channel_idx]
    future_preds = future_predictions[batch_idx, :, channel_idx]

    if offset:
        past_vals = past_vals[offset:]
    index = np.arange(len(past_vals) + len(future_vals))
    # plot actual past values
    ax.plot(
        index[: len(past_vals)],
        past_vals,
        label="actual past",
        color="#073b4c",
    )
    # plot actual future values
    ax.plot(
        index[len(past_vals) :],
        future_vals,
        label="actual future",
        color="#073b4c",
        alpha=0.9,
        # linestyle="--"
    )
    # plot the gap between past and future
    ax.plot(
        [index[len(past_vals) - 1], index[len(past_vals)]],
        [past_vals[-1], future_vals[0]],
        color="#073b4c",
        alpha=0.9,
        # linestyle="--"
    )
    # plot predictions
    ax.plot(
        index[len(past_vals) :],
        future_preds,
        label="predictions",
        color="#ef476f",
        alpha=0.9,
    )

    if y_axis_limit:
        ax.set_ylim(y_axis_limit[0], y_axis_limit[1])
    ax.legend(loc="lower left")
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_prediction(
    sample,
    predictions,
    batch_idx=0,
    channel_idx=0,
    single_example_idx=40,
    show=True,
    save_path=None,
    offset=None,
    y_axis_limit=(-2, 2),
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 10))

    index = np.arange(
        len(sample["past_values"][batch_idx, :, channel_idx])
        + len(sample["future_values"][batch_idx, :, channel_idx])
    )

    past_vals = sample["past_values"][batch_idx, :, channel_idx]
    print(past_vals.shape)
    future_vals = sample["future_values"][batch_idx, :, channel_idx]
    mean_prediction = predictions.mean(axis=1)[batch_idx, :, channel_idx]
    predictions_std = predictions.std(axis=1)[batch_idx, :, channel_idx]
    single_example = predictions[batch_idx, single_example_idx, :, channel_idx]

    if offset:
        past_vals = past_vals[offset:]

    index = np.arange(len(past_vals) + len(future_vals))

    # plot actual past values
    ax.plot(
        index[: len(past_vals)],
        past_vals,
        label="actual past",
        color="#073b4c",
    )
    # plot actual future values
    ax.plot(
        index[len(past_vals) - 1 : -1],
        future_vals,
        label="actual future",
        color="#073b4c",
        alpha=0.9,
        # linestyle="--"
    )

    # plot predictions mean
    ax.plot(
        index[len(past_vals) - 1 : -1],
        mean_prediction,
        label="mean prediction",
        color="#ffd166",
        alpha=0.9,
    )

    # plot a single example from the 100 predictions
    if single_example_idx is not None:
        ax.plot(
            index[len(past_vals) - 1 : -1],
            single_example,
            label="single example",
            alpha=0.7,
            color="#ef476f",
            # linestyle="--",
        )

    ax.fill_between(
        index[len(past_vals) - 1 : -1],
        mean_prediction - predictions_std,
        mean_prediction + predictions_std,
        alpha=0.8,
        interpolate=True,
        color="#118ab2",
        label="+/- 1-std",
    )

    if y_axis_limit:
        ax.set_ylim(y_axis_limit[0], y_axis_limit[1])

    ax.legend(loc="lower left")
    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)
