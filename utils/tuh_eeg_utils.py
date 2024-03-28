import mne
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import MinMaxScaler

# The standard 10-20 system montage that is generally used for EEG data, especially in the Temple University Dataset.
STANDARD_10_20 = mne.channels.make_standard_montage("standard_1020")


# file: $NEDC_NFC/util/python/nedc_convert_ann_files/montages/01_tcp_ar_montage.txt
#
# This file contains a definition for one of our standard montages. These
# montages are described in this document:
#
#  Ferrell, S., Mathew, V., Refford, M., Tchiong, V., Ahsan, T.,
#  Obeid, I., & Picone, J. (2020). The Temple University Hospital EEG Corpus:
#  Electrode Location and Channel Labels.
#  URL: www.isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes
#
# This file contains the most popular bipolar Temporal Central Parasagittal
# (TCP) Averaged Reference (AR) montage. See TCP_AR in Appendix A.

MONTAGE_ALL = [  # Montage that is included in all EEG Recordings
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]  # len 21/20

MONTAGE_PAIRS_TCP_AR = [  # 01_tcp_ar
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "Cz"),
    ("Cz", "C4"),
    ("C4", "T4"),
    ("T4", "A2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]


MONTAGE_PAIRS_TCP_AR_TUH = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG A1-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG CZ-REF"),
    ("EEG CZ-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG A2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]
MONTAGE_PAIRS_TCP_AR_TUH_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


# This file contains a definition for one of our standard montages. These
# montages are described in this document:
#
#  Ferrell, S., Mathew, V., Refford, M., Tchiong, V., Ahsan, T.,
#  Obeid, I., & Picone, J. (2020). The Temple University Hospital EEG Corpus:
#  Electrode Location and Channel Labels.
#  URL: www.isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes
#
# This file contains the second most popular bipolar Temporal Central
# Parasagittal (TCP) montage: a Linked Ears Reference (LE) montage.
# See TCP_LE in Appendix A.

MONTAGE_PAIRS_TCP_LE = [  # 02_tcp_le
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("A1", "T3"),
    ("T3", "C3"),
    ("C3", "Cz"),
    ("Cz", "C4"),
    ("C4", "T4"),
    ("T4", "A2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

MONTAGE_PAIRS_TCP_LE_TUH = [
    ("EEG FP1-LE", "EEG F7-LE"),
    ("EEG F7-LE", "EEG T3-LE"),
    ("EEG T3-LE", "EEG T5-LE"),
    ("EEG T5-LE", "EEG O1-LE"),
    ("EEG FP2-LE", "EEG F8-LE"),
    ("EEG F8-LE", "EEG T4-LE"),
    ("EEG T4-LE", "EEG T6-LE"),
    ("EEG T6-LE", "EEG O2-LE"),
    ("EEG A1-LE", "EEG T3-LE"),
    ("EEG T3-LE", "EEG C3-LE"),
    ("EEG C3-LE", "EEG CZ-LE"),
    ("EEG CZ-LE", "EEG C4-LE"),
    ("EEG C4-LE", "EEG T4-LE"),
    ("EEG T4-LE", "EEG A2-LE"),
    ("EEG FP1-LE", "EEG F3-LE"),
    ("EEG F3-LE", "EEG C3-LE"),
    ("EEG C3-LE", "EEG P3-LE"),
    ("EEG P3-LE", "EEG O1-LE"),
    ("EEG FP2-LE", "EEG F4-LE"),
    ("EEG F4-LE", "EEG C4-LE"),
    ("EEG C4-LE", "EEG P4-LE"),
    ("EEG P4-LE", "EEG O2-LE"),
    # ("EEG EKG-LE", "EEG EKG-LE")  # Special case for EKG # remove ekg cause it spikes to much
]

MONTAGE_PAIRS_TCP_LE_TUH_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]  # , 'EKG-EKG']


# This file contains a definition for one of our standard montages. These
# montages are described in this document:
#
#  Ferrell, S., Mathew, V., Refford, M., Tchiong, V., Ahsan, T.,
#  Obeid, I., & Picone, J. (2020). The Temple University Hospital EEG Corpus:
#  Electrode Location and Channel Labels.
#  URL: www.isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes
#
# This file contains a less common Temporal Central Parasagittal (TCP)
# Averaged Reference (AR) montage that only uses 20 channels. See TCP_AR_A
# in Appendix A.

MONTAGE_PAIRS_TCP_AR_A = [  # 03_tcp_ar_a
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("T3", "C3"),
    ("C3", "Cz"),
    ("Cz", "C4"),
    ("C4", "T4"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

MONTAGE_PAIRS_TCP_AR_A_TUH = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG T3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG CZ-REF"),
    ("EEG CZ-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG T4-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]

MONTAGE_PAIRS_TCP_AR_A_TUH_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


# This file contains a definition for one of our standard montages. These
# montages are described in this document:
#
#  Ferrell, S., Mathew, V., Refford, M., Tchiong, V., Ahsan, T.,
#  Obeid, I., & Picone, J. (2020). The Temple University Hospital EEG Corpus:
#  Electrode Location and Channel Labels.
#  URL: www.isip.piconepress.com/publications/reports/2020/tuh_eeg/electrodes
#
# This file contains one of the least common bipolar Temporal Central
# Parasagittal (TCP) montages: a Linked Ears Reference (LE) montage that
# uses only 20 channels. See TCP_LE_A in Appendix A.

MONTAGE_PAIRS_TCP_LE_A = [  # 04_tcp_le_a
    ("FP1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("FP2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("T3", "C3"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("C4", "T4"),
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

MONTAGE_PAIRS_TCP_LE_A_TUH = [
    ("EEG FP1-LE", "EEG F7-LE"),
    ("EEG F7-LE", "EEG T3-LE"),
    ("EEG T3-LE", "EEG T5-LE"),
    ("EEG T5-LE", "EEG O1-LE"),
    ("EEG FP2-LE", "EEG F8-LE"),
    ("EEG F8-LE", "EEG T4-LE"),
    ("EEG T4-LE", "EEG T6-LE"),
    ("EEG T6-LE", "EEG O2-LE"),
    ("EEG T3-LE", "EEG C3-LE"),
    ("EEG C3-LE", "EEG CZ-LE"),
    ("EEG CZ-LE", "EEG C4-LE"),
    ("EEG C4-LE", "EEG T4-LE"),
    ("EEG FP1-LE", "EEG F3-LE"),
    ("EEG F3-LE", "EEG C3-LE"),
    ("EEG C3-LE", "EEG P3-LE"),
    ("EEG P3-LE", "EEG O1-LE"),
    ("EEG FP2-LE", "EEG F4-LE"),
    ("EEG F4-LE", "EEG C4-LE"),
    ("EEG C4-LE", "EEG P4-LE"),
    ("EEG P4-LE", "EEG O2-LE"),
]

MONTAGE_PAIRS_TCP_LE_A_TUH_LIST = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


def tuh_eeg_load_complete_edf(
    edf_file: str,
    annotations_csv_file: str,
    plot_data: bool = False,
    montage: str = "standard_1020",
    apply_montage: bool = True,
    drop_unknown_channels: bool = False,
) -> mne.io.Raw:
    """
    Do not use.
    This function was used in earlier versions of the code. It does not work properly wheen you want to apply a montage.
    """

    # Load the EEG file
    raw = tuh_eeg_load_data_with_annotations(
        edf_file, annotations_csv_file, plot_data=plot_data
    )

    # change channel names to match standard_1020
    raw = tuh_eeg_change_channel_names(
        raw,
        montage=montage,
        apply_montage=apply_montage,
        drop_unknown_channels=drop_unknown_channels,
    )

    return raw


def tuh_eeg_change_channel_names(
    raw: mne.io.Raw,
    montage: str = "standard_1020",
    apply_montage: bool = True,
    drop_unknown_channels: bool = False,
) -> mne.io.Raw:
    """
    Change the channel names of the raw EEG data to match the standard 10-20 system and optionally apply a montage.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data in MNE Raw format.

    montage : str, optional
        The name of the montage to apply. Default is "standard_1020".

    apply_montage : bool, optional
        Whether to apply the specified montage to the raw EEG data. Default is True.

    drop_unknown_channels : bool, optional
        Whether to drop channels that do not match any in the standard 10-20 system. Default is False.

    Returns:
    --------
    raw_new : mne.io.Raw
        The raw EEG data with updated channel names and optionally applied montage.

    Example:
    --------
    >>> raw_new = tuh_eeg_change_channel_names(raw, montage='standard_1020', apply_montage=True)

    Notes:
    ------
    The function uses the 'rename_channels' and 'set_montage' methods from MNE to update channel names and apply the montage.
    """

    channel_names = raw.ch_names.copy()
    drop_list = []

    for i, channel_name in enumerate(channel_names):
        for standard_channel_name in STANDARD_10_20.ch_names:
            if standard_channel_name.lower() in channel_name.lower():
                logging.debug(
                    f"standard: {standard_channel_name}, tuh_channel: {channel_name}"
                )
                channel_names[i] = standard_channel_name
                break

        if drop_unknown_channels:
            found = False
            for standard_channel_name in STANDARD_10_20.ch_names:
                if standard_channel_name.lower() in channel_name.lower():
                    found = True
                    break
            if not found:
                drop_list.append(channel_name)

    # create a dictionary of old and new channel names
    channel_names_dict = dict(zip(raw.ch_names, channel_names))
    logging.debug(f"Channel names dictionary: {channel_names_dict}")

    # return a copy of raw with new channel names
    raw_new = raw.copy()
    raw_new.rename_channels(channel_names_dict)  # allow_duplicates=True ?

    # drop non standard_1020 channels
    if drop_unknown_channels:
        logging.info(f"Dropping {len(drop_list)} channels")
        for channel in drop_list:
            logging.debug(f"Dropping channel: {channel}")
            raw_new.drop_channels(channel)

    if apply_montage:
        logging.info(f"Applying {montage} montage")
        print(f"Applying {montage} montage")
        montage = mne.channels.make_standard_montage(montage)
        raw_new.set_montage(montage, on_missing="warn")
    else:
        logging.info("Not applying montage")
        print("Not applying montage")

    return raw_new


def tuh_eeg_load_data_with_annotations(
    edf_file: str, annotations_csv_file: str, plot_data: bool = False
) -> mne.io.Raw:
    """
    Load raw EEG data from an EDF file and add annotations from a CSV file.

    Parameters:
    -----------
    edf_file : str
        The path to the EDF file containing the raw EEG data.

    annotations_csv_file : str
        The path to the CSV file containing the annotations for events or seizures.

    plot_data : bool, optional
        Whether to plot the raw EEG data along with annotations for visual inspection.
        Default is False.

    Returns:
    --------
    raw_tuh : mne.io.Raw
        The raw EEG data in MNE Raw format, with annotations added.

    Example:
    --------
    >>> raw_tuh = tuh_eeg_load_data_with_annotations('path/to/edf_file.edf', 'path/to/annotations.csv')

    Notes:
    ------
    The function uses the 'read_raw_edf' method from MNE to load the EDF file and
    sets the annotations using the 'set_annotations' method.
    """
    # Load the EEG file
    raw_tuh = mne.io.read_raw_edf(edf_file, preload=True)

    # Load the events / seizures and add them to the raw object
    seiz_events_name = annotations_csv_file
    events_csv = pd.read_csv(seiz_events_name, skiprows=5, sep=",")
    #  header=None,
    #  names =['Start', 'End', 'Code', 'Certainty'])
    logging.debug(events_csv.head())
    # head: channel,start_time,stop_time,label,confidence

    # add the events to the raw object
    raw_tuh.set_annotations(
        mne.Annotations(
            events_csv["start_time"],
            events_csv["stop_time"] - events_csv["start_time"],
            events_csv["label"],
        )
    )

    if plot_data:
        # get duration of eeg data
        duration = raw_tuh.n_times / raw_tuh.info["sfreq"]
        if 100 < duration:
            logging.debug(
                "Duration is greater than 100 seconds. Start plot at 80 seconds."
            )
            start_plot = 80
        else:
            logging.debug(
                "Duration is less than 100 seconds. Start plot at 20% of duration."
            )
            start_plot = duration // 5

        # plot the raw data
        raw_tuh.plot(duration=10, n_channels=30, scalings=1e-4, start=start_plot)

    return raw_tuh


def tuh_eeg_downsample(
    raw: mne.io.Raw, new_sfreq: int = 100, plot_data: bool = False
) -> mne.io.Raw:
    """
    Downsample raw EEG data to a new sampling frequency.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data in MNE Raw format.

    new_sfreq : int, optional
        The new sampling frequency in Hz. Default is 100 Hz.

    plot_data : bool, optional
        Whether to plot the downsampled data for visual inspection. Default is False.

    Returns:
    --------
    raw_tuh_ds : mne.io.Raw
        The downsampled raw EEG data in MNE Raw format.

    Example:
    --------
    >>> raw_tuh_ds = tuh_eeg_downsample(raw, new_sfreq=100)

    Notes:
    ------
    The function uses the 'resample' method from MNE, with npad set to "auto".
    """

    # downsample the data
    raw_tuh_ds = raw.copy().resample(new_sfreq, npad="auto")

    if plot_data:
        # plot the downsampled data
        raw_tuh_ds.plot(duration=10, n_channels=30, scalings=1e-4, start=20)

    return raw_tuh_ds


def tuh_eeg_filter_data(
    tuh_mne, l_freq, h_freq, method="fir", fir_design="firwin2", verbose=False
):
    """
    Apply a bandpass filter to the raw EEG data.

    Parameters:
    -----------
    tuh_mne : mne.io.Raw
        The raw EEG data in MNE Raw format.

    l_freq : float
        The lower frequency limit for the bandpass filter.

    h_freq : float
        The higher frequency limit for the bandpass filter.

    method : str, optional
        The filtering method to use. Default is "fir" (Finite Impulse Response).

    fir_design : str, optional
        The FIR filter design method. Default is "firwin2".

    verbose : bool, optional
        Whether to print log messages during filtering. Default is False.

    Returns:
    --------
    filt_data : mne.io.Raw
        The raw EEG data after applying the bandpass filter.

    Example:
    --------
    >>> filt_data = tuh_eeg_filter_data(raw, l_freq=0.5, h_freq=40)

    Notes:
    ------
    The function uses the 'create_filter' and 'filter' methods from MNE to create and apply the filter.
    """
    sfreq = tuh_mne.info["sfreq"]
    filter_params = mne.filter.create_filter(
        tuh_mne.get_data(),
        sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design=fir_design,
        verbose=verbose,
    )
    filt_data = tuh_mne.filter(
        l_freq, h_freq, method=method, fir_design=fir_design, verbose=verbose
    )

    return filt_data


def tuh_eeg_apply_bipolar_montage(
    raw, filepath, drop_references=True, copy=True, only_return_bipolar=False
):
    """
    Apply a bipolar montage to a given MNE Raw object based on the filepath.

    Parameters:
    -----------
    raw : mne.io.Raw
        The MNE Raw object containing the EEG data.
    filepath : str
        The filepath of the EEG data file. Used to determine which montage to apply.
    drop_references : bool, optional
        Whether to drop the original reference channels after creating the bipolar channels.
        Default is True.
    copy : bool, optional
        Whether to operate on a copy of the data. Default is True.
    only_return_bipolar : bool, optional
        Whether to return only the bipolar channels. Default is False.

    Returns:
    --------
    mne.io.Raw
        The MNE Raw object with the applied bipolar montage. If `only_return_bipolar` is True,
        only the bipolar channels are returned.

    Notes:
    ------
    This function supports four types of montages based on the filepath:
    - "01_tcp_ar": Temporal Central Parasagittal Averaged Reference (TCP AR)
    - "02_tcp_le": Temporal Central Parasagittal Linked Ears Reference (TCP LE)
    - "03_tcp_ar_a": A less common TCP AR montage with 20 channels
    - "04_tcp_le_a": A less common TCP LE montage with 20 channels

    The function first changes the channel names to match the standard 10-20 system using
    `tuh_eeg_change_channel_names` function. Then, it applies the bipolar montage based on the
    filepath and returns the modified MNE Raw object.
    """

    # check what montage to use
    if "01_tcp_ar" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_AR
    elif "02_tcp_le" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_LE
    elif "03_tcp_ar_a" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_AR_A
    elif "04_tcp_le_a" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_LE_A
    else:
        print("No montage for this file")
        return raw

    # change channel names to match standard_1020
    tuh_data = tuh_eeg_change_channel_names(raw)

    # get the bipolar channel names
    bipolar_channel_names = [f"{pair[0]}--{pair[1]}" for pair in montage_pairs]

    # apply bipolar montage
    tuh_data_bipolar = mne.set_bipolar_reference(
        tuh_data,
        anode=[pair[0] for pair in montage_pairs],
        cathode=[pair[1] for pair in montage_pairs],
        ch_name=bipolar_channel_names,  # set channel names for bipolar channels
        drop_refs=drop_references,
        copy=copy,
    )

    if only_return_bipolar:
        # return only the bipolar channels
        tuh_data_bipolar.pick(
            bipolar_channel_names
        )  # pick_channels(bipolar_channel_names)
        return tuh_data_bipolar
    else:
        return tuh_data_bipolar


def tuh_eeg_apply_TUH_bipolar_montage(
    raw, filepath, drop_references=True, copy=True, only_return_bipolar=True
):
    # check what montage to use
    if "01_tcp_ar" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_AR_TUH
        montage_names = MONTAGE_PAIRS_TCP_AR_TUH_LIST
    elif "02_tcp_le" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_LE_TUH
        montage_names = MONTAGE_PAIRS_TCP_LE_TUH_LIST
    elif "03_tcp_ar_a" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_AR_A_TUH
        montage_names = MONTAGE_PAIRS_TCP_AR_A_TUH_LIST
    elif "04_tcp_le_a" in filepath:
        montage_pairs = MONTAGE_PAIRS_TCP_LE_A_TUH
        montage_names = MONTAGE_PAIRS_TCP_LE_A_TUH_LIST
    else:
        print("No montage for this file")
        return None

    tuh_data_bipolar = mne.set_bipolar_reference(
        raw,
        anode=[pair[0] for pair in montage_pairs],
        cathode=[pair[1] for pair in montage_pairs],
        ch_name=montage_names,
        drop_refs=drop_references,
        copy=copy,
    )
    if only_return_bipolar:
        tuh_data_bipolar.pick(montage_names)
        return tuh_data_bipolar
    else:
        return tuh_data_bipolar


def tuh_eeg_max_local_scale_data(
    epochs: mne.epochs.Epochs, feature_range: tuple = (-1, 1), per_channel: bool = False
):
    """
    Scales EEG epoch data using Min-Max scaling.

    Parameters:
    -----------
    epochs : mne.epochs.Epochs
        The EEG epochs to be scaled. Should be an instance of mne.epochs.Epochs.

    feature_range : tuple, optional
        The range of scaled values. Default is (-1, 1).

    per_channel : bool, optional
        Whether to scale each channel separately. Default is False.
        If False, the Min-Max scaling is applied across all channels.
        If True, each channel is scaled individually.

    Returns:
    --------
    np.array
        A NumPy array containing the scaled epochs.

    Example:
    --------
    TODO

    Notes:
    ------
    - The function uses sklearn's MinMaxScaler for the scaling operation.
    - When `per_channel` is set to True, each channel is scaled individually, which might be useful for certain EEG analyses.
    """

    scaled_epochs = []

    # copy the epochs
    epochs_copy = epochs.copy()

    if not per_channel:
        for epoch in epochs_copy:
            scaler = MinMaxScaler(feature_range=feature_range)
            epoch_scaled = scaler.fit_transform(epoch)

            scaled_epochs.append(epoch_scaled)

    else:
        # scale each channel separately
        for epoch in epochs_copy:
            epoch_scaled = []
            for channel in epoch:
                scaler = MinMaxScaler(feature_range=feature_range)
                channel_scaled = scaler.fit_transform(channel.reshape(-1, 1))
                epoch_scaled.append(channel_scaled)
            epoch_scaled = np.array(epoch_scaled)
            scaled_epochs.append(epoch_scaled)

        # reshape from (20, 300, 1) to (20, 300)
        scaled_epochs = np.array(scaled_epochs).reshape(epochs_copy.get_data().shape)

    return np.array(scaled_epochs)


def tuh_eeg_create_epochs_with_labels(
    tuh_data: mne.io.Raw,
    window_length: int = 6,
    overlap=0,
    drop_bad_epochs=True,
    verbose=False,
):
    """
    Create fixed-length epochs from raw EEG data and associate labels based on seizure annotations.

    Parameters:
    -----------
    tuh_data : mne.io.Raw
        The raw EEG data in MNE Raw format.

    window_length : int, optional
        The length of each epoch window in seconds. Default is 6 seconds.

    overlap : float, optional
        The overlap between consecutive epochs in seconds. Default is 0.

    drop_bad_epochs : bool, optional
        Whether to drop epochs that are marked as bad. Default is True.

    verbose : bool, optional
        Whether to print debug information. Default is False.

    Returns:
    --------
    fixed_epochs : mne.Epochs
        The fixed-length epochs created from the raw EEG data.

    labels : numpy.ndarray
        A boolean array where each element corresponds to an epoch. True indicates the presence of a seizure, and False indicates its absence.

    Raises:
    -------
    AssertionError
        If the number of labels does not match the number of epochs.

    Example:
    --------
    >>> fixed_epochs, labels = tuh_eeg_create_epochs_with_labels(tuh_data)
    """

    # # create fixed length epochs
    # fixed_epochs = mne.make_fixed_length_epochs(
    #     raw=tuh_data, duration=window_length, overlap=overlap
    # )

    # if drop_bad_epochs:
    #     fixed_epochs = fixed_epochs.drop_bad()  # I don't know if this is necessary

    # logging.debug(f"Type of fixed_epochs: {type(fixed_epochs)}")

    # # Map annotations to epochs
    # labels = []
    # labels_events = []
    # logging.debug("Annotations of file:")
    # if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
    #     for ann in tuh_data.annotations:
    #         logging.debug(f"onset: {ann['onset']}, duration: {ann['duration']}")
    #         logging.debug(f"{ann}")

    # for start, stop in zip(
    #     fixed_epochs.events[:, 0],
    #     fixed_epochs.events[:, 0] + window_length * fixed_epochs.info["sfreq"],
    # ):
    #     logging.debug(f"start: {start}, stop: {stop}")
    #     logging.debug(f"tuh_data.info['sfreq']: {tuh_data.info['sfreq']}")
    #     logging.debug(f"tuh_data.annotations: {tuh_data.annotations}")
    #     is_event = any(
    #         (ann["onset"] * tuh_data.info["sfreq"] <= stop)
    #         and ((ann["onset"] + ann["duration"]) * tuh_data.info["sfreq"] >= start)
    #         for ann in tuh_data.annotations
    #     )
    #     labels.append(is_event)

    # # convert labels to numpy array
    # labels = np.array(labels)

    # # assert that the number of labels is equal to the number of epochs
    # assert len(labels) == len(
    #     fixed_epochs
    # ), f"ERROR: Number of labels is not equal to the number of epochs. len(labels) = {len(labels)}, len(fixed_epochs) = {len(fixed_epochs)}"

    # return fixed_epochs, labels

    # create fixed length epochs
    fixed_epochs = mne.make_fixed_length_epochs(
        raw=tuh_data, duration=window_length, overlap=overlap
    )

    if drop_bad_epochs:
        fixed_epochs = fixed_epochs.drop_bad()

    logging.debug(f"Type of fixed_epochs: {type(fixed_epochs)}")

    # Map annotations to epochs
    labels = []
    labels_events = []
    logging.debug("Annotations of file:")
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        for ann in tuh_data.annotations:
            logging.debug(f"onset: {ann['onset']}, duration: {ann['duration']}")
            logging.debug(f"{ann}")

    for start, stop in zip(
        fixed_epochs.events[:, 0],
        fixed_epochs.events[:, 0] + window_length * fixed_epochs.info["sfreq"],
    ):
        event_description = "No Event"
        for ann in tuh_data.annotations:
            if (ann["onset"] * tuh_data.info["sfreq"] <= stop) and (
                (ann["onset"] + ann["duration"]) * tuh_data.info["sfreq"] >= start
            ):
                event_description = ann["description"]
                break
        labels.append(event_description)

    # convert labels to numpy array
    labels = np.array(labels)

    # assert that the number of labels is equal to the number of epochs
    assert len(labels) == len(
        fixed_epochs
    ), f"ERROR: Number of labels is not equal to the number of epochs. len(labels) = {len(labels)}, len(fixed_epochs) = {len(fixed_epochs)}"

    return fixed_epochs, labels


def tuh_eeg_create_heatmap_from_epoch(
    epoch,
    figsize=(15, 6),
    cmap="viridis",
    title="Heatmap of one epoch",
    save=False,
    save_path=".",
    xlabel="Time",
    ylabel="Channel",
    colorbar_label="Value",
    aspect="auto",
    show_plot=True,
):
    """
    Creates and displays a heatmap for a given EEG epoch.

    Parameters:
    -----------
    epoch : np.array
        The EEG epoch data to be visualized. Should be a 2D numpy array.

    figsize : tuple, optional
        The size of the figure for the heatmap. Default is (15, 6).

    cmap : str, optional
        The colormap to use for the heatmap. Default is "viridis".

    title : str, optional
        The title for the heatmap. Default is "Heatmap of one epoch".

    save : bool, optional
        Whether to save the heatmap as a PNG file. Default is False.

    save_path : str, optional
        The directory where the heatmap will be saved if `save` is True. Default is the current directory.

    xlabel : str, optional
        The label for the x-axis. Default is "Time".

    ylabel : str, optional
        The label for the y-axis. Default is "Channel".

    colorbar_label : str, optional
        The label for the colorbar. Default is "Value".

    aspect : str, optional
        Controls the aspect ratio of the axes. Default is "auto".

    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns:
    --------
    None

    Example:
    --------
    >>> tuh_eeg_create_heatmap_from_epoch(epoch_data, figsize=(10, 5), cmap="plasma", title="EEG Heatmap")
    """
    plt.figure(figsize=figsize)
    plt.imshow(epoch, aspect=aspect, cmap=cmap)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_plot:
        plt.show()
    if save:
        plt.savefig(f"{save_path}/{title}.png")


def tuh_eeg_to_greyscale(
    tuh_data_scaled_epoch, scaling=(-1, 1), plot=False, title="Greyscale of one epoch"
):
    logging.debug(
        f"tuh_eeg_to_greyscale with shape: {tuh_data_scaled_epoch.shape}, type: {type(tuh_data_scaled_epoch)}, plot: {plot}"
    )

    # convert to greyscale
    tuh_greyscaled = np.interp(tuh_data_scaled_epoch, scaling, (0, 255))

    # convert data to uint8
    tuh_greyscaled = tuh_greyscaled.astype(np.uint8)

    if plot:
        plt.figure(figsize=(15, 6))
        plt.imshow(tuh_greyscaled, cmap="gray", aspect="auto")
        plt.colorbar(label="Greyscale Value")
        plt.title(title)
        plt.xlabel("Timestamps (Seconds * Sampling Frequency)")
        plt.ylabel("Channel")
        plt.show()

    return tuh_greyscaled


def tuh_eeg_resize_data(
    tuh_data_greyscaled_epoch,
    new_shape=(224, 224),
    interpolation="lanczos",
    leave_blank=False,
) -> tuple[np.array, np.array]:
    """
    Resizes the given greyscaled EEG epoch data to a new shape.

    Parameters:
    -----------
    tuh_data_greyscaled_epoch : np.array
        The greyscaled EEG epoch data to be resized. Should be a 2D numpy array.

    new_shape : tuple, optional
        The new shape to resize the image to. Default is (256, 256).

    interpolation : str, optional
        The interpolation algorithm to use for resizing.
        Options are "cubic" for bicubic interpolation and "lanczos" for Lanczos interpolation.
        Default is "lanczos".

    leave_blank : bool, optional
        If True, the function will create a blank image of size `new_shape` and place the
        greyscaled data at the top-left corner. If False, the function will resize the image to `new_shape`.
        Default is False.

    Returns:
    --------
    np.array
        The resized EEG epoch data as a 2D numpy array.

    Example:
    --------
    >>> resized_data = tuh_eeg_resize_data(greyscaled_data, new_shape=(256, 256), interpolation="cubic")
    """

    logging.debug(
        f"tuh_eeg_resize_data with shape: {tuh_data_greyscaled_epoch.shape}, type: {type(tuh_data_greyscaled_epoch)}"
    )
    logging.debug(f"New shape: {new_shape}, interpolation: {interpolation}")

    # resize the data
    if leave_blank:
        # create a blank image of size new_shape
        blank_image = np.zeros(new_shape, dtype=np.uint8)
        # place the greyscale data at the top-left corner
        blank_image[
            : tuh_data_greyscaled_epoch.shape[0], : tuh_data_greyscaled_epoch.shape[1]
        ] = tuh_data_greyscaled_epoch
        # return
        return blank_image

    # Convert  numpy array to a PIL Image
    image = Image.fromarray((tuh_data_greyscaled_epoch).astype(np.uint8))

    if interpolation == "cubic":
        # resize the image
        resized_image = image.resize(new_shape, resample=Image.BICUBIC)

    elif interpolation == "lanczos":
        # resize the image
        resized_image = image.resize(new_shape, resample=Image.LANCZOS)

    return resized_image


def check_if_only_bckg_in_eeg_recording(annotation):
    file = pd.read_csv(annotation, skiprows=5)
    # check if in the label column only the background label is present
    if len(file["label"].unique()) == 1:
        if file["label"].unique()[0] == "bckg":
            return True
    return False


if __name__ == "__main__":
    print("This is a utility module.")
    pass
