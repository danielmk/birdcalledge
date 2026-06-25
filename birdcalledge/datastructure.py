# -*- coding: utf-8 -*-
"""
Utilities for creating an empty PyTables HDF5 dataset
for machine learning training data.

Schema:
    /dataset
        ├── samples        (Table)
        ├── audio           (EArray)
        ├── spike_times     (VLArray)
        └── spike_channels  (VLArray)
"""

import tables
import numpy as np


class SampleMeta(tables.IsDescription):
    """
    Fixed-size per-sample metadata.
    """
    call_duration    = tables.Float64Col()
    confidence = tables.Float32Col()

    t_start = tables.Float64Col()
    t_stop  = tables.Float64Col()

    species  = tables.StringCol(64)
    filename = tables.StringCol(128)
    
    # SAMPle ExTRACTION PARAMETERS
    confidence_threshold = tables.Float32Col()
    f_low = tables.Float32Col()
    f_high = tables.Float32Col()
    bandpass_order = tables.Int16Col()
    threshold_k = tables.Float32Col()
    time_pre = tables.Float32Col()
    time_post = tables.Float32Col()    
    
    sr = tables.Float64Col()


def create_empty_dataset(
    h5_path: str,
    audio_length: int,
    group_name: str = "train",
    audio_dtype=tables.Float32Atom(),
    compression_level: int = 5,
    compression_lib: str = "blosc",
):
    """
    Create an empty HDF5 file with the standard dataset layout.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file to create.
        Existing files will be overwritten.

    audio_length : int
        Length (number of samples) of each raw audio array.

    group_name : str, optional
        Name of the top-level group (default: "dataset").

    audio_dtype : tables.Atom, optional
        Atom type used for storing audio (default: Float32).

    compression_level : int, optional
        Compression level, 0–9 (default: 5).

    compression_lib : str, optional
        Compression library (default: "blosc").
    """

    filters = tables.Filters(
        complevel=compression_level,
        complib=compression_lib
    )

    with tables.open_file(h5_path, mode="w") as h5:
        # Create top-level group
        group = h5.create_group("/", group_name)

        # Metadata table
        h5.create_table(
            group,
            "samples",
            description=SampleMeta,
            filters=filters
        )

        # Raw audio: growable array (one row per sample)
        h5.create_earray(
            group,
            "audio",
            atom=audio_dtype,
            shape=(0, audio_length),
            filters=filters
        )

        # Variable-length spike information
        h5.create_vlarray(
            group,
            "spike_times",
            atom=tables.Float64Atom(),
            filters=filters
        )

        h5.create_vlarray(
            group,
            "spike_channels",
            atom=tables.Int16Atom(),
            filters=filters
        )


def append_sample(
    h5file: tables.File,
    call_duration: float,
    confidence: float,
    t_start: float,
    t_stop: float,
    species: str,
    filename: str,
    confidence_threshold: float,
    f_low: float,
    f_high: float,
    bandpass_order: int,
    threshold_k: float,
    time_pre: float,
    time_post: float,
    sr: float,
    audio: np.ndarray,
    spike_times: np.ndarray,
    spike_channels: np.ndarray,
    group_name: str = "train",
):
    """
    Append a single sample to the dataset.
    """

    group = h5file.get_node(f"/{group_name}")

    samples_tbl = group.samples
    audio_arr = group.audio
    spike_times_arr = group.spike_times
    spike_channels_arr = group.spike_channels
    quality_rating_arr = group.quality_rating

    # ---- Validation -------------------------------------------------------

    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("audio must be a 1D array")

    if audio.shape[0] != audio_arr.shape[1]:
        raise ValueError(
            f"audio length {audio.shape[0]} does not match "
            f"dataset audio length {audio_arr.shape[1]}"
        )

    spike_times = np.asarray(spike_times, dtype=np.float64)
    spike_channels = np.asarray(spike_channels, dtype=np.int16)

    if spike_times.shape != spike_channels.shape:
        raise ValueError(
            "spike_times and spike_channels must have the same shape"
        )

    # ---- Append sample table ----------------------------------------------

    row = samples_tbl.row
    row["call_duration"] = call_duration
    row["confidence"] = confidence
    row["t_start"] = t_start
    row["t_stop"] = t_stop
    row["species"] = species
    row["filename"] = filename

    row["confidence_threshold"] = confidence_threshold
    row["f_low"] = f_low
    row["f_high"] = f_high
    row["bandpass_order"] = bandpass_order
    row["threshold_k"] = threshold_k
    row["time_pre"] = time_pre
    row["time_post"] = time_post

    row["sr"] = sr

    row.append()
    samples_tbl.flush()

    # ---- Append arrays ----------------------------------------------------

    audio_arr.append(audio[np.newaxis, :].astype(audio_arr.atom.dtype))
    spike_times_arr.append(spike_times)
    spike_channels_arr.append(spike_channels)
    quality_rating_arr.append(np.array([np.nan], dtype=np.float32))
