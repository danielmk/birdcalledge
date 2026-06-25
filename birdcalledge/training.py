# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:05:40 2026

@author: Daniel
"""
import numpy as np
import torch
from rockpool.timeseries import TSEvent
import pdb

rng = np.random.default_rng()

def build_all_rasters(train, t_stop, dt, size_in):
    n = train.spike_times.nrows
    n_steps = int(t_stop / dt)

    rasters_np = np.zeros((n, n_steps, size_in), dtype=np.float32)

    for i in range(n):
        event = TSEvent(
            times=train.spike_times[i],
            channels=train.spike_channels[i],
            t_start=0.0,
            t_stop=t_stop
        )

        rasters_np[i] = event.raster(
            dt, t_start=0.0, t_stop=t_stop, add_events=True
        )

    return torch.from_numpy(rasters_np)

def build_all_rasters_new(train, t_stop, dt, size_in):
    n = train.spike_times.nrows

    first_event = TSEvent(
        times=train.spike_times[0],
        channels=train.spike_channels[0],
        t_start=0.0,
        t_stop=t_stop
    )

    first_raster = first_event.raster(dt, t_start=0.0, t_stop=t_stop, add_events=True)
    n_steps = first_raster.shape[0]

    rasters_np = np.zeros((n, n_steps, size_in), dtype=np.float32)
    rasters_np[0] = first_raster

    for i in range(1, n):
        event = TSEvent(
            times=train.spike_times[i],
            channels=train.spike_channels[i],
            t_start=0.0,
            t_stop=t_stop
        )
        rasters_np[i] = event.raster(dt, t_start=0.0, t_stop=t_stop, add_events=True)

    return torch.from_numpy(rasters_np)

def build_all_labels(train, species, t_stop, dt, size_out, label_amplitude=1.0):
    n = train.samples.nrows
    n_steps = int(t_stop / dt)

    labels = torch.zeros((n, n_steps, size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        start = int(1 / dt)
        stop = start + int(sample["call_duration"] / dt)
        labels[i, start:stop, 0] = label_amplitude

    return labels

def build_all_labels_new(train, species, n_steps, dt, size_out, label_amplitude=1.0):
    n = train.samples.nrows

    labels = torch.zeros((n, n_steps, size_out), dtype=torch.float32)

    for i, sample in enumerate(train.samples):
        if species[i] == "None":
            continue

        start = int(1 / dt)
        stop = start + int(sample["call_duration"] / dt)

        stop = min(stop, n_steps)  # ✅ prevent overflow

        labels[i, start:stop, 0] = label_amplitude

    return labels

def sample_batch(batch_size, signal_idx, noise_idx):
    half = batch_size // 2

    sig = rng.choice(signal_idx, size=half, replace=False)
    noi = rng.choice(noise_idx, size=half, replace=False)

    idx = np.concatenate([sig, noi])
    rng.shuffle(idx)

    return torch.as_tensor(idx)

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    loss,
    extra=None
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)

