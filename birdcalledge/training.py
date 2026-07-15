# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:05:40 2026

@author: Daniel
"""
import numpy as np
import torch
from rockpool.timeseries import TSEvent
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
import tables
import pdb

_DATA_DIR = Path(__file__).parent.parent / 'data'
_CHECKPOINTS_DIR = _DATA_DIR / 'checkpoints'
_LOSS_DIR = _DATA_DIR / 'loss'
_DATASET_PATH = _DATA_DIR / 'dataset_split.h5'


@dataclass
class TrainingConfig:
    run_name: str
    net_fn: Callable
    n_epochs: int = 2001
    batch_size: int = 512
    t_stop: float = 2.504
    lr: float = 1e-4
    optimizer: str = 'AdamW'      # 'Adam' or 'AdamW'
    weight_decay: float = 0.0
    checkpoint_path: Optional[str] = None  # filename relative to checkpoints dir; if set, load before training
    qat_enabled: bool = False
    qat_warmup: int = 0           # epochs over which qat_alpha ramps from 0 to 1; 0 means no ramp
    grad_clip: Optional[float] = None
    label_amplitude: float = 1.0
    checkpoint_every: int = 50
    torch_seed: int = 65
    numpy_seed: int = 68
    quality_threshold: int = 1    # signal samples require quality_rating > quality_threshold

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items()}
        d['net_fn'] = self.net_fn.__name__
        return d


def train(cfg: TrainingConfig):
    torch.manual_seed(cfg.torch_seed)
    np.random.seed(cfg.numpy_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dst = tables.open_file(_DATASET_PATH, mode="r")
    train_data = dst.root.train

    q = train_data.quality_rating[:]
    species = train_data.samples.col("species")
    species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

    signal_idx = np.where((q > cfg.quality_threshold) & (species != "None"))[0]
    noise_idx = np.where(species == "None")[0]

    net = cfg.net_fn(output='vmem')

    if cfg.checkpoint_path is not None:
        ckpt = torch.load(_CHECKPOINTS_DIR / cfg.checkpoint_path, map_location=device)
        net.load_state_dict(ckpt['model_state'])
    else:
        with torch.no_grad():
            net.seq['0_linear'].weight.data /= 2

    net.qat_enabled = cfg.qat_enabled
    net.qat_alpha = 0.0 if cfg.qat_warmup > 0 else 1.0

    net.to(device)
    net.train()

    all_rasters = build_all_rasters_new(train_data, cfg.t_stop, net.dt, net.size_in).to(device)
    n_steps = all_rasters.shape[1]
    all_labels = build_all_labels_new(train_data, species, n_steps, net.dt, net.size_out,
                                      label_amplitude=cfg.label_amplitude).to(device)

    dst.close()

    if cfg.optimizer == 'AdamW':
        from torch.optim import AdamW
        optimizer = AdamW(net.parameters().astorch(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        from torch.optim import Adam
        optimizer = Adam(net.parameters().astorch(), lr=cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    loss_fun = torch.nn.MSELoss().to(device=device)

    loss_t = []
    for epoch in range(cfg.n_epochs):
        if cfg.qat_warmup > 0:
            net.qat_alpha = min(1.0, epoch / cfg.qat_warmup)

        batch_idc = sample_batch(cfg.batch_size, signal_idx, noise_idx)
        rasters, labels = all_rasters[batch_idc], all_labels[batch_idc]

        optimizer.zero_grad()
        output, _, _ = net(rasters, record=False)
        output = output.to(device)

        loss = loss_fun(output, labels)
        this_loss = loss.item()

        if epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                _CHECKPOINTS_DIR / f'{cfg.run_name}_epoch_{epoch:04d}.pt',
                net, optimizer, epoch, this_loss,
                extra={'config': cfg.to_dict()},
            )

        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters().astorch(), cfg.grad_clip)

        optimizer.step()
        loss_t.append(this_loss)

        print(f'Epoch: {epoch} | Loss: {this_loss} | Global scale: {net.global_scale} | Output scale: {net.output_scale}')

    _LOSS_DIR.mkdir(exist_ok=True)
    np.save(_LOSS_DIR / f'{cfg.run_name}_loss.npy', loss_t)

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

