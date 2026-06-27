# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tables
from matplotlib.animation import FuncAnimation, FFMpegWriter
import soundfile as sf
from rockpool.nn.networks import SynNet
import torch
import birdcalledge
from pathlib import Path

dataset_path = Path(__file__).parent.parent / 'data' / 'dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

high_quality = np.argwhere(dst.root.train.quality_rating.read() == 3)[:, 0]

example_idx = high_quality[2]

sr=44100

device = torch.device('cpu')

test = dst.root.test

q = test.quality_rating[:]
species = test.samples.col("species")
species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

signal_idx = np.where(
    (q > 1) & (species != "None")
)[0]

noise_idx = np.where(
    species == "None"
)[0]

rng = np.random.default_rng()

"""HYPERPARAMETERS"""
t_stop=2.504

# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------

net = birdcalledge.nets.synnetqatv1(output='spikes')

ckpt_dir = Path(r"C:\Users\Daniel\repos\xylo\scripts\results\checkpoints")

ckpt_dir = Path(__file__).parent.parent / 'data' / 'checkpoints'

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and "synnetqatv2_pretraining_" in p.name
)

synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location=device).get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location=device)
    for path in synnet_ckpts
]

epoch = 2000

curr_ckpt = [x for x in checkpoints if x['epoch'] == epoch][0]

net.load_state_dict(curr_ckpt["model_state"])
net.eval()

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

print("Building rasters...")
all_rasters = birdcalledge.training.build_all_rasters_new(test, t_stop, net.dt, net.size_in)

# Move **once**
all_rasters = all_rasters.to(device)

output, _, out3 = net(all_rasters, record=True)

t = np.arange(0, 2.5, 1/sr)

t_spikes = np.arange(0, t_stop, net.dt)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(
    nrows=3,
    figsize=(10.0, 5.8),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [1, 1, 1]}
)

audio = dst.root.train.audio[example_idx]

ax[0].plot(t, audio, color='k', linewidth=0.5)
ax[0].set_ylabel("Raw Audio")
# ax[0].set_xticklabels([])

D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
img = librosa.display.specshow(D, y_axis='log', x_axis='s', sr=sr, ax=ax[1])
ax[1].set_xlabel("")
# ax[1].set_xticklabels([])

spike_times = np.argwhere(output[example_idx, :, 0]) * net.dt
ax[2].plot(t_spikes, out3['out_neurons']['vmem'].detach()[example_idx, :, 0], color='k')
ax[2].vlines(spike_times, ymin=1.3, ymax=1.4, color='r')
# ax[2].vlines(spike_times, ymin=2.5, ymax=2.7, color='r')

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Output Voltage")

fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

for i in range(3):
    ax[i].set_xlim((0, 2.5))

# Exclude long calls
# precise_intervals = precise_intervals[(precise_intervals[:, 1] - precise_intervals[:, 0]) <= 2, :]

fig, ax = plt.subplots(
    nrows=2,
    figsize=(10.0, 5.8),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [1, 1]}
)

ax[0].plot(t_spikes, out3['1_neurons']['vmem'].detach().numpy()[0,:,-1], color='k', linewidth=0.5)
ax[0].set_ylabel("Membrane Voltage")

ax[1].plot(t_spikes, out3['1_neurons']['isyn'].detach().numpy()[0,:,-1], color='k', linewidth=0.5)
ax[1].set_ylabel("Synaptic Input")
