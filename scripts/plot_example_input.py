# -*- coding: utf-8 -*-
"""
Plot a representative audio sample with spectrum and time stamps.

Used in Figure01.
"""
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tables
from pathlib import Path


dataset_path = Path(__file__).parent.parent / 'data' / 'dataset_split.h5'
dst = tables.open_file(dataset_path, mode="r")

high_quality = np.argwhere(dst.root.train.quality_rating.read() == 3)[:, 0]
noise_samples = np.argwhere(dst.root.train.samples.read()['species'] == b'None')[:, 0]

kingfisher_idx = high_quality[2]
noise_idx = noise_samples[0]

sr = 44100
t = np.arange(0, 2.5, 1/sr)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'Arial'


fig, ax = plt.subplots(
    nrows=4,
    ncols=2,
    figsize=(10.0, 5.8),
    sharex=True,
    constrained_layout=True,
    gridspec_kw={'height_ratios': [1, 1, 1, 0.1]}
)


def plot_example(example_idx, col):
    audio = dst.root.train.audio[example_idx]
    spike_times = dst.root.train.spike_times[example_idx]
    spike_channels = dst.root.train.spike_channels[example_idx]

    # raw audio
    ax[0, col].plot(t, audio, color='k', linewidth=0.5)
    ax[0, col].set_ylabel("Raw Audio" if col == 0 else "")

    # spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(
        D,
        y_axis='log',
        x_axis='time',
        sr=sr,
        ax=ax[1, col],
        cmap='magma',
        vmin=-80,        
        vmax=0

    )
    ax[1, col].set_aspect('auto')

    # spikes
    ax[2, col].scatter(
        spike_times,
        spike_channels,
        marker="|",
        color='k',
        alpha=0.8,
        linewidth=0.5,
        vmin=-80,       
        vmax=0
    )

    ax[2, col].set_ylabel("# Neuron" if col == 0 else "")

    # label
    call_duration = dst.root.train.samples.read()['call_duration'][example_idx]
    training_label = np.zeros(len(t))
    if dst.root.train.samples.read()['species'][example_idx] == b'Ruddy Kingfisher':
        training_label[(t > 1) & (t <= 1 + call_duration)] = 1

    ax[3, col].plot(t, training_label, color='k')
    ax[3, col].set_xlabel("Time (s)")
    ax[3, col].set_ylim((-0.05, 1.05))

    return img


# ✅ plot both columns
img_left = plot_example(kingfisher_idx, col=0)
img_right = plot_example(noise_idx, col=1)

# ✅ titles
ax[0, 0].set_title("Kingfisher")
ax[0, 1].set_title("Noise")

# ✅ consistent limits
for i in range(4):
    for j in range(2):
        ax[i, j].set_xlim((0, 2.5))

fig.colorbar(img_right, ax=ax[1, 1], format="%+2.0f dB")
