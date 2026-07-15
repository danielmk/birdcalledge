# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.networks import SynNet
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import birdcalledge

results_dir = Path(__file__).parent.parent / 'data'

matrix_path = {"synnetqatv2 Pretraining":  results_dir / 'confusion_metric_synnetqatv2_pretraining_threshold_checkpoint.npz',
               "synnetqatv3 Pretraining":  results_dir / 'confusion_metric_synnetqatv3_pretraining_threshold_checkpoint.npz',
               "synnetqatv2 QAT": results_dir / 'confusion_metric_synnetqatv2_from_checkpoint_2000_threshold_checkpoint.npz',
               "synnetqatv3 QAT": results_dir / 'confusion_metric_synnetqatv3_from_checkpoint_2000_threshold_checkpoint.npz'}

CONFUSION_KEYS = [
    "tpr", "fnr",
    "tnr", "fpr",
    "precision", "fdr",
    "accuracy", "balanced_accuracy",
    "TP", "TN", "FP", "FN",
]

training_dict = {}
test_dict = {}

matrix_dict = {}

for net in matrix_path.keys():
    training_dict[net] = {}
    test_dict[net] = {}
    matrix_dict[net] = np.load(matrix_path[net], allow_pickle=True)
    for k in CONFUSION_KEYS:
        training_dict[net][k] = np.array([x[k] for x in matrix_dict[net]['training_metrics']])
        test_dict[net][k] = np.array([x[k] for x in matrix_dict[net]['test_metrics']])


"""PLOTTING"""
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 18

"""COLORMAPS BALANCED ACCURACY"""
fig, ax = plt.subplots(1, 4)

max_ba = {}
for idx, k in enumerate(matrix_path.keys()):
    im = ax[idx].pcolormesh(
        matrix_dict[k]['thresholds'],
        matrix_dict[k]['epochs'],
        test_dict[k]['balanced_accuracy'],
        shading="nearest",
        vmin=0.5,
        vmax=1.0,
        cmap="viridis")
    
    
    ax[idx].set_xlabel("Threshold")
    ax[idx].set_ylabel("Epoch")
    ax[idx].set_title(k)
    



cbar = fig.colorbar(
    im,
    ax=ax,              # <-- pass ALL axes
    orientation="vertical",
    shrink=0.9,
    pad=0.02
)
cbar.set_label("Balanced Test Accuracy")

fig, ax = plt.subplots(1)

for idx, k in enumerate(matrix_path.keys()):
    ax.plot(test_dict[k]['fpr'][-1,:], test_dict[k]['tpr'][-1,:], marker='o', color=birdcalledge.config.colors_diverging[idx])

ax.legend(matrix_path.keys())
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_ylim((0, 1.0))
ax.set_xlim((0, 1.0))

fig, ax = plt.subplots(1)

for idx, k in enumerate(matrix_path.keys()):
    ax.scatter(test_dict[k]['fpr'], test_dict[k]['tpr'], marker='o', color=birdcalledge.config.colors_diverging[idx])

ax.legend(matrix_path.keys())
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_ylim((0, 1.0))
ax.set_xlim((0, 1.0))

"""FIND BEST CHECKPOINTS"""
maxima = {}
for net in matrix_path.keys():
    maxima[net] = {}
    for k in CONFUSION_KEYS:
        flat_idx = test_dict[net][k].argmax()
        row, col = np.unravel_index(flat_idx, test_dict[net][k].shape)
        maxima[net][k] = np.array([row, col])

# BEST BALANCED ACCURACY
best_dict = {"net": [],
           "Balanced Accuracy": [],
           "True Positive Rate": [],
           "False Positive Rate": [],
           "Epoch": [],
           "Threshold": []}

crit = 'balanced_accuracy'
for net in maxima.keys():
    best_dict['net'].append(net)
    best_dict['Balanced Accuracy'].append(test_dict[net]['balanced_accuracy'][maxima[net]['balanced_accuracy'][0], maxima[net]['balanced_accuracy'][1]])
    best_dict['True Positive Rate'].append(test_dict[net]['tpr'][maxima[net]['balanced_accuracy'][0], maxima[net]['balanced_accuracy'][1]])
    best_dict['False Positive Rate'].append(test_dict[net]['fpr'][maxima[net]['balanced_accuracy'][0], maxima[net]['balanced_accuracy'][1]])
    best_dict['Epoch'].append(matrix_dict[net]['epochs'][maxima[net]['balanced_accuracy'][0]])
    best_dict['Threshold'].append(matrix_dict[net]['thresholds'][maxima[net]['balanced_accuracy'][1]])


# test_dict['synnet-long']['balanced_accuracy'].argmax()
# row, col = np.unravel_index(flat_idx, test_dict['synnet-long']['balanced_accuracy'].shape)

# fig.colorbar(im, ax=ax[0], label="Balanced accuracy")

fig, ax = plt.subplots(
    ncols=2,
    figsize=(10.0, 5.8),
    sharex=True,
    constrained_layout=True,
)


epochs=[0, 2, 4, 6, 8]
for idx, epoch in enumerate(epochs):
    ax[0].plot(matrix_dict['synnetqatv2 Pretraining']['thresholds'], matrix_dict['synnetqatv2 Pretraining']['training_metrics'][epoch]['balanced_accuracy'], marker='o', color=birdcalledge.config.colors_diverging[idx], label=f"Epoch {matrix_dict['synnetqatv2 Pretraining']['epochs'][epoch]}")
    
ax[0].set_ylabel("Training Balanced Accuracy")


epochs=[0, 2, 4, 6, 8]
for idx, epoch in enumerate(epochs):
    plt.plot(matrix_dict['synnetqatv2 Pretraining']['thresholds'], matrix_dict['synnetqatv2 Pretraining']['test_metrics'][epoch]['balanced_accuracy'], marker='o', color=birdcalledge.config.colors_diverging[idx], label=f"Epoch {matrix_dict['synnetqatv2 Pretraining']['epochs'][epoch]}")
    
legend = [f"Epoch {matrix_dict['synnetqatv2 Pretraining']['epochs'][x]}" for x in epochs]
for a in ax.flatten():
    a.legend(legend)
    a.set_xlim((0.4, 1.6))
    a.set_ylim((0.4, 1.0))
    a.set_xlabel("Threshold")
    
ax[1].set_ylabel("Test Balanced Accuracy")




