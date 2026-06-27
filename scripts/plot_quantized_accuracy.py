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

matrix_path = {"synnetqatv2 Pretraining":  results_dir / 'synnetqatv2_pretraining_checkpoint_epoch_2000_confusion_metric_quantized_xylosim.npz',
               "synnetqatv2 QAT": results_dir / 'synnetqatv2_from_checkpoint_2000_epoch_2000_confusion_metric_quantized_xylosim.npz',}


matrix_dict = {}

for net in matrix_path.keys():
    matrix_dict[net] = np.load(matrix_path[net], allow_pickle=True)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(
    ncols=1,
    figsize=(10.0, 10.0),
    sharex=True,
    constrained_layout=True,
)

ax.plot(matrix_dict['synnetqatv2 Pretraining']['thresholds'], matrix_dict['synnetqatv2 Pretraining']['test_metrics'].item()['balanced_accuracy'], marker='o', color=birdcalledge.config.colors[0], label='Pretrained')
ax.plot(matrix_dict['synnetqatv2 QAT']['thresholds'], matrix_dict['synnetqatv2 QAT']['test_metrics'].item()['balanced_accuracy'], marker='o', color=birdcalledge.config.colors[1], label='QAT trained')
ax.set_ylabel("Test Balanced Accuracy")

ax.set_ylim((0.5, 1))
ax.set_xlabel("Threshold")
