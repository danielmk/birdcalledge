# -*- coding: utf-8 -*-
"""
Two-panel power + accuracy figure.

Panel 1: Power and balanced accuracy vs output threshold for SynNetQATv2
         (pretraining vs QAT), from threshold-sweep deployment files.

Panel 2: Cross-network power comparison at a fixed threshold.
         SynNetQATv3 QAT data already available. SynNetQATv2 pretraining
         and QAT still need to be deployed at a fixed threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import birdcalledge

results_dir = Path(__file__).parent.parent / 'data'

# Files with threshold sweeps — used for Panel 1
THRESHOLD_FILES = {
    'SynNetQATv2 Pretraining': results_dir / 'confusion_metric_accelerated_time_synnetqatv2_pretraining_epoch_2000_thresholds.npz',
    'SynNetQATv2 QAT':         results_dir / 'confusion_metric_accelerated_time_synnetqatv2_QAT_from_2000_epoch_2000_thresholds.npz',
}

# Files with multiple samples at a fixed threshold — used for Panel 2
# TODO: add synnetqatv2_pretraining and synnetqatv2_QAT once deployed at fixed threshold
CROSSNET_FILES = {
    'SynNetQATv3 QAT': results_dir / 'confusion_metric_accelerated_time_synnetqatv3_QAT_from_2000_epoch_2000.npz',
}

POWER_KEYS = ['io_power', 'analog_power', 'digital_power']


def total_power_per_threshold(recs):
    """Return array of mean total power (mW) for each threshold slice."""
    totals = []
    for thr_recs in recs:
        sample_totals = []
        for rec in thr_recs:
            sample_totals.append(sum(np.mean(rec[k]) * 1e3 for k in POWER_KEYS))
        totals.append(np.mean(sample_totals))
    return np.array(totals)



# --- Load threshold-sweep data ---
thr_results = {}
for name, path in THRESHOLD_FILES.items():
    data = np.load(path, allow_pickle=True)
    thr_results[name] = {
        'thresholds':        data['thresholds'],
        'thr_totals':        total_power_per_threshold(data['recs']),
        'balanced_accuracy': np.array(data['test_metrics'].item()['balanced_accuracy']),
    }

# --- Load cross-network data ---
crossnet_results = {}
for name, path in CROSSNET_FILES.items():
    data = np.load(path, allow_pickle=True)
    all_vals = []
    for thr_recs in data['recs']:
        for rec in thr_recs:
            all_vals.append(sum(np.mean(rec[k]) * 1e3 for k in POWER_KEYS))
    crossnet_results[name] = {'power_samples': np.array(all_vals)}

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Panel 1: threshold sweep for v2
for i, (name, res) in enumerate(thr_results.items()):
    color = birdcalledge.config.colors_qual[i]
    axes[0].plot(res['thresholds'], res['thr_totals'], marker='o', color=color, label=name)

axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Average power (mW)')
axes[0].set_ylim(bottom=0)
axes[0].legend()

# Panel 2: cross-network comparison at fixed threshold
names = list(crossnet_results.keys())
data_list = [crossnet_results[n]['power_samples'] for n in names]

bp = axes[1].boxplot(data_list, patch_artist=True)
for patch, color in zip(bp['boxes'], birdcalledge.config.colors_qual):
    patch.set_facecolor(color)

axes[1].set_xticks(range(1, len(names) + 1))
axes[1].set_xticklabels(names, rotation=15, ha='right')
axes[1].set_ylabel('Average power (mW)')
axes[1].set_ylim(bottom=0)

plt.show()
