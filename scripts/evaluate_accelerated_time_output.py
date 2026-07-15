# -*- coding: utf-8 -*-
"""
Evaluate confusion metrics at each threshold from an accelerated-time deployment
output file (e.g. accelerate_time_deployment_synnetqatv2_pretraining_checkpoint_epoch_2000.npz).

The NPZ file is expected to contain:
  - output    : array of shape (n_thresholds, n_samples, n_timesteps, n_channels)
  - thresholds: array of threshold values, length n_thresholds
  - recs      : recording metadata (unused here)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import birdcalledge

results_dir = Path(__file__).parent.parent / 'data'
dataset_path = results_dir / 'dataset_split.h5'

input_file = results_dir / 'accelerate_time_deployment_synnetqatv2_QAT_from_2000_epoch_2000.npz'
dt = 0.001
t_start_eval = 1.0  # ignore first second (pre-stimulus)

y_true_test, species_test = birdcalledge.datastructure.load_test_labels(dataset_path)

data = np.load(input_file, allow_pickle=True)
output = data['output']        # (n_thresholds, n_samples, n_timesteps, n_channels)
thresholds = data['thresholds']

eval_start_idx = int(t_start_eval / dt)

test_metrics = birdcalledge.evaluation.confusion_metrics_over_thresholds(
    output, y_true_test, eval_start_idx=eval_start_idx
)

for thr, tpr, fpr, acc in zip(thresholds, test_metrics['tpr'], test_metrics['fpr'], test_metrics['accuracy']):
    print(f"threshold={thr:.2f}  TPR={tpr:.3f}  FPR={fpr:.3f}  acc={acc:.3f}")

stem = input_file.stem.replace('accelerate_time_deployment_', '')
out_path = results_dir / f'confusion_metric_accelerated_time_{stem}.npz'

np.savez(
    out_path,
    thresholds=thresholds,
    test_metrics=test_metrics,
    recs=data['recs'],
    allow_pickle=True,
)

print(f"\nSaved to {out_path}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(thresholds, test_metrics['tpr'], label='TPR')
axes[0].plot(thresholds, test_metrics['fpr'], label='FPR')
axes[0].plot(thresholds, test_metrics['accuracy'], label='Accuracy')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Rate')
axes[0].set_title('Metrics vs Threshold')
axes[0].legend()

axes[1].plot(test_metrics['fpr'], test_metrics['tpr'])
axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve')

plt.tight_layout()
# plt.savefig(results_dir / f'confusion_metric_accelerated_time_{stem}.png', dpi=150)
plt.show()
