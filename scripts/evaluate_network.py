# -*- coding: utf-8 -*-
"""
Calculate the test accuracy of a network at different output thresholds.
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
import birdcalledge

"""HYPERPARAMETERS"""
t_stop=2.504
n_train=500
results_dir = Path(__file__).parent.parent / 'data'
ckpt_dir = results_dir / "checkpoints"
checkpoint_prefix = 'synnetqatv2_pretraining_'
threshold_grid = np.arange(0.5, 1.55, 0.1)
test_net = birdcalledge.nets.synnetqatv1
QAT_ENABLED = False

"""VERIFY HYPERPARAMETERS AND NETWORK"""
net = test_net(output='spikes')

if not hasattr(net, 'qat_enabled'):
    if QAT_ENABLED:
        raise Warning("The script tries to enable QAT but the network does not support it. QAT setting will be ignored.")

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

device = torch.device("cpu")

dataset_path = Path(__file__).parent.parent / 'data' / 'dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

train = dst.root.train

q_train = train.quality_rating[:]
species_train = train.samples.col("species")
species_train = np.array([s.decode() if isinstance(s, bytes) else s for s in species_train])

y_true_train = np.zeros((species_train.shape[0]))
y_true_train[species_train=='Ruddy Kingfisher'] = 1

print("Building rasters...")
all_rasters_test = birdcalledge.training.build_all_rasters_new(test, t_stop, net.dt, size_in=net.size_in)

all_rasters_train = birdcalledge.training.build_all_rasters_new(train, t_stop, net.dt, size_in=net.size_in)

all_rasters_train = all_rasters_train[:n_train, :, :]

y_true_train = y_true_train[:n_train]

# Move **once**
all_rasters_test = all_rasters_test.to(device)

all_rasters_train = all_rasters_train.to(device)

synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and checkpoint_prefix in p.name
)

def predict_events_from_net(net, rasters):
    output, _, _ = net(rasters, record=False)

    return torch.any(
        output[:, int(1.0 / net.dt):, 0] == 1,
        axis=1
    ).cpu().numpy()

synnet_ckpts = sorted(
    synnet_ckpts,
    key=lambda p: torch.load(p, map_location="cpu").get("epoch", 0)
)

checkpoints = [
    torch.load(path, map_location="cpu")
    for path in synnet_ckpts
]

checkpoints = [x for x in checkpoints if x['epoch'] % 250 == 0 and x['epoch'] <= 5000]

training_metrics = []
test_metrics = []

epochs = []

loss= []

for ckpt in checkpoints:

    train_ckpt_metrics = {k: [] for k in birdcalledge.evaluation.CONFUSION_KEYS}
    test_ckpt_metrics  = {k: [] for k in birdcalledge.evaluation.CONFUSION_KEYS}
    
    epochs.append(ckpt['epoch'])
    loss.append(ckpt['loss'])

    for thr in threshold_grid:

        net = test_net(output='spikes', threshold_out=thr).to(device)
                
        net.load_state_dict(ckpt["model_state"])

        net.qat_enabled = QAT_ENABLED
        if QAT_ENABLED:
            net.qat_alpha = 1.0
        
        net.eval()                # important!

        y_pred_train = predict_events_from_net(net, all_rasters_train)
        y_pred_test  = predict_events_from_net(net, all_rasters_test)

        train_rates = birdcalledge.evaluation.confusion_rates(y_true_train, y_pred_train)
        test_rates = birdcalledge.evaluation.confusion_rates(y_true_test, y_pred_test)

        # Store everythig
        for k in birdcalledge.evaluation.CONFUSION_KEYS:
            train_ckpt_metrics[k].append(train_rates[k])
            test_ckpt_metrics[k].append(test_rates[k])

        print(
            f"Epoch {ckpt['epoch']:>3} | "
            f"thr={thr:.2f} | "
            f"BA train={train_rates['balanced_accuracy']:.3f}, "
            f"test={test_rates['balanced_accuracy']:.3f} | "
            f"FPR test={test_rates['fpr']:.3f}"
        )

    training_metrics.append(train_ckpt_metrics)
    test_metrics.append(test_ckpt_metrics)


np.savez(
    results_dir / f"{checkpoint_prefix}threshold_checkpoint_confusion_metric_quantized.npz",
    thresholds=threshold_grid,
    loss=loss,
    epochs=epochs,
    training_metrics=training_metrics,
    test_metrics=test_metrics,
    allow_pickle=True
)
