# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:34:58 2026

@author: Daniel
"""

import tables
import numpy as np
from rockpool.nn.networks import SynNetQAT
from torch.optim import AdamW, SGD, Adam
from torch.nn import MSELoss
from rockpool.timeseries import TSEvent
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
import torch
import sys
import pdb
import xylo
from pathlib import Path

"""HYPERPARAMETERS"""
t_stop = 2.505
batch_size = 512

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

dataset_path = r'Y:\danielmk\okeon\dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

train = dst.root.train

q = train.quality_rating[:]
species = train.samples.col("species")
species = np.array([s.decode() if isinstance(s, bytes) else s for s in species])

signal_idx = np.where(
    (q > 1) & (species != "None")
)[0]

noise_idx = np.where(
    species == "None"
)[0]


net = xylo.nets.synnetqatv3(output='vmem')

net.qat_alpha = 0.0

net.qat_enabled = False

net.to(device)

with torch.no_grad():
    net.seq['0_linear'].weight.data = net.seq['0_linear'].weight.data / 2

net.train()

for m in net.seq:
    if 'linear' in m.name:
        print(f'Min: {m.weight.min()}; Max: {m.weight.max()}')

print("Building rasters...")
all_rasters = xylo.training.build_all_rasters_new(train, t_stop, net.dt, net.size_in).to(device)
n_steps = all_rasters.shape[1]

print("Building labels...")
all_labels = xylo.training.build_all_labels_new(train, species, n_steps, net.dt, net.size_out, label_amplitude=1.0).to(device)

print("Labels Done.")

optimizer = AdamW(net.parameters().astorch(), lr=1e-4, weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

loss_fun = MSELoss().to(device=device)

loss_t = []
for epoch in range(2001):
    
    # net.qat_alpha = min(1.0, epoch / qat_warmup)

    batch_idc = xylo.training.sample_batch(batch_size, signal_idx, noise_idx)

    rasters, labels = all_rasters[batch_idc], all_labels[batch_idc]

    optimizer.zero_grad()
    
    output, _, rec = net(rasters, record=False)

    output = output.to(device)
    
    loss = loss_fun(output, labels)

    this_loss = loss.item()
    
    if epoch % 50 == 0:
        xylo.training.save_checkpoint(
            rf"C:\Users\Daniel\repos\xylo\scripts\checkpoints\synnetqatv3_pretraining_checkpoint_epoch_{epoch:04d}.pt",
            net,
            optimizer,
            epoch,
            this_loss,
        )

    loss.backward()

    optimizer.step()

    loss_t.append(this_loss)

    print(f'Epoch: {epoch} | Loss: {this_loss} | Global scale: {net.global_scale} | Output scale: {net.output_scale}')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 18
plt.plot(loss_t)
plt.xlabel("Epoch")
plt.ylabel("Training MSE Loss")
plt.title("Synnetqatv3_pretraining")