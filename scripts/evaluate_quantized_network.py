# -*- coding: utf-8 -*-
"""
Calculate the test accuracy of a quantized network run with XyloSim.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tables
from matplotlib.animation import FuncAnimation, FFMpegWriter
import soundfile as sf
from rockpool.nn.networks import SynNet
import torch
import xylo
from pathlib import Path
from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
import rockpool.transform.quantize_methods as q
from rockpool.devices.xylo.syns65302 import XyloSim
import samna
import pickle
import sys
import time
from rockpool.timeseries import TSEvent

from rockpool.nn.modules import to_nir, LinearTorch, LIFTorch
from rockpool.nn.combinators import Sequential
import nir
import copy
import birdcalledge

import pdb

"""HYPERPARAMETERS"""
results_dir = Path(__file__).parent.parent / 'data'
ckpt_dir = results_dir / "checkpoints"
checkpoint_prefix = 'synnetqatv2_pretraining_checkpoint_epoch_2000.pt'
device = torch.device("cpu")
test_net = birdcalledge.nets.synnetqatv1
threshold_grid = np.arange(0.5, 1.55, 0.1)

dataset_path = Path(__file__).parent.parent / 'data' / 'dataset_split.h5'

dst = tables.open_file(dataset_path, mode="r")

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

# example_idx = high_quality[2]
example_idx = 1

test = dst.root.test

q_test = test.quality_rating[:]
species_test = test.samples.col("species")
species_test = np.array([s.decode() if isinstance(s, bytes) else s for s in species_test])

y_true_test = np.zeros((species_test.shape[0]))
y_true_test[species_test=='Ruddy Kingfisher'] = 1

rng = np.random.default_rng()

"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64
"""
# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------
synnet_ckpts = sorted(
    p for p in ckpt_dir.iterdir()
    if p.is_file() and checkpoint_prefix in p.name
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
"""

curr_ckpt = torch.load(ckpt_dir / checkpoint_prefix, map_location=device)

net = test_net().to(device)

all_rasters_test = birdcalledge.training.build_all_rasters_new(test, t_stop, net.dt, size_in=net.size_in)

net.load_state_dict(curr_ckpt["model_state"])

spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

spec_pre = copy.copy(spec)

spec.update(q.global_quantize(**spec))

"""
The recurrent weight matrix is nonzero in the top-right quarter, reflecting
that the first 63 neurons connect to the next 63 neurons (filled with the 
weights on the second linear layer of the model). Note that there are no 
recurrent connections as the diagonal blocks are all zero.
"""

quantized_net = XyloSim.from_specification(**spec)

output_test = []
for idx, curr_raster in enumerate(all_rasters_test):
    print(f"Current raster: {idx}")
    curr_output, _, _ = net(curr_raster, record=False)
    output_test.append(curr_output)

output_test = np.array(output_test)



# nir_graph = to_nir(net)

# nir.write("sntc_epoch_5500.nir", nir_graph)


