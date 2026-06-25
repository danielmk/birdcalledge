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
from pathlib import Path
from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
import rockpool.transform.quantize_methods as q
from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as hdu
from rockpool.devices.xylo.syns65302 import XyloSamna
import samna
import pickle
import sys
import time
import pdb
import birdcalledge

results_dir = Path(__file__).parent.parent / 'data'
ckpt_dir = results_dir / "checkpoints"
checkpoint = Path('synnetqatv2_pretraining_checkpoint_epoch_2000.pt')
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

"""HYPERPARAMETERS"""
t_stop=2.504
batch_size=64

# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------

net = test_net(output='spikes')
curr_ckpt = torch.load(ckpt_dir / checkpoint, map_location=device)

"""SETUP"""
print("Building rasters...")
all_rasters = birdcalledge.training.build_all_rasters(test, t_stop, net.dt, net.size_in)

# Move **once**
all_rasters = all_rasters

thresholds = np.arange(0.5, 1.55, 0.1)

all_outputs = []
all_recs = []

for th in thresholds:
    print(f"Threshold: {th}")
    net = test_net(output='spikes', threshold_out=th)
    
    net.load_state_dict(curr_ckpt["model_state"])
    
    """QUANTIZE AND BULID XYLO 3 CONFIGURATION"""
    # getting the model specifications using the mapper function
    spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

    # quantizing the model
    # spec.update(q.channel_quantize(**spec))
    spec.update(q.global_quantize(**spec))
    
    xylo_conf, is_valid, msg = config_from_specification(**spec)
    
    # Getting the connected devices and choosing XyloAudio 3 board
    xylo_nodes = hdu.find_xylo_a3_boards()
    
    if len(xylo_nodes) == 0:
        raise ValueError('A connected XyloAudio 3 development board is required for this tutorial.')
    
    xa3 = xylo_nodes[0]
    
    # Instantiating XyloSamna and deploying to the dev kit; make sure your dt corresponds to the dt of your input data
    Xmod = XyloSamna(device=xa3, config=xylo_conf, dt=net.dt)
    
    time.sleep(10)
    
    try:
        Xmod(all_rasters[0], record=False, record_power=True)
    except:
        raise Warning()

    out_list = []
    state_list = []
    rec_list = []
    
    for idx, raster in enumerate(all_rasters):
        print(f"Curr idx: {idx}")
        out, state, rec = Xmod(raster, record=False, record_power=True)
        out_list.append(out)
        state_list.append(state)
        rec_list.append(rec)
    
    all_outputs.append(out_list)

    all_recs.append(rec_list)
    

np.savez(f'{checkpoint.stem}_accelerate_time_deployment.npz',
         output=all_outputs,
         thresholds=thresholds,
         recs=all_recs)
    
    # output, out2, out3 = net(all_rasters, record=True)




