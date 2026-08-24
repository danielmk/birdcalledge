# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:00:25 2026

@author: Daniel
"""

import numpy as np
import torch
from pathlib import Path
from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
import rockpool.transform.quantize_methods as q
from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as hdu
from rockpool.devices.xylo.syns65302 import XyloMonitor
import samna
import time
import birdcalledge

results_dir = Path(__file__).parent.parent / 'data'
ckpt_dir = results_dir / "checkpoints"
checkpoint = Path('synnetqatv2_replicate_from_checkpoint_2000_epoch_2000.pt')
device = torch.device("cpu")
test_net = birdcalledge.nets.synnetqatv2

"""HYPERPARAMETERS"""
# ---------------------------------------------------------------------
# MODEL + OPTIMIZER (must match original training!)
# ---------------------------------------------------------------------
threshold_out = 1.0
recording_duration = 120  # seconds

net = test_net(output='spikes', threshold_out=threshold_out)
curr_ckpt = torch.load(ckpt_dir / checkpoint, map_location=device)
net.load_state_dict(curr_ckpt["model_state"])
net.eval()

"""SETUP"""
torch.manual_seed(65)
np.random.seed(68)

"""QUANTIZE AND BULID XYLO 3 CONFIGURATION"""
# getting the model specifications using the mapper function
spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')
# quantizing the model
spec.update(q.global_quantize(**spec))

xylo_conf, is_valid, msg = config_from_specification(**spec)

# Getting the connected devices and choosing XyloAudio 3 board
xylo_nodes = hdu.find_xylo_a3_boards()

if len(xylo_nodes) == 0:
    raise ValueError('A connected XyloAudio 3 development board is required for this tutorial.')

xa3 = xylo_nodes[0]

# Instantiating XyloMonitor and deploying to the dev kit; make sure your dt corresponds to the dt of your input data
xylo_monitor = XyloMonitor(device=xa3, config=xylo_conf, dt=net.dt, output_mode='Spike', dn_active=True, main_clk_rate=120, power_frequency=10)

time.sleep(5)

T = int(recording_duration / net.dt)

xylo_input = np.zeros((T, 1))

print("Start Recording")

out, state, rec = xylo_monitor.evolve(xylo_input, record_power=True)

np.savez(results_dir / f'realtime_deployment_{checkpoint.stem}.npz',
         out=out,
         state=state,
         rec=rec)
