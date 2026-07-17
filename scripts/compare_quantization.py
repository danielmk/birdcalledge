# -*- coding: utf-8 -*-
"""
Diagnostic: compare mapper()/global_quantize() output for the old
(known-good, 93% accuracy) checkpoint vs the new replicate checkpoint,
to find where their quantized representations diverge.
"""

import numpy as np
import torch
from pathlib import Path

import birdcalledge
from rockpool.devices.xylo.syns65302 import mapper
import rockpool.transform.quantize_methods as q

ckpt_dir = Path(__file__).parent.parent / 'data' / 'checkpoints'

cases = [
    ("OLD_GOOD", "synnetqatv2_from_checkpoint_2000_epoch_2000.pt"),
    ("NEW_REPLICATE", "synnetqatv2_replicate_from_checkpoint_2000_epoch_2000.pt"),
]

for label, fname in cases:
    print(f"\n=== {label}: {fname} ===")
    ckpt = torch.load(ckpt_dir / fname, map_location="cpu")
    net = birdcalledge.nets.synnetqatv2(output='spikes', threshold_out=1.0)
    net.load_state_dict(ckpt["model_state"])

    spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float', dash_dtype='float')

    w_in = spec['weights_in']
    w_rec = spec['weights_rec']
    w_out = spec['weights_out']
    thr = spec['threshold']
    thr_out = spec['threshold_out']

    print(f"max|w_in|={np.max(np.abs(w_in)):.4f}  max|w_rec|={np.max(np.abs(w_rec)):.4f}  max|w_out|={np.max(np.abs(w_out)):.4f}")
    print(f"threshold: min={np.min(thr):.4f} max={np.max(thr):.4f}  threshold_out: {thr_out}")
    print(f"dash_mem: {spec['dash_mem']}")
    print(f"dash_mem_out: {spec['dash_mem_out']}")
    print(f"dash_syn: {spec['dash_syn']}")
    print(f"dash_syn_2: {spec['dash_syn_2']}")
    print(f"dash_syn_out: {spec['dash_syn_out']}")

    quan = q.global_quantize(**spec)
    print(f"quantized max|w_in|={np.max(np.abs(quan['weights_in']))}  max|w_rec|={np.max(np.abs(quan['weights_rec']))}  max|w_out|={np.max(np.abs(quan['weights_out']))}")
    print(f"quantized threshold min/max: {quan['threshold'].min()}/{quan['threshold'].max()}  threshold_out: {quan['threshold_out']}")
    print(f"quantized dash_mem: {quan['dash_mem']}")
    print(f"quantized dash_syn: {quan['dash_syn']}")
