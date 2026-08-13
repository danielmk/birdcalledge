# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:05:40 2026

@author: Daniel
"""

from rockpool.nn.networks import SynNet, SynNetQAT

def synnetqatv2(output, threshold_out=1.0):
    """This is the basic net with QAT"""
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-3,
    tau_mem=2e-2,
    )

def synnetqatv3(output, threshold_out=1.0):
    """This is the basic net with QAT"""
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=0.7,
    tau_syn_base=1e-2,
    tau_syn_out=1e-2,
    tau_mem=1e-2,
    dt=5e-3
    )

def synnetqatv4(output, threshold_out=1.0):
    """Same as v3 but with the same dt as v1"""
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=0.7,
    tau_syn_base=1e-2,
    tau_syn_out=1e-2,
    tau_mem=1e-2,
    dt=1e-3
    )

def synnetqatv5(output, threshold_out=1.0):
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [64, 48, 32, 32, 32, 32],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-3,
    tau_mem=2e-2,
    dt=1e-3
    )

def synnetqatv6(output, threshold_out=1.0):
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64],
    time_constants_per_layer = [2, 4, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-3,
    tau_mem=2e-2,
    dt=1e-3
    )

def synnetqatv7(output, threshold_out=1.0):
    if output == 'vmem': threshold_out = None
    return SynNetQAT(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [96, 64, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-3,
    tau_mem=2e-2,
    dt=1e-3
    )