# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:05:40 2026

@author: Daniel
"""

from rockpool.nn.networks import SynNet, SynNetQAT

def synnet(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    )

def sntc(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    train_time_constants=True,
    )

def sntcth(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    train_time_constants=True,
    train_threshold=True,
    )

def sntcth(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [140, 40, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    train_time_constants=True,
    train_threshold=True,
    )

def synnetv2(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 64, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    )

def snthv2(output):
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 64, 40, 40, 40, 40],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold=0.5,
    train_threshold=True,
    )

def synnetv3(output, threshold_out=1.0):
    if output == 'vmem': threshold_out = None
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-2,
    tau_mem=2e-2,
    )

def synnetv4(output, threshold_out=1.0):
    if output == 'vmem': threshold_out = None
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 2, 4, 4, 4],
    output=output,
    threshold=1.0,
    threshold_out=threshold_out,
    tau_syn_base=5e-3,
    tau_syn_out=2e-2,
    tau_mem=2e-2,
    )

def synnetv5(output, threshold_out=1.0):
    """This is identical to synnetv3 but used in a different script."""
    if output == 'vmem': threshold_out = None
    return SynNet(
    n_channels = 16,
    n_classes = 1,
    size_hidden_layers = [128, 96, 64, 64, 64, 64],
    time_constants_per_layer = [2, 2, 4, 4, 8, 8],
    output=output,
    threshold_out=threshold_out,
    threshold=1.0,
    tau_syn_out=2e-2,
    tau_mem=2e-2,
    )

def synnetqatv1(output, threshold_out=1.0):
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
