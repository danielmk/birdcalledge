# -*- coding: utf-8 -*-
"""
Single entry point for all training runs.
Set CONFIG to the experiment you want to run, then execute this script.

Past experiments (checkpoints in data/checkpoints/):
  synnetqatv2_pretraining          -- synnetqatv1, AdamW lr=1e-4 wd=1e-5, 2001 epochs, no QAT
                                       note: first linear weight halved before training
  synnetqatv2_from_checkpoint_2000 -- synnetqatv1, Adam lr=1e-4, QAT warmup=50, grad_clip=1.0,
                                       loaded from synnetqatv2_pretraining_checkpoint_epoch_2000.pt
  synnetqatv3_pretraining          -- synnetqatv3, AdamW lr=1e-4 wd=1e-6, 2001 epochs, t_stop=2.505, no QAT
                                       note: first linear weight halved before training
  synnetqatv3_from_checkpoint_2000 -- synnetqatv3, Adam lr=1e-4, QAT warmup=50, grad_clip=1.0,
                                       loaded from synnetqatv3_pretraining_checkpoint_epoch_2000.pt
"""

import birdcalledge
from birdcalledge.training import TrainingConfig, train

CONFIG = TrainingConfig(
    run_name        = 'synnetqatv2_replicate_from_checkpoint_2000',
    net_fn          = birdcalledge.nets.synnetqatv1,
    checkpoint_path = 'synnetqatv2_pretraining_replicate_checkpoint_epoch_2000.pt',
    optimizer       = 'Adam',
    lr              = 1e-4,
    qat_enabled     = True,
    qat_warmup      = 50,
    grad_clip       = 1.0,
    checkpoint_every= 250,
)

train(CONFIG)

CONFIG = TrainingConfig(
    run_name        = 'synnetqatv4_pretraining',
    net_fn          = birdcalledge.nets.synnetqatv4,
    checkpoint_path = None,
    optimizer       = 'AdamW',
    weight_decay    = 1e-6,
    lr              = 1e-4,
    qat_enabled     = False,
    grad_clip       = None,
    checkpoint_every= 250,
)

train(CONFIG)

CONFIG = TrainingConfig(
    run_name        = 'synnetqatv5_pretraining',
    net_fn          = birdcalledge.nets.synnetqatv5,
    checkpoint_path = None,
    optimizer       = 'AdamW',
    weight_decay    = 1e-6,
    lr              = 1e-4,
    qat_enabled     = False,
    grad_clip       = None,
    checkpoint_every= 250,
)

train(CONFIG)

CONFIG = TrainingConfig(
    run_name        = 'synnetqatv5_from_checkpoint_2000',
    net_fn          = birdcalledge.nets.synnetqatv5,
    checkpoint_path = 'synnetqatv5_pretraining_epoch_2000.pt',
    optimizer       = 'Adam',
    lr              = 1e-4,
    qat_enabled     = True,
    qat_warmup      = 50,
    grad_clip       = 1.0,
    checkpoint_every= 250,
)

train(CONFIG)