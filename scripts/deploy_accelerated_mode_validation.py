# -*- coding: utf-8 -*-
"""
Run the deployed SynNet on the multi-species validation set in accelerated mode.

Same hardware path as deploy_accelerated_mode_synnet.py, pointed at the
/validation group of dataset_validation.h5 instead of the /test split, with
three changes that the size of the set makes necessary:

  * 8172 samples instead of 443, so the run is checkpointed every
    CHECKPOINT_EVERY samples and resumes from the partial file. A USB hiccup
    costs minutes rather than the whole run.
  * record_power is off by default. Power traces for 8172 samples dominate the
    output file, and the power figure already comes from the test-set run.
  * is_valid from config_from_specification is checked rather than ignored.

The output file has the same layout as the test-set one, so
evaluate_accelerated_time_output.py style analysis works on it unchanged.

Run on the machine with the XyloAudio 3 dev kit:

    python scripts/deploy_accelerated_mode_validation.py --limit 20   # timing check
    python scripts/deploy_accelerated_mode_validation.py

@author: Daniel
"""

import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tables
import torch

import birdcalledge

from rockpool.devices.xylo.syns65302 import config_from_specification, mapper
from rockpool.devices.xylo.syns65302 import xa3_devkit_utils as hdu
from rockpool.devices.xylo.syns65302 import XyloSamna
import rockpool.transform.quantize_methods as q

results_dir = Path(__file__).parent.parent / 'data'
ckpt_dir = results_dir / "checkpoints"
checkpoint = Path('synnetqatv2_replicate_from_checkpoint_2000_epoch_2000.pt')
device = torch.device("cpu")
test_net = birdcalledge.nets.synnetqatv2

if platform.system() == "Windows":
    DATASET = Path(r"\\bucket.oist.jp\bucket\FukaiU\danielmk\okeon\dataset_validation.h5")
else:
    DATASET = Path("/bucket/FukaiU/danielmk/okeon/dataset_validation.h5")

GROUP_NAME = "validation"

"""HYPERPARAMETERS"""
t_stop = 2.504

# one operating point by default, must match the threshold the test-set
# numbers are reported at
THRESHOLDS = np.array([1.0])

RECORD_POWER = False
CHECKPOINT_EVERY = 500
WARMUP_SLEEP = 10


def build_configuration(th, curr_ckpt):
    """Quantized Xylo configuration for one output threshold."""
    net = test_net(output='spikes', threshold_out=th)
    net.load_state_dict(curr_ckpt["model_state"])
    net.eval()

    spec = mapper(net.as_graph(), weight_dtype='float', threshold_dtype='float',
                  dash_dtype='float')
    spec.update(q.global_quantize(**spec))

    xylo_conf, is_valid, msg = config_from_specification(**spec)
    if not is_valid:
        raise ValueError(f"invalid Xylo configuration: {msg}")

    return net, xylo_conf


def connect(xylo_conf, dt, raster):
    """Open the dev kit and push one raster through to wake it up."""
    xylo_nodes = hdu.find_xylo_a3_boards()
    if len(xylo_nodes) == 0:
        raise ValueError('A connected XyloAudio 3 development board is required.')

    Xmod = XyloSamna(device=xylo_nodes[0], config=xylo_conf, dt=dt)
    time.sleep(WARMUP_SLEEP)

    try:
        Xmod(raster, record=False, record_power=RECORD_POWER)
    except Exception:
        warnings.warn("Xmod was not ready yet.")

    return Xmod


def save_partial(path, outputs, n_done, thresholds):
    """Checkpoint atomically, so an interrupted run loses nothing."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, output=outputs, n_done=n_done, thresholds=thresholds)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--group-name", default=GROUP_NAME)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N samples, for a timing check")
    parser.add_argument("--record-power", action="store_true", default=RECORD_POWER)
    parser.add_argument("--out-tag", default="validation")
    args = parser.parse_args()

    dst = tables.open_file(args.dataset, mode="r")
    group = dst.get_node(f"/{args.group_name}")

    curr_ckpt = torch.load(ckpt_dir / checkpoint, map_location=device)
    net = test_net(output='spikes')

    print("Building rasters...", flush=True)
    all_rasters = birdcalledge.training.build_all_rasters_new(group, t_stop, net.dt,
                                                              net.size_in)
    if args.limit is not None:
        all_rasters = all_rasters[:args.limit]
    n_samples = len(all_rasters)
    print(f"{n_samples} samples, {len(THRESHOLDS)} threshold(s)", flush=True)

    out_path = results_dir / f'accelerate_time_deployment_{checkpoint.stem}_{args.out_tag}.npz'
    partial_path = out_path.with_suffix('.partial.npz')

    outputs = None
    n_done = 0
    if partial_path.exists():
        partial = np.load(partial_path, allow_pickle=True)
        outputs = partial['output']
        n_done = int(partial['n_done'])
        print(f"resuming, {n_done}/{len(THRESHOLDS) * n_samples} already deployed", flush=True)

    t0 = time.time()
    for thr_idx, th in enumerate(THRESHOLDS):
        start = thr_idx * n_samples
        if n_done >= start + n_samples:
            print(f"threshold {th} already done", flush=True)
            continue

        print(f"Threshold: {th}", flush=True)
        net, xylo_conf = build_configuration(th, curr_ckpt)
        Xmod = connect(xylo_conf, net.dt, all_rasters[0])

        for idx in range(max(0, n_done - start), n_samples):
            out, state, rec = Xmod(all_rasters[idx], record=False,
                                   record_power=args.record_power)

            out = np.asarray(out)
            if outputs is None:
                outputs = np.zeros((len(THRESHOLDS), n_samples) + out.shape,
                                   dtype=np.int16)
            outputs[thr_idx, idx] = out

            n_done = start + idx + 1
            if n_done % CHECKPOINT_EVERY == 0:
                save_partial(partial_path, outputs, n_done, THRESHOLDS)
                elapsed = time.time() - t0
                total = len(THRESHOLDS) * n_samples
                print(f"[{n_done}/{total}] {elapsed / 60:.1f} min elapsed, "
                      f"{(total - n_done) * elapsed / max(n_done, 1) / 60:.1f} min left",
                      flush=True)

    np.savez(out_path, output=outputs, thresholds=THRESHOLDS,
             dataset=str(args.dataset), group_name=args.group_name)
    partial_path.unlink(missing_ok=True)
    dst.close()

    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
