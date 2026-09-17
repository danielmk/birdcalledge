# -*- coding: utf-8 -*-
"""
False positives of the deployed network on the multi-species validation set.

Reads the accelerated-time output written by
deploy_accelerated_mode_validation.py and reports, per species, the fraction of
samples that made the readout fire. For the 69 non-target species that fraction
is a false-positive rate; for the Ruddy Kingfisher it is the true-positive rate
and serves as the reference bar in the figure.

One difference from evaluate_accelerated_time_output.py: that script ignores the
first second of every sample because the training windows start 1 s before the
labelled onset, so the first second is pre-stimulus. The validation windows are
the last 2.5 s of a BirdNET detection interval with no onset alignment, so a call
can fall anywhere in the window and skipping the first second would hide real
firing. The full window is therefore the primary measure here; the 1 s-onward
number is printed as well, since that is the one directly comparable with the
test-set figures.

@author: Daniel
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import birdcalledge

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42  # embed TrueType so text stays editable in Illustrator
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = 'Arial'

results_dir = Path(__file__).parent.parent / 'data'

INPUT_FILE = results_dir / ('accelerate_time_deployment_'
                            'synnetqatv2_replicate_from_checkpoint_2000_epoch_2000_validation.npz')

TARGET_SPECIES = 'Ruddy Kingfisher'
GROUP_NAME = 'validation'
DT = 0.005
READOUT_CHANNEL = 0


def fired(output, eval_start_idx):
    """True where the readout spiked at least once in the evaluation window."""
    return np.any(output[:, eval_start_idx:, READOUT_CHANNEL] > 0, axis=1)


def per_species_rates(species, y_pred):
    """Fraction of samples of each species that made the readout fire."""
    names = sorted(set(species))
    rates, counts = [], []
    for name in names:
        mask = species == name
        rates.append(float(np.mean(y_pred[mask])))
        counts.append(int(mask.sum()))

    return np.array(names), np.array(rates), np.array(counts)


def plot_false_positives(names, rates, counts, overall_fpr, target_rate, out_stem):
    order = np.argsort(rates)[::-1]
    names, rates, counts = names[order], rates[order], counts[order]

    is_target = names == TARGET_SPECIES
    colors = np.where(is_target, birdcalledge.config.colors_oist[0],
                      birdcalledge.config.colors_oist[1])
    labels = [f'{n} (target)' if t else n for n, t in zip(names, is_target)]

    fig, ax = plt.subplots(figsize=(9, 0.21 * len(names) + 2), constrained_layout=True)

    y = np.arange(len(names))
    ax.barh(y, 100 * rates, color=list(colors))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Samples that triggered the readout (%)')
    x_max = max(100 * rates.max(), 100 * target_rate) * 1.15 + 1
    ax.set_xlim(0, x_max)

    # a column at the right edge, so the counts never collide with short bars
    for yi, n in enumerate(counts):
        ax.text(x_max * 0.995, yi, f'n={n}', va='center', ha='right',
                fontsize=7, color='0.35')

    ax.axvline(100 * overall_fpr, ls='--', color='k', lw=1.2,
               label=f'non-target mean {100 * overall_fpr:.1f}%')
    ax.legend(fontsize=11, loc='lower right')

    fig.savefig(results_dir / f'{out_stem}.pdf')
    fig.savefig(results_dir / f'{out_stem}.png', dpi=200)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=INPUT_FILE)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="default: the dataset recorded in the deployment file")
    parser.add_argument("--threshold-index", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.input_file, allow_pickle=True)
    output = np.asarray(data['output'])[args.threshold_index]
    threshold = np.asarray(data['thresholds'])[args.threshold_index]

    dataset = args.dataset or Path(str(data['dataset']))
    group_name = str(data['group_name']) if 'group_name' in data else GROUP_NAME

    y_true, species = birdcalledge.datastructure.load_test_labels(
        dataset, target_species=TARGET_SPECIES, group_name=group_name)

    # a --limit run deploys only the first samples of the set
    n = len(output)
    y_true, species = y_true[:n], species[:n]

    y_pred = fired(output, 0)
    y_pred_late = fired(output, int(1.0 / DT))

    target = species == TARGET_SPECIES
    overall_fpr = float(np.mean(y_pred[~target]))
    target_rate = float(np.mean(y_pred[target]))

    print(f"{n} samples at threshold {threshold}, {group_name} of {dataset.name}")
    print(f"  full window   : FP rate {100 * overall_fpr:5.2f}%   "
          f"target TPR {100 * target_rate:5.2f}%")
    print(f"  from 1 s on   : FP rate {100 * np.mean(y_pred_late[~target]):5.2f}%   "
          f"target TPR {100 * np.mean(y_pred_late[target]):5.2f}%   (test-set convention)")

    rates = birdcalledge.evaluation.confusion_rates(y_true, y_pred)
    print("  balanced accuracy %.3f, precision %.3f, FDR %.3f"
          % (rates['balanced_accuracy'], rates['precision'], rates['fdr']))

    names, species_rates, counts = per_species_rates(species, y_pred)

    print(f"\nworst 15 non-target species ({(~target).sum()} samples total):")
    order = np.argsort(species_rates)[::-1]
    shown = 0
    for i in order:
        if names[i] == TARGET_SPECIES:
            continue
        print("  %-32s %5.1f%%  (n=%d)" % (names[i], 100 * species_rates[i], counts[i]))
        shown += 1
        if shown == 15:
            break

    out_stem = args.input_file.stem.replace('accelerate_time_deployment_', 'false_positives_')
    np.savez(results_dir / f'{out_stem}.npz',
             species=names, fp_rate=species_rates, n_samples=counts,
             y_pred=y_pred, y_pred_late=y_pred_late, y_true=y_true,
             threshold=threshold, overall_fpr=overall_fpr, target_tpr=target_rate)

    plot_false_positives(names, species_rates, counts, overall_fpr, target_rate, out_stem)
    print(f"\nwrote {results_dir / out_stem}.pdf / .png / .npz")

    plt.show()


if __name__ == "__main__":
    main()
