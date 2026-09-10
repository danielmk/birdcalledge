# -*- coding: utf-8 -*-
"""
Build a multi-species validation dataset from the BirdNET survey.

process_birdnet_survey.py labelled the whole OKEON archive with BirdNET, which
gives detections for hundreds of species besides the Ruddy Kingfisher. This
script turns those detections into a dataset with the same structure as
dataset_split.h5, so the specificity of the trained network can be measured
against calls it was never meant to fire on.

The extraction follows process_sample_extraction.py: overlapping detections of
one species in one recording are merged into a single call event, and a window
of TIME_PRE + TIME_POST seconds is stored together with the AFE spikes for that
audio, so the samples have the same length and encoding as the training set.

Two things differ from the Ruddy Kingfisher extraction:

  * No envelope onset alignment. The window is simply the last TIME_PRE +
    TIME_POST seconds of the BirdNET detection interval. The 1200-2500 Hz band
    that alignment relies on is specific to the target species, and a threshold
    on the envelope of a continuous sound such as a cicada never triggers, which
    dropped a third of the events in testing. The f_low, f_high and threshold_k
    columns are therefore NaN and bandpass_order is 0 for every sample in this
    set, and call_duration holds the length of the merged detection interval
    rather than a measured call.
  * Samples are capped per species (MAX_PER_SPECIES) and spread over recordings
    and sites, so one abundant species cannot dominate the set.

Writing is resumable: samples already in the output file are skipped, so an
interrupted run continues where it stopped.

Run with the birdcalledge conda environment, e.g.

    conda run -n birdcalledge python scripts/process_validation_dataset.py --limit 20

@author: Daniel
"""

import argparse
import pathlib
import platform
import time

import numpy as np
import pandas as pd
import librosa
import tables

import birdcalledge

from rockpool.devices.xylo.syns65302 import AFESimExternal
from rockpool.timeseries import TSEvent

# the same bucket, mounted as a share on Windows and under /bucket on Linux
if platform.system() == "Windows":
    BUCKET = pathlib.Path(r"\\bucket.oist.jp\bucket\FukaiU\danielmk")
else:
    BUCKET = pathlib.Path("/bucket/FukaiU/danielmk")

DETECTIONS = BUCKET / "okeon_birdnet" / "all_detections.csv"
RAW_ROOT = BUCKET / "okeon"
OUT_PATH = RAW_ROOT / "dataset_validation.h5"  # beside dataset_split.h5

GROUP_NAME = "validation"

# SPECIES SELECTION
CONFIDENCE_THRESHOLD = 0.8
MIN_DETECTIONS = 50    # species below this are dropped
MAX_PER_SPECIES = 200  # samples kept per species
MAX_PER_RECORDING = 2  # samples one recording may contribute to one species
SEED = 20260910

# EXTRACTION TIMES, the window is centred on the detection interval
TIME_PRE = 1.0
TIME_POST = 1.5

# NOT APPLIED HERE, recorded as such in the sample table
BANDPASS_ORDER = 0

# AFE SIMULATION
DT_S = 0.009994

EXPECTED_SR = 44100.0


def call_events(detections):
    """Merge overlapping detections of one species in one recording."""
    events = []
    for (filename, species), group in detections.groupby(["Filename", "Species_Name"]):
        merged = birdcalledge.features.merge_intervals_pandas(group)
        merged["Filename"] = filename
        merged["Species_Name"] = species
        merged["Site"] = group["Site"].iloc[0]
        merged["Scientific_Name"] = group["Scientific_Name"].iloc[0]
        events.append(merged)

    return pd.concat(events, ignore_index=True)


def select_events(events, args, rng):
    """Cap events per species, spread over sites and recordings."""
    keep = []
    for species, group in events.groupby("Species_Name"):
        # round-robin over sites so no single site fills the quota
        by_site = []
        for _, site_group in group.groupby("Site"):
            shuffled = site_group.sample(frac=1.0, random_state=rng.integers(2**32))
            by_site.append(shuffled)

        taken = 0
        per_recording = {}
        for row in _interleave(by_site):
            if taken >= args.max_per_species:
                break
            seen = per_recording.get(row["Filename"], 0)
            if seen >= args.max_per_recording:
                continue
            per_recording[row["Filename"]] = seen + 1
            keep.append(row)
            taken += 1

    selected = pd.DataFrame(keep).reset_index(drop=True)

    return selected.sort_values(["Filename", "Time_start"]).reset_index(drop=True)


def _interleave(frames):
    """Yield rows from several frames in turn, longest tail last."""
    per_site = [[row for _, row in frame.iterrows()] for frame in frames]
    for i in range(max(len(rows) for rows in per_site)):
        for rows in per_site:
            if i < len(rows):
                yield rows[i]


def recording_path(raw_root, filename):
    """SITE/YEAR/SITE_YYYYMMDD_HHMMSS.flac, the layout of the raw archive."""
    site, date, _ = pathlib.Path(filename).stem.split("_")

    return raw_root / site / date[:4] / filename


def window_start(event, file_duration, window_seconds):
    """Start of the last window_seconds of the detection, kept inside the file."""
    if file_duration < window_seconds:
        return None

    start = event["Time_End"] - window_seconds

    return float(np.clip(start, 0.0, file_duration - window_seconds))


def encode_spikes(audio, sr):
    """Xylo audio front end spikes for one sample, as in the training set."""
    afesim = AFESimExternal.from_specification(
        spike_gen_mode="divisive_norm",
        fixed_threshold_vec=None,
        dt=DT_S,
    )

    out_external, _, _ = afesim((audio, sr))

    return TSEvent.from_raster(out_external, dt=DT_S)


def existing_samples(out_path, group_name):
    """(filename, t_start) of samples already written, for resuming."""
    if not out_path.exists():
        return set()

    with tables.open_file(out_path, mode="r") as h5:
        table = h5.get_node(f"/{group_name}").samples
        if table.nrows == 0:
            return set()

        names = np.char.decode(table.col("filename"), encoding="utf-8")

        return set(zip(names.tolist(), np.round(table.col("t_start"), 6).tolist()))


def extract(selected, args):
    """Write one sample per selected event into the dataset."""
    target_len = int(round((args.time_pre + args.time_post) * EXPECTED_SR))

    if not args.out_path.exists():
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        birdcalledge.datastructure.create_empty_dataset(
            str(args.out_path), audio_length=target_len, group_name=args.group_name)
        print(f"created {args.out_path} for {target_len} sample windows", flush=True)

    done = existing_samples(args.out_path, args.group_name)
    print(f"{len(done)} samples already in the dataset", flush=True)

    window_seconds = args.time_pre + args.time_post
    counts = {"written": 0, "resumed": 0, "short": 0, "sample rate": 0, "unreadable": 0}
    t0 = time.time()

    with tables.open_file(args.out_path, mode="r+") as h5:
        for i, (_, event) in enumerate(selected.iterrows(), start=1):
            audio_path = recording_path(args.raw_root, event["Filename"])

            try:
                file_duration = librosa.get_duration(path=audio_path)
            except Exception as error:
                counts["unreadable"] += 1
                print(f"    unreadable {event['Filename']}: {error!r}", flush=True)
                continue

            t_start = window_start(event, file_duration, window_seconds)
            if t_start is None:
                counts["short"] += 1
                continue

            # the window follows from the detection alone, so a resumed run
            # recognises finished samples without reading any audio
            if (event["Filename"], round(t_start, 6)) in done:
                counts["resumed"] += 1
                continue

            try:
                audio, sr = librosa.load(audio_path, sr=None, offset=t_start,
                                         duration=window_seconds)
            except Exception as error:
                counts["unreadable"] += 1
                print(f"    unreadable {event['Filename']}: {error!r}", flush=True)
                continue

            if sr != EXPECTED_SR:
                counts["sample rate"] += 1
                continue

            if len(audio) < target_len:
                counts["short"] += 1
                continue

            window = audio[:target_len]

            spikes = encode_spikes(window, sr)

            birdcalledge.datastructure.append_sample(
                h5,
                event["Time_End"] - event["Time_start"],
                event["Confidence"],
                t_start,
                t_start + target_len / sr,
                event["Species_Name"],
                event["Filename"],
                args.confidence_threshold,
                np.nan,
                np.nan,
                BANDPASS_ORDER,
                np.nan,
                args.time_pre,
                args.time_post,
                sr,
                window,
                spikes.times,
                spikes.channels,
                group_name=args.group_name,
            )

            counts["written"] += 1
            elapsed = time.time() - t0
            remaining = (len(selected) - i) * elapsed / i
            print(f"[{i}/{len(selected)}] {event['Species_Name']}: {event['Filename']} "
                  f"at {t_start:.1f}s ({counts['written']} written, "
                  f"{remaining / 3600:.1f} h left)", flush=True)

    print(f"\n{counts['written']} samples written to {args.out_path}", flush=True)
    print("  %d already present, skipped: %d recording too short, %d wrong sample rate, "
          "%d unreadable" % (counts["resumed"], counts["short"],
                             counts["sample rate"], counts["unreadable"]), flush=True)


def summarise(selected):
    counts = selected.groupby("Species_Name").size().sort_values(ascending=False)
    print(f"\nselected {len(selected)} events from {counts.size} species, "
          f"{selected.Filename.nunique()} recordings", flush=True)
    print(counts.head(15).to_string(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", type=pathlib.Path, default=DETECTIONS)
    parser.add_argument("--raw-root", type=pathlib.Path, default=RAW_ROOT)
    parser.add_argument("--out-path", type=pathlib.Path, default=OUT_PATH)
    parser.add_argument("--group-name", default=GROUP_NAME)
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--min-detections", type=int, default=MIN_DETECTIONS)
    parser.add_argument("--max-per-species", type=int, default=MAX_PER_SPECIES)
    parser.add_argument("--max-per-recording", type=int, default=MAX_PER_RECORDING)
    parser.add_argument("--time-pre", type=float, default=TIME_PRE)
    parser.add_argument("--time-post", type=float, default=TIME_POST)
    parser.add_argument("--species", nargs="+", default=None,
                        help="only these species, default every species above the minimum")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many events, for test runs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the selection without extracting anything")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    detections = pd.read_csv(args.detections)
    detections = detections[detections.Confidence >= args.confidence_threshold]
    print(f"{len(detections)} detections of {detections.Species_Name.nunique()} species",
          flush=True)

    per_species = detections.groupby("Species_Name").size()
    species = per_species[per_species >= args.min_detections].index
    if args.species:
        species = [s for s in species if s in set(args.species)]
    detections = detections[detections.Species_Name.isin(species)]
    print(f"{len(species)} species with at least {args.min_detections} detections",
          flush=True)

    events = call_events(detections)
    print(f"{len(events)} call events after merging overlapping detections", flush=True)

    selected = select_events(events, args, rng)
    summarise(selected)

    if args.limit is not None:
        selected = selected.iloc[:args.limit]

    if args.dry_run:
        return

    extract(selected, args)


if __name__ == "__main__":
    main()
