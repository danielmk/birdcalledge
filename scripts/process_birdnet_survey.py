# -*- coding: utf-8 -*-
"""
Run BirdNET over the whole OKEON raw acoustic monitoring archive.

The archive is organised as SITE/YEAR/SITE_YYYYMMDD_HHMMSS.flac and holds
roughly 20000 ten-minute recordings (~640 GB). At the measured throughput of
~60x realtime on this machine a full pass takes on the order of two days, so
the run is written to be resumable and splittable:

  * one detection CSV per recording, mirroring the raw directory layout, so a
    finished recording is never analysed twice (delete a CSV to redo it),
  * --sites restricts the run to individual sites, which allows several
    machines or several sessions to share the work,
  * --merge concatenates the per-file CSVs into a single table.

The CSV columns follow 2sp_detect_forDaniel.csv (Filename, Time_start,
Time_End, Species_Name, Confidence, Site, DateTime, Logit_conf) so the
detections are a drop-in replacement for the existing single-species table in
the downstream extraction scripts. Species_Name is the common name and
Scientific_Name is added next to it.

Run with the birdnet conda environment, e.g.

    conda run -n birdnet python scripts/process_birdnet_survey.py --sites GESASHIOP

@author: Daniel
"""

import argparse
import collections
import csv
import os
import pathlib
import re
import threading
import time

import numpy as np
import pandas as pd

import birdnet

RAW_ROOT = pathlib.Path(r"Y:\danielmk\okeon")
OUT_ROOT = pathlib.Path(r"Y:\danielmk\okeon_birdnet")

AUDIO_SUFFIXES = (".flac", ".wav", ".mp3")

# BIRDNET MODEL
MODEL_TYPE = "acoustic"
MODEL_VERSION = "3.0"
MODEL_BACKEND = "pt"

# DETECTION PARAMETERS
TOP_K = 5
MIN_CONFIDENCE = 0.8  # as in process_sample_extraction.py

# INFERENCE PERFORMANCE
# 4 workers / 2 producers measured fastest on the 16 core Xeon (~60x realtime);
# more workers oversubscribe the cores and get slower.
N_WORKERS = 4
N_PRODUCERS = 2
BATCH_SIZE = 16
CHUNK_SIZE = 32  # files handed to one session.run call, bounds memory

CSV_COLUMNS = ["Filename", "Time_start", "Time_End", "Species_Name",
               "Scientific_Name", "Confidence", "Logit_conf", "Site", "DateTime"]

FILENAME_PATTERN = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def find_audio_files(raw_root, sites=None):
    """All recordings below raw_root, optionally restricted to site folders."""
    roots = [raw_root / site for site in sites] if sites else [raw_root]

    files = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"no such directory: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                files.append(path)

    return sorted(files)


def output_path(audio_path, raw_root, out_root):
    """Detection CSV mirroring the position of audio_path in the archive."""
    return out_root / audio_path.relative_to(raw_root).with_suffix(".csv")


def parse_filename(filename):
    """Site and recording time from SITE_YYYYMMDD_HHMMSS, ('', '') if unusual."""
    match = FILENAME_PATTERN.match(pathlib.Path(filename).stem)
    if match is None:
        return "", ""

    stamp = pd.to_datetime(match["date"] + match["time"], format="%Y%m%d%H%M%S",
                           errors="coerce")
    if pd.isna(stamp):
        return match["site"], ""

    return match["site"], stamp.strftime("%Y-%m-%d %H:%M:%S")


def detections_to_frame(result, audio_path):
    """BirdNET result of a single file as a 2sp_detect_forDaniel-style table."""
    df = result.to_dataframe()

    site, date_time = parse_filename(audio_path.name)

    names = [str(name) for name in df["species_name"]]
    split = [name.split("_", 1) if "_" in name else ["", name] for name in names]

    confidence = df["confidence"].to_numpy(dtype="float64")
    clipped = np.clip(confidence, 1e-7, 1.0 - 1e-7)

    return pd.DataFrame({
        "Filename": [audio_path.name] * len(df),
        "Time_start": df["start_time"].to_numpy(dtype="float64"),
        "Time_End": df["end_time"].to_numpy(dtype="float64"),
        "Species_Name": [pair[1] for pair in split],
        "Scientific_Name": [pair[0] for pair in split],
        "Confidence": confidence,
        "Logit_conf": np.log(clipped / (1.0 - clipped)),
        "Site": [site] * len(df),
        "DateTime": [date_time] * len(df),
    }, columns=CSV_COLUMNS)


def write_frame(frame, dest):
    """Write dest atomically, so an interrupted run leaves no partial CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, dest)


class DetectionWriter:
    """Persists each file result as soon as BirdNET finishes that file."""

    def __init__(self, raw_root, out_root, failure_log, n_total):
        self.raw_root = raw_root
        self.out_root = out_root
        self.failure_log = failure_log
        self.n_total = n_total

        self.lock = threading.Lock()
        self.n_done = 0
        self.n_failed = 0
        self.n_detections = 0
        self.audio_seconds = 0.0
        self.t_start = time.time()

    def __call__(self, result):
        """on_file_complete callback, invoked from a BirdNET worker thread."""
        try:
            self._store(result)
        except Exception as error:  # a raising callback would cancel the run
            print(f"    writing the result failed: {error!r}", flush=True)

    def _store(self, result):
        audio_path = pathlib.Path(result.inputs[0])

        if len(result.unprocessable_inputs) > 0:
            self.quarantine(audio_path, "unreadable")
            return

        frame = detections_to_frame(result, audio_path)
        write_frame(frame, output_path(audio_path, self.raw_root, self.out_root))

        with self.lock:
            self.n_done += 1
            self.n_detections += len(frame)
            self.audio_seconds += float(sum(result.input_durations))
            self._report(audio_path, len(frame))

    def quarantine(self, audio_path, reason):
        """Drop a recording BirdNET cannot decode and note it for later."""
        with self.lock:
            self.n_failed += 1
            self.failure_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failure_log, "a", encoding="utf-8") as handle:
                handle.write(f"{audio_path}\t{reason}\n")

        print(f"    skipped, {reason}: {audio_path}", flush=True)

    def _report(self, audio_path, n_rows):
        elapsed = time.time() - self.t_start
        speed = self.audio_seconds / elapsed if elapsed else 0.0
        remaining = (self.n_total - self.n_done) * elapsed / max(self.n_done, 1)

        print(f"[{self.n_done}/{self.n_total}] {audio_path.name}: {n_rows} detections "
              f"({speed:.0f}x realtime, {remaining / 3600:.1f} h left)", flush=True)


def analyse(files, args):
    """Run BirdNET over files, writing one CSV per recording."""
    print(f"loading birdnet {MODEL_TYPE} {MODEL_VERSION} ({MODEL_BACKEND})", flush=True)
    model = birdnet.load(MODEL_TYPE, MODEL_VERSION, MODEL_BACKEND)
    print(f"model covers {model.n_species} species", flush=True)

    writer = DetectionWriter(args.raw_root, args.out_root,
                             args.out_root / "unprocessable_files.txt", len(files))

    queue = collections.deque(files)

    # A recording BirdNET cannot decode kills the producer process and with it the
    # whole session (a truncated FLAC raises "flac decoder lost sync" halfway
    # through the file, which the unprocessable-input path does not catch). Restart
    # the session on such a failure and retry the lost chunk one recording at a
    # time, so the offender can be identified, quarantined and stepped over.
    n_solo = 0

    while queue:
        chunk = []
        try:
            with model.predict_session(
                top_k=args.top_k,
                default_confidence_threshold=args.min_confidence,
                n_workers=args.workers,
                n_producers=args.producers,
                batch_size=args.batch_size,
                device=args.device,
                on_file_complete=writer,
            ) as session:
                while queue:
                    size = 1 if n_solo else args.chunk_size
                    chunk = [queue.popleft() for _ in range(min(size, len(queue)))]
                    session.run(chunk)
                    n_solo = max(n_solo - 1, 0)
        except Exception as error:
            if not chunk:  # the session itself would not start, nothing to skip
                raise

            if len(chunk) == 1:
                # culprit found, the rest of the chunk can go back to full speed
                writer.quarantine(chunk[0], repr(error))
                n_solo = 0
            else:
                retry = [f for f in chunk
                         if not output_path(f, args.raw_root, args.out_root).exists()]
                queue.extendleft(reversed(retry))
                n_solo = len(retry)
                print(f"    session cancelled ({error}), retrying {len(retry)} "
                      f"recordings one by one", flush=True)

    elapsed = time.time() - writer.t_start
    print(f"\nanalysed {writer.n_done} recordings ({writer.audio_seconds / 3600:.1f} h "
          f"of audio) in {elapsed / 3600:.2f} h, {writer.n_detections} detections, "
          f"{writer.n_failed} skipped", flush=True)


def merge(out_root, dest):
    """Concatenate all per-file CSVs below out_root into a single table."""
    if not out_root.is_dir():
        raise FileNotFoundError(f"no detections to merge, no such directory: {out_root}")

    dest = dest.resolve()
    sources = [s for s in sorted(out_root.rglob("*.csv")) if s.resolve() != dest]
    dest.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(dest, "w", encoding="utf-8", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(CSV_COLUMNS)

        for i, source in enumerate(sources, start=1):
            frame = pd.read_csv(source)
            if len(frame):
                out.writerows(frame[CSV_COLUMNS].itertuples(index=False))
                n_rows += len(frame)
            if i % 500 == 0:
                print(f"  merged {i}/{len(sources)} files, {n_rows} rows", flush=True)

    print(f"wrote {n_rows} detections from {len(sources)} recordings to {dest}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=pathlib.Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=pathlib.Path, default=OUT_ROOT)
    parser.add_argument("--sites", nargs="+", default=None,
                        help="site folders to analyse, default all")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many recordings, for test runs")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo recordings that already have a CSV")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--producers", type=int, default=N_PRODUCERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--merge", type=pathlib.Path, default=None,
                        help="concatenate existing CSVs into this file and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.merge is not None:
        merge(args.out_root, args.merge)
        return

    files = find_audio_files(args.raw_root, args.sites)
    print(f"found {len(files)} recordings below {args.raw_root}", flush=True)

    if not args.overwrite:
        files = [f for f in files
                 if not output_path(f, args.raw_root, args.out_root).exists()]
        print(f"{len(files)} still without detections", flush=True)

    if args.limit is not None:
        files = files[:args.limit]

    if not files:
        print("nothing to do", flush=True)
        return

    analyse(files, args)


if __name__ == "__main__":  # required, birdnet spawns worker processes
    main()
