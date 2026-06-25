#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:03:57 2026

@author: daniel-muller
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d

def merge_intervals_pandas(df, start_col="Time_start", end_col="Time_End", conf_col="Confidence", merge_touching=False):
    """
    Merge intervals using a group labeling trick and groupby.
    """
    if df.empty:
        return df[[start_col, end_col, conf_col]].copy()

    d = df[[start_col, end_col, conf_col]].sort_values([start_col, end_col]).reset_index(drop=True)
    # running maximum of ends
    running_end = d[end_col].cummax()

    # define "new group" breaks
    if merge_touching:
        # start a new group if current start > previous running_end
        new_group = d[start_col] > running_end.shift(fill_value=-np.inf)
    else:
        # strict: start a new group if current start >= previous running_end
        new_group = d[start_col] >= running_end.shift(fill_value=-np.inf)

    grp = new_group.cumsum()

    out = (d.groupby(grp, as_index=False)
             .agg({start_col: 'min', end_col: 'max', conf_col: "max",}))

    return out


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        raise ValueError("highcut must be < Nyquist. Increase sample rate or reduce highcut.")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(y, sr, low, high, order=4):
    b, a = butter_bandpass(low, high, sr, order=order)
    # zero-phase filtering preserves onset/offset timing
    return filtfilt(b, a, y)


def robust_threshold(x, k=4.0):
    # Median and MAD (scaled to std)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad  # approx std for Gaussian
    return med + k * sigma

def detect_regions(envelope, sr, hop, threshold, min_event_dur, min_silence):
    """
    envelope: 1D envelope at sample rate "sr" (same as audio; here we keep per-sample envelope)
    hop: step between envelope samples in seconds (for our per-sample envelope -> 1/sr)
    threshold: scalar threshold
    min_event_dur: seconds
    min_silence: seconds
    """
    above = envelope > threshold
    # Find rising and falling edges
    edges = np.diff(above.astype(int))
    onsets = np.where(edges == 1)[0] + 1
    offsets = np.where(edges == -1)[0] + 1

    # Handle leading/trailing regions
    if above[0]:
        onsets = np.concatenate([[0], onsets])
    if above[-1]:
        offsets = np.concatenate([offsets, [len(above) - 1]])

    # Merge short gaps (hysteresis via min_silence)
    merged_onsets = []
    merged_offsets = []
    if len(onsets) > 0:
        cur_on = onsets[0]
        cur_off = offsets[0]
        for i in range(1, len(onsets)):
            """
            if gap < min_silence:
                # merge with current
                cur_off = offsets[i]
            else:
            """
            merged_onsets.append(cur_on)
            merged_offsets.append(cur_off)
            cur_on = onsets[i]
            cur_off = offsets[i]
        merged_onsets.append(cur_on)
        merged_offsets.append(cur_off)

    # Enforce minimum event duration
    final_on = []
    final_off = []
    for on, off in zip(merged_onsets, merged_offsets):
        dur = (off - on) * hop
        if dur >= min_event_dur:
            final_on.append(on)
            final_off.append(off)

    return final_on, final_off

def detect_regions_single(env, threshold):
    """
    env: 1D envelope inside a coarse interval
    threshold: scalar threshold value

    Returns:
        onset_index, offset_index (sample indices, ints)
        or (None, None) if no event detected
    """

    above = env > threshold
    if not np.any(above):
        return None, None

    # Rising / falling edges
    edges = np.diff(above.astype(int))
    onsets = np.where(edges == 1)[0] + 1
    offsets = np.where(edges == -1)[0] + 1

    # If the interval starts above threshold
    if above[0]:
        onsets = np.concatenate(([0], onsets))

    # If the interval ends above threshold
    if above[-1]:
        offsets = np.concatenate((offsets, [len(env) - 1]))

    # Enforce exactly one event:
    #   onset = first onset
    #   offset = last offset
    if len(onsets) == 0 or len(offsets) == 0:
        return None, None

    onset = onsets[0]
    offset = offsets[-1]

    return onset, offset
