"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on:

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
from typing import Any

import numpy as np

DataFrame = Any


def track_initialization_duration(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Computes the track initialization duration, which is the duration from the first occurrence of an object to
    it's first correct detection (TP).
    Note that this True Positive metric is undefined if there are no matched tracks.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
    :return: The track initialization time.
    """
    tid = 0
    missed_tracks = 0
    for gt_tracking_id in obj_frequencies.index:
        # Get matches.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        notmiss = dfo[dfo.Type != 'MISS']

        if len(notmiss) == 0:
            # Consider only tracked objects.
            diff = 0
            missed_tracks += 1
        else:
            # Find the first time the object was detected and compute the difference to first time the object
            # entered the scene.
            diff = notmiss.index[0][0] - dfo.index[0][0]

        # Multiply number of sample differences with approx. sample period (0.5 sec).
        assert diff >= 0, 'Time difference should be larger than or equal to zero: %.2f'
        tid += diff * 0.5

    matched_tracks = len(obj_frequencies) - missed_tracks
    if matched_tracks == 0:
        # Return nan if there are no matches.
        return np.nan
    else:
        return tid / matched_tracks


def longest_gap_duration(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Computes the longest gap duration, which is the longest duration of any gaps in the detection of an object.
    Note that this True Positive metric is undefined if there are no matched tracks.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Dataframe with all object frequencies.
    :return: The longest gap duration.
    """
    # Return nan if the class is not in the GT.
    if len(obj_frequencies.index) == 0:
        return np.nan

    lgd = 0
    missed_tracks = 0
    for gt_tracking_id in obj_frequencies.index:
        # Find the frame_ids object is tracked and compute the gaps between those. Take the maximum one for longest gap.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        matched = set(dfo[dfo.Type != 'MISS'].index.get_level_values(0).values)

        if len(matched) == 0:
            # Ignore untracked objects.
            gap = 0
            missed_tracks += 1
        else:
            # Find the biggest gap.
            # Note that we don't need to deal with FPs within the track as the GT is interpolated.
            gap = 0  # The biggest gap found.
            cur_gap = 0  # Current gap.
            first_index = dfo.index[0][0]
            last_index = dfo.index[-1][0]

            for i in range(first_index, last_index + 1):
                if i in matched:
                    # Reset when matched.
                    gap = np.maximum(gap, cur_gap)
                    cur_gap = 0
                else:  # Grow gap when missed.
                    # Gap grows.
                    cur_gap += 1

            gap = np.maximum(gap, cur_gap)

        # Multiply number of sample differences with approx. sample period (0.5 sec).
        assert gap >= 0, 'Time difference should be larger than or equal to zero: %.2f'
        lgd += gap * 0.5

    # Average LGD over the number of tracks.
    matched_tracks = len(obj_frequencies) - missed_tracks
    if matched_tracks == 0:
        # Return nan if there are no matches.
        lgd = np.nan
    else:
        lgd = lgd / matched_tracks

    return lgd


def motar(df: DataFrame, num_matches: int, num_misses: int, num_switches: int, num_false_positives: int,
          num_objects: int, alpha: float = 1.0) -> float:
    """
    Initializes a MOTAR class which refers to the modified MOTA metric at https://www.nuscenes.org/tracking.
    Note that we use the measured recall, which is not identical to the hypothetical recall of the
    AMOTA/AMOTP thresholds.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_matches: The number of matches, aka. false positives.
    :param num_misses: The number of misses, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :param alpha: MOTAR weighting factor (previously 0.2).
    :return: The MOTAR or nan if there are no GT objects.
    """
    recall = num_matches / num_objects
    nominator = (num_misses + num_switches + num_false_positives) - (1 - recall) * num_objects
    denominator = recall * num_objects
    if denominator == 0:
        motar_val = np.nan
    else:
        motar_val = 1 - alpha * nominator / denominator
        motar_val = np.maximum(0, motar_val)

    return motar_val


def mota_custom(df: DataFrame, num_misses: int, num_switches: int, num_false_positives: int, num_objects: int) -> float:
    """
    Multiple object tracker accuracy.
    Based on py-motmetric's mota function.
    Compared to the original MOTA definition, we clip values below 0.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_misses: The number of misses, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :return: The MOTA or 0 if below 0.
    """
    mota = 1. - (num_misses + num_switches + num_false_positives) / num_objects
    mota = np.maximum(0, mota)
    return mota


def motp_custom(df: DataFrame, num_detections: float) -> float:
    """
    Multiple object tracker precision.
    Based on py-motmetric's motp function.
    Additionally we check whether there are any detections.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_detections: The number of detections.
    :return: The MOTP or 0 if there are no detections.
    """
    # Note that the default motmetrics function throws a warning when num_detections == 0.
    if num_detections == 0:
        return np.nan
    return df.noraw['D'].sum() / num_detections


def faf(df: DataFrame, num_false_positives: float, num_frames: float) -> float:
    """
    The average number of false alarms per frame.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_false_positives: The number of false positives.
    :param num_frames: The number of frames.
    :return: Average FAF.
    """
    return num_false_positives / num_frames * 100


def num_fragmentations_custom(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Total number of switches from tracked to not tracked.
    Based on py-motmetric's num_fragmentations function.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
    :return: The number of fragmentations.
    """
    fra = 0
    for o in obj_frequencies.index:
        # Find first and last time object was not missed (track span). Then count
        # the number switches from NOT MISS to MISS state.
        dfo = df.noraw[df.noraw.OId == o]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            continue
        first = notmiss.index[0]
        last = notmiss.index[-1]
        diffs = dfo.loc[first:last].Type.apply(lambda x: 1 if x == 'MISS' else 0).diff()
        fra += diffs[diffs == 1].count()

    return fra
