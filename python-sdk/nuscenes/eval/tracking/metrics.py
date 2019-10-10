from typing import Any
import numpy as np


def track_initialization_duration(df: Any, obj_frequencies: Any) -> float:
    """
    Computes the track initialization duration, which is the duration from the first occurrance of an object to
    it's first correct detection (TP).
    :param df:
    :param obj_frequencies: Stores the GT tracking_ids and their frequencies.
    :return: The track initialization time.
    """
    tid = 0
    for gt_tracking_id in obj_frequencies.index:
        # Get matches.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        notmiss = dfo[dfo.Type != 'MISS']

        if len(notmiss) == 0:
            # For missed objects return the length of the track.
            diff = dfo.index[-1][0] - dfo.index[0][0]
        else:
            # Find the first time the object was detected and compute the difference to first time the object
            # entered the scene.
            diff = notmiss.index[0][0] - dfo.index[0][0]
        assert diff >= 0, 'Time difference should be larger than or equal to zero'
        # Multiply number of sample differences with sample period (0.5 sec)
        tid += float(diff) * 0.5
    return tid / len(obj_frequencies)


def longest_gap_duration(df, obj_frequencies):
    gap = 0
    for gt_tracking_id in obj_frequencies.index:
        # Find the frame_ids object is tracked and compute the gaps between those. Take the maximum one for longest
        # gap.
        dfo = df.noraw[df.noraw.OId == gt_tracking_id]
        notmiss = dfo[dfo.Type != 'MISS']
        if len(notmiss) == 0:
            # For missed objects return the length of the track.
            diff = dfo.index[-1][0] - dfo.index[0][0]
        else:
            diff = notmiss.index.get_level_values(0).to_series().diff().max() - 1
        if np.isnan(diff):
            diff = 0
        assert diff >= 0, 'Time difference should be larger than or equal to zero {0:f}'.format(diff)
        gap += diff * 0.5
    return gap / len(obj_frequencies)


def motap(num_misses: int, num_switches: int, num_false_positives: int, num_objects: int, recall: float) -> float:
    """
    Initializes a MOTAP (MOTA') class which refers to the modified MOTA metric at https://www.nuscenes.org/tracking.
    :param num_misses: The number of missed, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :param recall: The current recall threshold.
    :return: The MOTA'.
    """
    nominator = num_misses + num_switches + num_false_positives + (1 - recall) * num_objects
    denominator = recall * num_objects
    if denominator == 0:
        motap = np.nan
    else:
        motap = 1 - nominator / denominator
        motap = np.maximum(0, motap)

    return motap


def motp_custom(df, num_detections):
    """Multiple object tracker precision."""
    # Note that the default motmetrics function throws a warning when num_detections == 0.
    if num_detections == 0:
        return np.nan
    return df.noraw['D'].sum() / num_detections


def faf_custom(df, num_false_positives, num_frames):
    return num_false_positives / num_frames * 100
