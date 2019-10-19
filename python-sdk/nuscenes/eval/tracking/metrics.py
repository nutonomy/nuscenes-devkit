from typing import Any
import numpy as np

DataFrame = Any


def track_initialization_duration(df: DataFrame, obj_frequencies: DataFrame) -> float:
    """
    Computes the track initialization duration, which is the duration from the first occurrance of an object to
    it's first correct detection (TP).
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
        matched = dfo[dfo.Type != 'MISS']

        if len(matched) == 0:
            # Consider only tracked objects.
            gap = 0
            missed_tracks += 1
        else:
            # Find the biggest gap. This looks more difficult as some samples may be occluded and therefore don't have
            # GT. In that case we assume that the gap size did not grow, rather than ending the gap or growing it.
            gap = 0  # The biggest gap found.
            cur_gap = 0  # Current gap.

            matches = set(dfo[dfo.Type == 'MATCH'].index.get_level_values(0).values)
            misses = set(dfo[dfo.Type == 'MISS'].index.get_level_values(0).values)
            first_index = dfo.index[0][0]
            last_index = dfo.index[-1][0]

            for i in range(first_index, last_index + 1):
                # Find biggest gap.
                if i in misses:
                    # Gap grows.
                    cur_gap += 1
                elif i in matches:
                    # Reset when matched.
                    gap = np.maximum(gap, cur_gap)
                    cur_gap = 0
                else:
                    # We have no GT here, so we make no assumptions and don't grow the gap.
                    pass

            gap = np.maximum(gap, cur_gap)
            assert gap <= len(misses)

        # Multiply number of sample differences with approx. sample period (0.5 sec).
        assert gap >= 0, 'Time difference should be larger than or equal to zero: %.2f'
        lgd += gap * 0.5

    matched_tracks = len(obj_frequencies) - missed_tracks
    if matched_tracks == 0:
        # Return nan if there are no matches.
        lgd = np.nan
    else:
        lgd = lgd / matched_tracks

    return lgd


def motap(df, num_matches: int, num_misses: int, num_switches: int, num_false_positives: int, num_objects: int) \
        -> float:
    """
    Initializes a MOTAP (MOTA') class which refers to the modified MOTA metric at https://www.nuscenes.org/tracking.
    Note that we use the measured recall, which is not identical to the hypothetical recall of the
    AMOTA/AMOTP thresholds.
    :param num_matches: The number of matches, aka. false positives.
    :param num_misses: The number of misses, aka. false negatives.
    :param num_switches: The number of identity switches.
    :param num_false_positives: The number of false positives.
    :param num_objects: The total number of objects of this class in the GT.
    :param recall: The current recall threshold.
    :return: The MOTA'.
    """
    recall = num_matches / num_objects
    nominator = num_misses + num_switches + num_false_positives - (1 - recall) * num_objects
    denominator = recall * num_objects
    if denominator == 0:
        motap = np.nan
    else:
        motap = 1 - nominator / denominator
        motap = np.maximum(0, motap)

    # Consistency checks to make sure that a positive MOTA also leads to a positive MOTAP.
    mota = 1 - (num_misses + num_switches + num_false_positives) / num_objects
    if mota > 0:
        assert motap > 0
    return motap


def mota_custom(df, num_misses: int, num_switches: int, num_false_positives: int, num_objects: int):
    """
    Multiple object tracker accuracy.
    Compared to the original MOTA definition, we clip values below 0.
    """
    mota = 1. - (num_misses + num_switches + num_false_positives) / num_objects
    mota = np.maximum(0, mota)
    return mota


def motp_custom(df, num_detections):
    """
    Multiple object tracker precision.
    Note: This function cannot have type hints as it breaks the introspection of motmetrics.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_detections: The number of detections.
    """
    # Note that the default motmetrics function throws a warning when num_detections == 0.
    if num_detections == 0:
        return np.nan
    return df.noraw['D'].sum() / num_detections


def faf_custom(df, num_false_positives, num_frames):
    """
    The average number of false alarms per frame
    Note: This function cannot have type hints as it breaks the introspection of motmetrics.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param num_false_positives: The number of false positives.
    :param num_frames: The number of frames.
    :return: Average FAF.
    """
    return num_false_positives / num_frames * 100
