from typing import Any
import numpy as np

DataFrame = Any


def track_initialization_duration(df: DataFrame, obj_frequencies: DataFrame):
    """
    Computes the track initialization duration, which is the duration from the first occurrance of an object to
    it's first correct detection (TP).
    :param df: Motmetrics dataframe that is required, but not used here.
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


def longest_gap_duration(df: DataFrame, obj_frequencies: DataFrame):
    """
    Computes the longest gap duration, which is the longest duration of any gaps in the detection of an object.
    :param df: Motmetrics dataframe that is required, but not used here.
    :param obj_frequencies: Dataframe with all object frequencies.
    :return: The longest gap duration.
    """
    # Return nan if the class is not in the GT.
    if len(obj_frequencies.index) == 0:
        return np.nan

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


class MOTAP:
    """
    Implements MOTAP aka MOTA'.
    The function is wrapped in a class to store the recall and manipulate it outside motmetrics.
    """
    def __init__(self):
        self.recall = np.nan

    def __call__(self, df, num_misses, num_switches, num_false_positives, num_objects):
        assert not np.isnan(self.recall)  # Check that current recall has been set.
        return MOTAP.motap(num_misses, num_switches, num_false_positives, num_objects, self.recall)

    @staticmethod
    def motap(num_misses: int, num_switches: int, num_false_positives: int, num_objects: int, recall: float):
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
