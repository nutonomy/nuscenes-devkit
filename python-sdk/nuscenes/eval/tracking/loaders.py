# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

from bisect import bisect
from collections import defaultdict
from typing import List, Dict, DefaultDict

import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


def interpolate_tracking_boxes(left_box: TrackingBox, right_box: TrackingBox, right_ratio: float) -> TrackingBox:
    """
    Linearly interpolate box parameters between two boxes.
    :param left_box: A Trackingbox.
    :param right_box: Another TrackingBox
    :param right_ratio: Weight given to the right box.
    :return: The interpolated TrackingBox.
    """
    def interp_list(left, right, rratio):
        return tuple(
            (1.0 - rratio) * np.array(left, dtype=float)
            + rratio * np.array(right, dtype=float)
        )

    def interp_float(left, right, rratio):
        return (1.0 - rratio) * float(left) + rratio * float(right)

    # Interpolate quaternion.
    rotation = Quaternion.slerp(
        q0=Quaternion(left_box.rotation),
        q1=Quaternion(right_box.rotation),
        amount=right_ratio
    ).elements

    # Score will remain -1 for GT.
    tracking_score = interp_float(left_box.tracking_score, right_box.tracking_score, right_ratio)

    return TrackingBox(sample_token=right_box.sample_token,
                       translation=interp_list(left_box.translation, right_box.translation, right_ratio),
                       size=interp_list(left_box.size, right_box.size, right_ratio),
                       rotation=rotation,
                       velocity=interp_list(left_box.velocity, right_box.velocity, right_ratio),
                       ego_translation=interp_list(left_box.ego_translation, right_box.ego_translation,
                                                   right_ratio),  # May be inaccurate.
                       tracking_id=right_box.tracking_id,
                       tracking_name=right_box.tracking_name,
                       tracking_score=tracking_score)


def interpolate_tracks(tracks_by_timestamp: DefaultDict[int, List[TrackingBox]]) -> DefaultDict[int, List[TrackingBox]]:
    """
    Interpolate the tracks to fill in holes, especially since GT boxes with 0 lidar points are removed.
    This interpolation does not take into account visibility. It interpolates despite occlusion.
    :param tracks_by_timestamp: The tracks.
    :return: The interpolated tracks.
    """
    # Group tracks by id.
    tracks_by_id = defaultdict(list)
    track_timestamps_by_id = defaultdict(list)
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            tracks_by_id[tracking_box.tracking_id].append(tracking_box)
            track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

    # Interpolate missing timestamps for each track.
    timestamps = tracks_by_timestamp.keys()
    interpolate_count = 0
    for timestamp in timestamps:
        for tracking_id, track in tracks_by_id.items():
            if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                    timestamp not in track_timestamps_by_id[tracking_id]:

                # Find the closest boxes before and after this timestamp.
                right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                left_ind = right_ind - 1
                right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                right_tracking_box = tracks_by_id[tracking_id][right_ind]
                left_tracking_box = tracks_by_id[tracking_id][left_ind]
                right_ratio = float(right_timestamp - timestamp) / (right_timestamp - left_timestamp)

                # Interpolate.
                tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                interpolate_count += 1
                tracks_by_timestamp[timestamp].append(tracking_box)

    return tracks_by_timestamp


def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str, gt: bool) \
        -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :param eval_split: The evaluation split for which we create tracks.
    :param gt: Whether we are creating tracks for GT or predictions
    :return: The tracks.
    """
    # Only keep samples from this split.
    splits = create_splits_scenes()
    scene_tokens = set()
    for sample_token in all_boxes.sample_tokens:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene = nusc.get('scene', scene_token)
        if scene['name'] in splits[eval_split]:
            scene_tokens.add(scene_token)

    # Tracks are stored as dict {scene_token: {timestamp: List[TrackingBox]}}.
    tracks = defaultdict(lambda: defaultdict(list))

    # Init all scenes and timestamps to guarantee completeness.
    for scene_token in scene_tokens:
        # Init all timestamps in this scene.
        scene = nusc.get('scene', scene_token)
        cur_sample_token = scene['first_sample_token']
        while True:
            # Initialize array for current timestamp.
            cur_sample = nusc.get('sample', cur_sample_token)
            tracks[scene_token][cur_sample['timestamp']] = []

            # Abort after the last sample.
            if cur_sample_token == scene['last_sample_token']:
                break

            # Move to next sample.
            cur_sample_token = cur_sample['next']

    # Group annotations wrt scene and timestamp.
    for sample_token in all_boxes.sample_tokens:
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        tracks[scene_token][sample_record['timestamp']] = all_boxes.boxes[sample_token]

    # Replace box scores with track score (average box score). This only affects the compute_thresholds method and
    # should be done before interpolation to avoid diluting the original scores with interpolated boxes.
    if not gt:
        for scene_id, scene_tracks in tracks.items():
            # For each track_id, collect the scores.
            track_id_scores = defaultdict(list)
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    track_id_scores[box.tracking_id].append(box.tracking_score)

            # Compute average scores for each track.
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)

            # Apply average score to each box.
            for timestamp, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = track_id_avg_scores[box.tracking_id]

    # Interpolate GT and predicted tracks.
    for scene_token in tracks.keys():
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

        if not gt:
            # Make sure predictions are sorted in in time. (Always true for GT).
            tracks[scene_token] = defaultdict(list, sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

    return tracks
