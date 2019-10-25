from bisect import bisect
from typing import List, Dict

import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes


def interpolate_tracking_boxes(left_box: TrackingBox, right_box: TrackingBox, right_ratio: float) -> TrackingBox:
    def interp_list(left, right, rratio):
        return tuple((1.0-rratio) * np.array(left, dtype=float) + rratio * np.array(right, dtype=float))

    def interp_float(left, right, rratio):
        return (1.0 - rratio) * float(left) + rratio * float(right)

    return TrackingBox(sample_token=right_box.sample_token,
                       translation=interp_list(left_box.translation, right_box.translation, right_ratio),
                       size=interp_list(left_box.size, right_box.size, right_ratio),
                       rotation=interp_list(left_box.rotation, right_box.rotation, right_ratio),
                       velocity=interp_list(left_box.velocity, right_box.velocity, right_ratio),
                       ego_dist=interp_float(left_box.ego_dist, right_box.ego_dist, right_ratio),
                       tracking_id=right_box.tracking_id,
                       tracking_name=right_box.tracking_name)


def interpolate_tracks(tracks_by_timestamp: Dict[int, List[TrackingBox]]) -> Dict[int, List[TrackingBox]]:
    # NOTE 1: We are rediscovering an information that already exists in nuscenes.
    # NOTE 2: This interpolation does not take into account visibility. It would interpolate through occlusion.

    # Group tracks by id
    tracks_by_id = {}
    track_timestamps_by_id = {}
    for timestamp, tracking_boxes in tracks_by_timestamp.items():
        for tracking_box in tracking_boxes:
            if tracking_box.tracking_id not in tracks_by_id.keys():
                tracks_by_id[tracking_box.tracking_id] = []
                track_timestamps_by_id[tracking_box.tracking_id] = []
            tracks_by_id[tracking_box.tracking_id].append(tracking_box)
            track_timestamps_by_id[tracking_box.tracking_id].append(timestamp)

    # Interpolate missing timestamps for each track
    timestamps = tracks_by_timestamp.keys()
    for timestamp in timestamps:
        for tracking_id, track in tracks_by_id.items():
            if track_timestamps_by_id[tracking_id][0] <= timestamp <= track_timestamps_by_id[tracking_id][-1] and \
                    timestamp not in track_timestamps_by_id[tracking_id]:
                # Find the closest before and after entries
                right_ind = bisect(track_timestamps_by_id[tracking_id], timestamp)
                left_ind = right_ind - 1
                right_timestamp = track_timestamps_by_id[tracking_id][right_ind]
                left_timestamp = track_timestamps_by_id[tracking_id][left_ind]
                right_tracking_box = tracks_by_id[tracking_id][right_ind]
                left_tracking_box = tracks_by_id[tracking_id][left_ind]
                right_ratio = float(right_timestamp - timestamp)/(right_timestamp - left_timestamp)
                tracking_box = interpolate_tracking_boxes(left_tracking_box, right_tracking_box, right_ratio)
                tracks_by_timestamp[timestamp].append(tracking_box)

    return tracks_by_timestamp


def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes, eval_split: str) -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :param eval_split: The evaluation split for which we create tracks.
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

    # Init all scenes and timestamps to guarantee completeness.
    tracks = {}
    for scene_token in scene_tokens:
        # Init scene.
        if scene_token not in tracks:
            tracks[scene_token] = {}

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

    # Make sure the tracks are sorted in time.
    # This is always the case for GT, but may not be the case for predictions.
    for scene_token in tracks.keys():
        tracks[scene_token] = dict(sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))
        tracks[scene_token] = interpolate_tracks(tracks[scene_token])

    return tracks
