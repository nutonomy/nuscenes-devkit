from typing import List, Dict

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes


def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes) -> Dict[str, Dict[str, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. This can be applied either to GT or predictions.
    :return: The tracks.
    """
    # Group annotations wrt scene and track_id.
    tracks = {}
    for sample_token in all_boxes.sample_tokens:

        # Init scene.
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        tracks[scene_token] = {}

        boxes: List[TrackingBox] = all_boxes.boxes[sample_token]
        for box in boxes:
            # Augment the boxes with timestamp. We will use timestamps to sort boxes in time later.
            box.timestamp = sample_record['timestamp']

            # Add box to tracks.
            if box.tracking_id not in tracks[scene_token].keys():
                tracks[scene_token][box.tracking_id] = []
            tracks[scene_token][box.tracking_id].append(box)

    # Make sure the tracks are sorted in time.
    for scene_token, scene in tracks.items():
        for tracking_id, track in scene.items():
            scene[tracking_id] = sorted(track, key=lambda _box: _box.timestamp)
        tracks[scene_token] = scene

    return tracks
