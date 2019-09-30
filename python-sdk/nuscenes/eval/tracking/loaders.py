from typing import List, Dict

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes


def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes) -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. This can be applied either to GT or predictions.
    :param all_boxes:
    :param nusc: The NuScenes instance to load the sample information from.
    :return: The tracks.
    """
    # Group annotations wrt scene and track_id.
    tracks = {}

    for sample_token in all_boxes.sample_tokens:

        # Init scene.
        sample_record = nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        if scene_token not in tracks:
            tracks[scene_token] = {}

        boxes: List[TrackingBox] = all_boxes.boxes[sample_token]
        for box in boxes:
            # Augment the boxes with timestamp. We will use timestamps to sort boxes in time later.
            box.timestamp = sample_record['timestamp']

            # Add box to tracks.
            if box.timestamp not in tracks[scene_token].keys():
                tracks[scene_token][box.timestamp] = []
            tracks[scene_token][box.timestamp].append(box)

    # Make sure the tracks are sorted in time.
    for scene_token, scene in tracks.items():
        tracks[scene_token] = dict(sorted(tracks[scene_token].items(), key=lambda kv: kv[0]))

    return tracks
