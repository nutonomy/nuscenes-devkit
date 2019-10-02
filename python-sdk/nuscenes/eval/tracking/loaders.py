from typing import List, Dict

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.nuscenes import NuScenes


def create_tracks(all_boxes: EvalBoxes, nusc: NuScenes) -> Dict[str, Dict[int, List[TrackingBox]]]:
    """
    Returns all tracks for all scenes. Samples within a track are sorted in chronological order.
    This can be applied either to GT or predictions.
    :param all_boxes: Holds all GT or predicted boxes.
    :param nusc: The NuScenes instance to load the sample information from.
    :return: The tracks.
    """
    # Init all scenes and timestamps to guarantee completeness.
    tracks = {}
    scene_tokens = [nusc.get('sample', st)['scene_token'] for st in all_boxes.sample_tokens]
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
        boxes: List[TrackingBox] = all_boxes.boxes[sample_token]
        for box in boxes:
            # Augment the boxes with timestamp. We will use timestamps to sort boxes in time later.
            box.timestamp = sample_record['timestamp']

            # Add box to tracks. The timestamps have been initialized above.
            tracks[scene_token][box.timestamp].append(box)

    return tracks
