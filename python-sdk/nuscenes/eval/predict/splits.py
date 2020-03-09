# nuScenes dev-kit.
# Code written by Freddy Boulton.

import json
import os
from itertools import chain
from typing import List

from nuscenes.utils.splits import create_splits_scenes

NUM_IN_VAL = 200

def get_prediction_challenge_split(split: str) -> List[str]:
    """
    Gets a list of {instance_token}_{sample_token} strings for each split
    :param split: One of 'mini_train', 'mini_val', 'train', 'val'.
    """
    if split not in {'mini_train', 'mini_val', 'train', 'train_val', 'val'}:
        raise ValueError("split must be one of (mini_train, mini_val, train, train_val, val)")
    
    if split == 'train_val':
        split_name = 'train'
    else:
        split_name = split

    path_to_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prediction_scenes.json")
    prediction_scenes = json.load(open(path_to_file, "r"))
    scenes = create_splits_scenes()
    scenes_for_split = scenes[split_name]
    
    if split == 'train':
        scenes_for_split = scenes_for_split[NUM_IN_VAL:]
    if split == 'train_val':
        scenes_for_split = scenes_for_split[:NUM_IN_VAL]

    return list(chain.from_iterable(map(lambda scene: prediction_scenes.get(scene, []), scenes_for_split)))
