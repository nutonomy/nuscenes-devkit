# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os

import numpy as np

from nuscenes.nuscenes import NuScenes


def create_splits_logs(nusc: NuScenes, verbose: bool=False) -> dict:
    """
    Returns the dataset splits of nuScenes.
    Note:
    - Previously this script included the teaser dataset splits. Since new scenes from those logs were added in the full
      dataset, that code is incompatible and was removed.
    :param nusc: NuScenes instance.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of logs in that split.
    """

    # Load splits from file.
    splits_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'splits.txt')
    with open(splits_path, 'r') as f:
        splits_data = f.readlines()
        splits_data = [s.strip() for s in splits_data]
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    for line in splits_data:
        tokens = line.split('\t')
        assert len(tokens) == 2
        splits[tokens[1]].append(tokens[0])

    # Check for duplicates.
    all = np.concatenate(tuple(splits.values()))
    assert len(all) == len(np.unique(all)), 'Error: Duplicate logs found in different splits!'
    splits['all'] = all

    # Optional: Print scene-level stats.
    if verbose:
        scene_lists = {'train': [], 'val': [], 'test': []}
        for split in scene_lists.keys():
            for scene in nusc.scene:
                if nusc.get('log', scene['log_token'])['logfile'] in splits[split]:
                    scene_lists[split].append(scene['name'])
            print('%s: %d' % (split, len(scene_lists[split])))
            print('%s' % scene_lists[split])

    return splits


if __name__ == '__main__':
    # Run this to print the stats to stdout.
    nusc = NuScenes()
    create_splits_logs(nusc, verbose=True)
