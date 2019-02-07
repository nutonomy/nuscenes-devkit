# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np

from nuscenes.nuscenes import NuScenes


def create_splits_logs(nusc: NuScenes, verbose: bool=False) -> dict:
    """
    Returns the dataset splits of nuScenes.
    Note:
    - Previously this script included the teaser dataset splits. Since new scenes from those logs were added in the full
      dataset, that code is incompatible and was removed.
    - Currently the splits only cover the initial teaser release of nuScenes.
      This script will be completed upon release of the full dataset.
    :param nusc: NuScenes instance.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of logs in that split.
    """

    # Manually define splits.
    train = \
        ['n008-2018-05-21-11-06-59-0400', 'n015-2018-07-18-11-18-34+0800', 'n015-2018-07-18-11-50-34+0800',
         'n015-2018-07-24-10-42-41+0800', 'n015-2018-07-24-11-03-52+0800', 'n015-2018-07-24-11-13-19+0800',
         'n015-2018-07-24-11-22-45+0800', 'n015-2018-08-01-16-32-59+0800', 'n015-2018-08-01-16-41-59+0800']
    val = \
        ['n008-2018-08-01-15-16-36-0400', 'n015-2018-07-18-11-41-49+0800', 'n015-2018-07-18-11-07-57+0800']
    test = []

    # Check for duplicates.
    all = np.concatenate((train, val, test))
    assert len(all) == len(np.unique(all)), 'Error: Duplicate logs found in different splits!'

    # Optional: Print scene-level stats.
    if verbose:
        scene_lists = {'train': [], 'val': [], 'test': []}
        scenes = nusc.scene
        for split in scene_lists.keys():
            for scene in nusc.scene:
                if nusc.get('log', scene['log_token'])['logfile'] in locals()[split]:
                    scene_lists[split].append(scene['name'])
            print('%s: %d' % (split, len(scene_lists[split])))
            print('%s' % scene_lists[split])

    # Return splits.
    splits = {
        'train': train,
        'val': val,
        'test': test,
        'all': all
    }
    return splits


if __name__ == '__main__':
    # Run this to print the stats to stdout.
    nusc = NuScenes()
    create_splits_logs(nusc, verbose=True)
