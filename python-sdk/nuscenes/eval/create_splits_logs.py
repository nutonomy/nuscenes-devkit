# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np

from nuscenes.nuscenes import NuScenes


def create_splits_logs(nusc: NuScenes, verbose: bool=False) -> dict:
    """
    Returns the dataset splits of nuScenes.
    Note: Currently the splits only cover the initial teaser release of nuScenes.
          This script will be completed upon release of the full dataset.
    :param nusc: NuScenes instance.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of logs in that split.
    """

    # Manually define splits.
    teaser_train_logs = \
        ['n008-2018-05-21-11-06-59-0400', 'n015-2018-07-18-11-18-34+0800', 'n015-2018-07-18-11-50-34+0800',
         'n015-2018-07-24-10-42-41+0800', 'n015-2018-07-24-11-03-52+0800', 'n015-2018-07-24-11-13-19+0800',
         'n015-2018-07-24-11-22-45+0800', 'n015-2018-08-01-16-32-59+0800', 'n015-2018-08-01-16-41-59+0800']
    teaser_val_logs = \
        ['n008-2018-08-01-15-16-36-0400', 'n015-2018-07-18-11-41-49+0800', 'n015-2018-07-18-11-07-57+0800']

    # Define splits.
    nusc_logs = [record['logfile'] for record in nusc.log]
    teaser_train = teaser_train_logs
    teaser_val = teaser_val_logs
    teaser_test = []
    noteaser_train = []
    noteaser_val = []
    noteaser_test = []

    # Check for duplicates.
    all_check = np.concatenate((teaser_train, teaser_val, teaser_test, noteaser_train, noteaser_val, noteaser_test))
    assert len(all_check) == len(np.unique(all_check)), 'Error: Duplicate logs found in different splits!'

    # Assemble combined splits.
    train = sorted(teaser_train + noteaser_train)
    val = sorted(teaser_val + noteaser_val)
    test = sorted(teaser_test + noteaser_test)
    teaser = sorted(teaser_train + teaser_val + teaser_test)
    all = sorted(train + val + test)

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
    splits = {'teaser_train': teaser_train,
              'teaser_val': teaser_val,
              'teaser_test': teaser_test,
              'train': train,
              'val': val,
              'test': test,
              'teaser': teaser,
              'all': all}
    return splits


if __name__ == '__main__':
    # Run this to print the stats to stdout.
    nusc = NuScenes()
    create_splits_logs(nusc, verbose=True)
