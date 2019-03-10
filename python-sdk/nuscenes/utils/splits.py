# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Licensed under the Creative Commons [see licence.txt]

import numpy as np
from typing import Dict, List
import argparse

split_map = {
    'n008-2018-05-21-11-06-59-0400': 'train',
    'n008-2018-07-20-13-54-16-0400': 'val',
    'n008-2018-07-26-12-13-50-0400': 'train',
    'n008-2018-07-27-12-07-38-0400': 'train',
    'n008-2018-08-01-15-16-36-0400': 'val',
    'n008-2018-08-01-15-34-25-0400': 'test',
    'n008-2018-08-01-15-52-19-0400': 'train',
    'n008-2018-08-01-16-03-27-0400': 'test',
    'n008-2018-08-06-15-06-32-0400': 'train',
    'n008-2018-08-21-11-53-44-0400': 'train',
    'n008-2018-08-22-15-53-49-0400': 'val',
    'n008-2018-08-22-16-06-57-0400': 'test',
    'n008-2018-08-27-11-48-51-0400': 'train',
    'n008-2018-08-28-13-40-50-0400': 'train',
    'n008-2018-08-28-15-47-40-0400': 'test',
    'n008-2018-08-28-16-05-27-0400': 'test',
    'n008-2018-08-28-16-16-48-0400': 'train',
    'n008-2018-08-28-16-43-51-0400': 'val',
    'n008-2018-08-28-16-57-39-0400': 'train',
    'n008-2018-08-29-16-04-13-0400': 'train',
    'n008-2018-08-30-10-33-52-0400': 'val',
    'n008-2018-08-30-15-16-55-0400': 'train',
    'n008-2018-08-30-15-31-50-0400': 'val',
    'n008-2018-08-30-15-52-26-0400': 'train',
    'n008-2018-08-31-11-19-57-0400': 'train',
    'n008-2018-08-31-11-37-23-0400': 'val',
    'n008-2018-08-31-11-56-46-0400': 'train',
    'n008-2018-08-31-12-15-24-0400': 'test',
    'n008-2018-09-18-12-07-26-0400': 'train',
    'n008-2018-09-18-12-53-31-0400': 'train',
    'n008-2018-09-18-13-10-39-0400': 'train',
    'n008-2018-09-18-13-41-50-0400': 'train',
    'n008-2018-09-18-14-18-33-0400': 'test',
    'n008-2018-09-18-14-35-12-0400': 'val',
    'n008-2018-09-18-14-43-59-0400': 'train',
    'n008-2018-09-18-14-54-39-0400': 'train',
    'n008-2018-09-18-15-12-01-0400': 'val',
    'n008-2018-09-18-15-26-58-0400': 'train',
    'n015-2018-07-11-11-54-16+0800': 'val',
    'n015-2018-07-16-11-49-16+0800': 'val',
    'n015-2018-07-18-11-07-57+0800': 'train',
    'n015-2018-07-18-11-18-34+0800': 'train',
    'n015-2018-07-18-11-41-49+0800': 'val',
    'n015-2018-07-18-11-50-34+0800': 'train',
    'n015-2018-07-24-10-42-41+0800': 'val',
    'n015-2018-07-24-11-03-52+0800': 'train',
    'n015-2018-07-24-11-13-19+0800': 'train',
    'n015-2018-07-24-11-22-45+0800': 'train',
    'n015-2018-07-25-16-15-50+0800': 'test',
    'n015-2018-07-27-11-24-31+0800': 'train',
    'n015-2018-07-27-11-36-48+0800': 'train',
    'n015-2018-08-01-15-10-21+0800': 'train',
    'n015-2018-08-01-16-32-59+0800': 'train',
    'n015-2018-08-01-16-41-59+0800': 'train',
    'n015-2018-08-01-16-54-05+0800': 'test',
    'n015-2018-08-01-17-04-15+0800': 'train',
    'n015-2018-08-01-17-13-57+0800': 'train',
    'n015-2018-08-02-17-16-37+0800': 'val',
    'n015-2018-08-02-17-28-51+0800': 'train',
    'n015-2018-08-03-12-54-49+0800': 'test',
    'n015-2018-08-03-15-00-36+0800': 'train',
    'n015-2018-08-03-15-21-40+0800': 'train',
    'n015-2018-08-03-15-31-50+0800': 'train',
    'n015-2018-09-25-11-10-38+0800': 'train',
    'n015-2018-09-25-13-17-43+0800': 'val',
    'n015-2018-09-26-11-17-24+0800': 'train',
    'n015-2018-09-27-15-33-17+0800': 'train',
    'n015-2018-10-02-10-50-40+0800': 'val',
    'n015-2018-10-02-10-56-37+0800': 'train',
    'n015-2018-10-02-11-11-43+0800': 'test',
    'n015-2018-10-02-11-23-23+0800': 'test',
    'n015-2018-10-08-15-36-50+0800': 'val',
    'n015-2018-10-08-15-44-23+0800': 'val',
    'n015-2018-10-08-15-52-24+0800': 'train',
    'n015-2018-10-08-16-03-24+0800': 'train',
    'n015-2018-11-14-18-57-54+0800': 'train',
    'n015-2018-11-14-19-09-14+0800': 'train',
    'n015-2018-11-14-19-21-41+0800': 'test',
    'n015-2018-11-14-19-45-36+0800': 'test',
    'n015-2018-11-14-19-52-02+0800': 'test',
    'n015-2018-11-21-19-11-29+0800': 'train',
    'n015-2018-11-21-19-21-35+0800': 'val',
    'n015-2018-11-21-19-38-26+0800': 'train',
    'n015-2018-11-21-19-58-31+0800': 'train'
}


def create_splits_logs() -> Dict[str, List[str]]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added in the
          full dataset, that code is incompatible and was removed.
    :return: A mapping from split name to a list of logs in that split.
    """

    # Load splits.
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    for (log, split) in split_map.items():
        splits[split].append(log)

    # Check for duplicates.
    all_logs = np.concatenate(tuple(splits.values()))
    assert len(all_logs) == len(np.unique(all_logs)), 'Error: Duplicate logs found in different splits!'
    splits['all'] = all_logs

    return splits


def create_splits_scenes(nusc: 'NuScenes', verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping to scene names, rather than log names.
    :param nusc: NuScenes instance.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of logs in that split.
    """
    # Get log splits
    log_splits = create_splits_logs()

    # Map logs to scene names.
    scene_splits = {'train': [], 'val': [], 'test': []}
    for split in scene_splits.keys():
        for scene in nusc.scene:
            logfile = nusc.get('log', scene['log_token'])['logfile']
            if logfile in log_splits[split]:
                scene_splits[split].append(scene['name'])

        # Optional: Print scene-level stats.
        if verbose:
            print('%s: %d' % (split, len(scene_splits[split])))
            print('%s' % scene_splits[split])

    return scene_splits


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Prints out the scenes for each split.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose = bool(args.verbose)

    # Init.
    nusc = NuScenes(version=version, verbose=verbose, dataroot=dataroot)

    # Print the stats to stdout.
    create_splits_scenes(nusc, verbose=True)
