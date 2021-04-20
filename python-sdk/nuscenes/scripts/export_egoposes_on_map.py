# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

"""
Exports an image for each map location with all the ego poses drawn on the map.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from nuscenes import NuScenes


def export_ego_poses(nusc: NuScenes, out_dir: str):
    """ Script to render where ego vehicle drives on the maps """

    # Load NuScenes locations
    locations = np.unique([log['location'] for log in nusc.log])

    # Create output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for location in locations:
        print('Rendering map {}...'.format(location))
        nusc.render_egoposes_on_map(location)
        out_path = os.path.join(out_dir, 'egoposes-{}.png'.format(location))
        plt.tight_layout()
        plt.savefig(out_path)


if __name__ == '__main__':

    # Settings.
    parser = argparse.ArgumentParser(description='Export all ego poses to an image.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', type=str, help='Directory where to save maps with ego poses.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')

    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose = bool(args.verbose)

    # Init.
    nusc_ = NuScenes(version=version, verbose=verbose, dataroot=dataroot)

    # Export ego poses
    export_ego_poses(nusc_, args.out_dir)
