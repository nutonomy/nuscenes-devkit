# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

"""
Exports a video of each scene (with annotations) to disk.
"""

import argparse
import os

from nuscenes import NuScenes


def export_videos(nusc: NuScenes, out_dir: str):
    """ Export videos of the images displayed in the images. """

    # Load NuScenes class
    scene_tokens = [s['token'] for s in nusc.scene]

    # Create output directory
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Write videos to disk
    for scene_token in scene_tokens:
        scene = nusc.get('scene', scene_token)
        print('Writing scene %s' % scene['name'])
        out_path = os.path.join(out_dir, scene['name']) + '.avi'
        if not os.path.exists(out_path):
            nusc.render_scene(scene['token'], out_path=out_path)


if __name__ == '__main__':

    # Settings.
    parser = argparse.ArgumentParser(description='Export all videos of annotations.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', type=str, help='Directory where to save videos.')
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

    # Export videos of annotations
    export_videos(nusc_, args.out_dir)
