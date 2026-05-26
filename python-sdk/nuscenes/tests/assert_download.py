# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.
# Extended with clearer validation output.

import argparse
import os

from tqdm import tqdm

from nuscenes import NuScenes


def _check_file_exists(file_path: str, label: str) -> None:
    """
    Raise a clear error if a required dataset file is missing.
    """
    if not os.path.exists(file_path):
        raise AssertionError(f'Error: Missing {label} at: {file_path}')


def verify_setup(nusc: NuScenes) -> None:
    """
    Script to verify that the nuScenes installation is complete.
    """
    print(f'Verifying nuScenes dataset at: {nusc.dataroot}')
    print(f'Version: {nusc.version}')

    print('Checking that sample_data files are complete...')
    for sd in tqdm(nusc.sample_data):
        file_path = os.path.join(nusc.dataroot, sd['filename'])
        _check_file_exists(file_path, 'sample_data')

    print('Checking that map files are complete...')
    for map_record in tqdm(nusc.map):
        file_path = os.path.join(nusc.dataroot, map_record['filename'])
        _check_file_exists(file_path, 'map')

    print('Dataset verification passed.')
    print(f'Checked {len(nusc.sample_data)} sample_data files and {len(nusc.map)} map files.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test that the installed dataset is complete.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataroot',
        type=str,
        default='/data/sets/nuscenes',
        help='Default nuScenes data directory.'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Whether to print to stdout.'
    )

    args = parser.parse_args()
    nusc_ = NuScenes(version=args.version, verbose=bool(args.verbose), dataroot=args.dataroot)
    verify_setup(nusc_)
