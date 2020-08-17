# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import argparse
import os

from tqdm import tqdm

from nuimages import NuImages


def verify_setup(nuim: NuImages):
    """
    Script to verify that the nuImages installation is complete.
    Note that this may take several minutes or hours.
    """

    # Check that each sample_data file exists.
    print('Checking that sample_data files are complete...')
    for sd in tqdm(nuim.sample_data):
        file_path = os.path.join(nuim.dataroot, sd['filename'])
        assert os.path.exists(file_path), 'Error: Missing sample_data at: %s' % file_path


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Test that the installed dataset is complete.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages',
                        help='Default nuImages data directory.')
    parser.add_argument('--version', type=str, default='v1.0-train',
                        help='Which version of the nuImages dataset to evaluate on, e.g. v1.0-train.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')

    args = parser.parse_args()
    dataroot = args.dataroot
    version = args.version
    verbose = bool(args.verbose)

    # Init.
    nuim_ = NuImages(version=version, verbose=verbose, dataroot=dataroot)

    # Verify data blobs.
    verify_setup(nuim_)
