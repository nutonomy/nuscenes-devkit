import argparse
import json
import os
import shutil

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper, get_samples_in_eval_set
from nuscenes.utils.data_classes import LidarPointCloud


def validate_submission(nusc: NuScenes,
                        results_folder: str,
                        eval_set: str,
                        verbose: bool = False,
                        zip_out: str = None) -> None:
    """
    Checks if a results folder is valid. The following checks are performed:
    - Check that the submission folder is according to that described in
      https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md
    - Check that the submission.json is of the following structure:
        {"meta": {"use_camera": false,
                  "use_lidar": true,
                  "use_radar": false,
                  "use_map": false,
                  "use_external": false}}
    - Check that each each lidar sample data in the evaluation set is present and valid.

    :param nusc: A NuScenes object.
    :param results_folder: Path to the folder.
    :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
    :param verbose: Whether to print messages during the evaluation.
    :param zip_out: Path to zip results_folder to, if provided.
    """
    mapper = LidarsegClassMapper(nusc)
    num_classes = len(mapper.coarse_name_2_coarse_idx_mapping)

    if verbose:
        print('Checking if folder structure of {} is correct...'.format(results_folder))

    # Check that {results_folder}/{eval_set} exists.
    results_meta_folder = os.path.join(results_folder, eval_set)
    assert os.path.exists(results_meta_folder), \
        'Error: The folder containing the submission.json ({}) does not exist.'.format(results_meta_folder)

    # Check that {results_folder}/{eval_set}/submission.json exists.
    submisson_json_path = os.path.join(results_meta_folder, 'submission.json')
    assert os.path.exists(submisson_json_path), \
        'Error: submission.json ({}) does not exist.'.format(submisson_json_path)

    # Check that {results_folder}/lidarseg/{eval_set} exists.
    results_bin_folder = os.path.join(results_folder, 'lidarseg', eval_set)
    assert os.path.exists(results_bin_folder), \
        'Error: The folder containing the .bin files ({}) does not exist.'.format(results_bin_folder)

    if verbose:
        print('\tPassed.')

    if verbose:
        print('Checking contents of {}...'.format(submisson_json_path))

    with open(submisson_json_path) as f:
        submission_meta = json.load(f)
        valid_meta = {"use_camera", "use_lidar", "use_radar", "use_map", "use_external"}
        assert valid_meta == set(submission_meta['meta'].keys()), \
            '{} must contain {}.'.format(submisson_json_path, valid_meta)
        for meta_key in valid_meta:
            meta_key_type = type(submission_meta['meta'][meta_key])
            assert meta_key_type == bool, 'Error: Value for {} should be bool, not {}.'.format(meta_key, meta_key_type)

    if verbose:
        print('\tPassed.')

    if verbose:
        print('Checking if all .bin files for {} exist and are valid...'.format(eval_set))
    sample_tokens = get_samples_in_eval_set(nusc, eval_set)
    for sample_token in tqdm(sample_tokens, disable=not verbose):
        sample = nusc.get('sample', sample_token)

        # Get the sample data token of the point cloud.
        sd_token = sample['data']['LIDAR_TOP']

        # Load the predictions for the point cloud.
        lidarseg_pred_filename = os.path.join(results_bin_folder, sd_token + '_lidarseg.bin')
        assert os.path.exists(lidarseg_pred_filename), \
            'Error: The prediction .bin file {} does not exist.'.format(lidarseg_pred_filename)
        lidarseg_pred = np.fromfile(lidarseg_pred_filename, dtype=np.uint8)

        # Check number of predictions for the point cloud.
        if len(nusc.lidarseg) > 0:  # If ground truth exists, compare the no. of predictions with that of ground truth.
            lidarseg_label_filename = os.path.join(nusc.dataroot, nusc.get('lidarseg', sd_token)['filename'])
            assert os.path.exists(lidarseg_label_filename), \
                'Error: The ground truth .bin file {} does not exist.'.format(lidarseg_label_filename)
            lidarseg_label = np.fromfile(lidarseg_label_filename, dtype=np.uint8)
            num_points = len(lidarseg_label)
        else:  # If no ground truth is available, compare the no. of predictions with that of points in a point cloud.
            pointsensor = nusc.get('sample_data', sd_token)
            pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
            pc = LidarPointCloud.from_file(pcl_path)
            points = pc.points
            num_points = points.shape[1]

        assert num_points == len(lidarseg_pred), \
            'Error: There are {} predictions for lidar sample data token {} ' \
            'but there are only {} points in the point cloud.'\
            .format(len(lidarseg_pred), sd_token, num_points)

        assert all((lidarseg_pred > 0) & (lidarseg_pred < num_classes)), \
            "Error: Array for predictions in {} must be between 1 and {} (inclusive)."\
            .format(lidarseg_pred_filename, num_classes - 1)

    if verbose:
        print('\tPassed.')

    if verbose:
        print('Results folder {} successfully validated!'.format(results_folder))

    # Zip up results folder if desired.
    if zip_out:
        assert os.path.exists(zip_out), \
            'Error: The folder {} to zip the results to does not exist.'.format(zip_out)

        results_zip = os.path.join(zip_out, os.path.basename(os.path.normpath(results_folder)))
        results_zip_name = shutil.make_archive(results_zip, 'zip', results_folder)
        if verbose:
            print('Results folder {} zipped to {}'.format(results_folder, results_zip_name))


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Check if a results folder is valid.')
    parser.add_argument('--result_path', type=str,
                        help='The path to the results folder.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to print to stdout.')
    parser.add_argument('--zip_out', type=str, default=None,
                        help='Path to zip the results folder to.')
    args = parser.parse_args()

    result_path_ = args.result_path
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = args.verbose
    zip_out_ = args.zip_out

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)
    validate_submission(nusc=nusc_,
                        results_folder=result_path_,
                        eval_set=eval_set_,
                        verbose=verbose_,
                        zip_out=zip_out_)
