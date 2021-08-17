"""
Script to generate baselines for nuScenes-panoptic tasks.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""
import argparse
import os
import itertools
from typing import List
import zipfile

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.panoptic.evaluate import NuScenesPanopticEval
from nuscenes.eval.panoptic.get_panoptic_from_seg_det_or_track import generate_panoptic_labels


def prepare_files(method_names: List[str], root_dir: str) -> None:
    """
    Prepare the files containing the predictions of the various method names.
    :param method_names: A list of method names.
    :param root_dir: The directory where the predictions of the various methods are stored at.
    """
    for method_name in method_names:
        zip_path_to_predictions_by_method = os.path.join(root_dir, method_name + '.zip')
        dir_path_to_predictions_by_method = os.path.join(root_dir, method_name)
        assert os.path.exists(zip_path_to_predictions_by_method), 'Error: Zip file for method {} does not exist at {}.'\
            .format(method_name, zip_path_to_predictions_by_method)
        zip_ref = zipfile.ZipFile(zip_path_to_predictions_by_method, 'r')
        zip_ref.extractall(dir_path_to_predictions_by_method)
        zip_ref.close()


def get_prediction_json_path(prediction_dir: str) -> str:
    """
    Get the name of the json file in a directory (abort if there is more than one).
    :param prediction_dir: Path to the directory to check for the json file.
    :return: Absolute path to the json file.
    """

    files_in_dir = os.listdir(prediction_dir)
    files_in_dir = [f for f in files_in_dir if f.endswith('.json')]
    assert len(files_in_dir) == 1, 'Error: The submission .zip file must contain exactly one .json file.'

    prediction_json_path = os.path.join(prediction_dir, files_in_dir[0])
    assert os.path.exists(prediction_json_path), \
        'Error: JSON result file {} does not exist!'.format(prediction_json_path)

    return prediction_json_path


def main(out_dir: str,
         lidarseg_preds_dir: str,
         lidarseg_method_names: List[str],
         det_or_track_preds_dir: str,
         det_or_track_method_names: List[str],
         task: str = 'tracking',
         version: str = 'v1.0-test',
         dataroot: str = '/data/sets/nuscenes') -> None:
    """
    Create baselines for a given panoptic task by merging the predictions of lidarseg and either tracking or detection
    methods.
    :param out_dir: Path to save any output to.
    :param lidarseg_preds_dir: Path to the directory where the lidarseg predictions are stored.
    :param lidarseg_method_names: List of lidarseg method names.
    :param det_or_track_preds_dir: Path to the directory which contains the predictions from some methods to merge with
        those of lidarseg to create panoptic predictions of a particular task.
    :param det_or_track_method_names: List of tracking (or detection) method names to merge with lidarseg to create
        panoptic predictions.
    :param task: The task to create the panoptic predictions for and run evaluation on (either tracking or
        segmentation).
    :param version: Version of nuScenes to use (e.g. "v1.0", ...).
    :param dataroot: Path to the tables and data for the specified version of nuScenes.
    """
    # Prepare the required files.
    prepare_files(lidarseg_method_names, lidarseg_preds_dir)
    prepare_files(det_or_track_method_names, det_or_track_preds_dir)

    nusc = NuScenes(version=version, dataroot=dataroot)
    eval_set = nusc.version.split('-')[-1]

    # Get all possible pairwise permutations.
    baselines = list(itertools.product(lidarseg_method_names, det_or_track_method_names))
    print('There are {} baselines: {}'.format(len(baselines), baselines))

    # Get the predictions for the panoptic task at hand.
    for i, (lidarseg_method, det_or_track_method) in enumerate(baselines):
        print('{:02d}/{:02d}: Getting predictions for panoptic {} from {} and {}.'
              .format(i + 1, len(baselines), task, lidarseg_method, det_or_track_method))

        dir_to_save_panoptic_preds_to = os.path.join(out_dir, task, 'panoptic_predictions',
                                                     '{}_with_{}'.format(lidarseg_method, det_or_track_method))
        os.makedirs(dir_to_save_panoptic_preds_to, exist_ok=True)

        dir_of_lidarseg_method_preds = os.path.join(lidarseg_preds_dir, lidarseg_method)

        json_of_preds_by_det_or_track_method = get_prediction_json_path(
            os.path.join(det_or_track_preds_dir, det_or_track_method))

        generate_panoptic_labels(nusc,
                                 dir_of_lidarseg_method_preds,
                                 json_of_preds_by_det_or_track_method,
                                 eval_set=eval_set,
                                 task=task,
                                 out_dir=dir_to_save_panoptic_preds_to,
                                 verbose=True)
        print('Panoptic {} predictions saved at {}.'.format(task, dir_to_save_panoptic_preds_to))

        print('{:02d}/{:02d}: Evaluating predictions for panoptic {} from {} and {}.'
              .format(i + 1, len(baselines), task, lidarseg_method, det_or_track_method))
        dir_to_save_evaluation_results_to = os.path.join(out_dir, task, 'panoptic_eval_results',
                                                         '{}_with_{}'.format(lidarseg_method, det_or_track_method))
        os.makedirs(dir_to_save_evaluation_results_to, exist_ok=True)
        dir_of_panoptic_preds = dir_to_save_panoptic_preds_to
        evaluator = NuScenesPanopticEval(nusc=nusc,
                                         results_folder=dir_of_panoptic_preds,
                                         eval_set=eval_set,
                                         task=task,
                                         min_inst_points=15,
                                         out_dir=dir_to_save_evaluation_results_to,
                                         verbose=True)
        evaluator.evaluate()
        print('Evaluation for panoptic {} using predictions merged from {} and {} saved at {}.'
              .format(task, lidarseg_method, det_or_track_method, dir_to_save_evaluation_results_to))


if __name__ == '__main__':
    """
    Example usage:
        python baselines.py --out_dir ~/Desktop/logs/panoptic \
                            --lidarseg_preds_dir ~/Desktop/logs/panoptic/submissions/lidarseg \
                            --lidarseg_method_names 2D3DNet mit_han_lab \
                            --det_or_track_preds_dir ~/Desktop/logs/panoptic/submissions/detection \
                            --det_or_track_method_names crossfusion mmfusion polarstream \
                            --task segmentation              
    """
    parser = argparse.ArgumentParser(description='Create baselines for a panoptic task (tracking or segmentation).')
    parser.add_argument('--out_dir', type=str, help='Path to save any output to.')
    parser.add_argument('--lidarseg_preds_dir', type=str,
                        help='Path to the directory where the lidarseg predictions are stored.')
    parser.add_argument('--lidarseg_method_names', nargs='+', help='List of lidarseg method names.')
    parser.add_argument('--det_or_track_preds_dir', type=str,
                        help='Path to the directory which contains the predictions from some methods to merge with '
                             'those of lidarseg to create panoptic predictions of a particular task.')
    parser.add_argument('--det_or_track_method_names', nargs='+',
                        help='List of tracking (or detection) method names to merge with lidarseg to create panoptic '
                             'predictions.')
    parser.add_argument('--task', type=str, default='tracking',
                        help='The task to create the panoptic predictions for and run evaluation on (either tracking '
                             'or segmentation')
    parser.add_argument('--version', type=str, default='v1.0-test',
                        help='Version of nuScenes to use (e.g. "v1.0", ...).')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Path to the tables and data for the specified version of nuScenes.')

    args = parser.parse_args()
    print(args)

    main(out_dir=args.out_dir,
         lidarseg_preds_dir=args.lidarseg_preds_dir,
         lidarseg_method_names=args.lidarseg_method_names,
         det_or_track_preds_dir=args.det_or_track_preds_dir,
         det_or_track_method_names=args.det_or_track_method_names,
         task=args.task,
         version=args.version,
         dataroot=args.dataroot)
