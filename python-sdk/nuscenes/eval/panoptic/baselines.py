import os
import itertools
from typing import List
import zipfile

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.panoptic.evaluate import NuScenesPanopticEval
from nuscenes.eval.panoptic.merge_lidarseg_and_tracking import generate_panoptic_labels \
    as get_panoptic_preds_from_lidarseg_and_tracking
from nuscenes.eval.panoptic.merge_lidarseg_and_detection import generate_panoptic_labels \
    as get_panoptic_preds_from_lidarseg_and_detection


def prepare_files(method_names: List[str], root_dir: str) -> None:
    """
    # TODO
    :param method_names:
    :param root_dir:
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
         lidarseg_preds_dir: str, lidarseg_method_names: List[str],
         to_merge_preds_dir: str = None, to_merge_method_names: List[str] = None,
         task: str = 'tracking',  # TODO or 'segmentation
         version: str = 'v1.0-test', dataroot: str = '/data/sets/nuscenes'):
    """
    # TODO
    :param out_dir: Path to save the evaluation for each baseline to.

    """
    # Prepare the required files.
    prepare_files(lidarseg_method_names, lidarseg_preds_dir)
    prepare_files(to_merge_method_names, to_merge_preds_dir)

    nusc = NuScenes(version=version, dataroot=dataroot)
    eval_set = nusc.version.split('-')[-1]

    # Get all possible pairwise permutations.
    baselines = list(itertools.product(lidarseg_method_names, to_merge_method_names))
    print('There are {} baselines: {}'.format(len(baselines), baselines))

    # Get the predictions for the panoptic task at hand.
    for i, (lidarseg_method, to_merge_method) in enumerate(baselines):
        print('{:02d}/{:02d}: Getting predictions for panoptic {} from {} and {}.'
              .format(i + 1, len(baselines), task, lidarseg_method, to_merge_method))

        dir_to_save_panoptic_preds_to = os.path.join(out_dir, task, 'panoptic_predictions',
                                                     '{}_with_{}'.format(lidarseg_method, to_merge_method))
        os.makedirs(dir_to_save_panoptic_preds_to, exist_ok=True)

        dir_of_lidarseg_method_preds = os.path.join(lidarseg_preds_dir, lidarseg_method)

        json_of_preds_by_to_merge_method = get_prediction_json_path(os.path.join(to_merge_preds_dir, to_merge_method))
        if task == 'tracking':
            get_panoptic_preds_from_lidarseg_and_tracking(nusc,
                                                          dir_of_lidarseg_method_preds,
                                                          json_of_preds_by_to_merge_method,
                                                          eval_set=eval_set,
                                                          out_dir=dir_to_save_panoptic_preds_to,
                                                          verbose=True)
        else:
            get_panoptic_preds_from_lidarseg_and_detection(nusc,
                                                           dir_of_lidarseg_method_preds,
                                                           json_of_preds_by_to_merge_method,
                                                           eval_set=eval_set,
                                                           out_dir=dir_to_save_panoptic_preds_to,
                                                           verbose=True)
        print('Panoptic {} predictions saved at {}.'.format(task, dir_to_save_panoptic_preds_to))

        print('{:02d}/{:02d}: Evaluation predictions for panoptic {} from {} and {}.'
              .format(i + 1, len(baselines), task, lidarseg_method, to_merge_method))
        dir_to_save_evaluation_results_to = os.path.join(out_dir, task, 'panoptic_eval_results',
                                                         '{}_with_{}'.format(lidarseg_method, to_merge_method))
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
              .format(task, lidarseg_method, to_merge_method, dir_to_save_evaluation_results_to))


if __name__ == '__main__':
    main(out_dir='/home/whye/Desktop/logs/panoptic/',
         lidarseg_preds_dir='/home/whye/Desktop/logs/panoptic/submissions/lidarseg',
         lidarseg_method_names=['2D3DNet', 'mit_han_lab'],

         # to_merge_preds_dir='/home/whye/Desktop/logs/panoptic/submissions/tracking',
         # to_merge_method_names=['TuSimple'],
         to_merge_preds_dir='/home/whye/Desktop/logs/panoptic/submissions/detection',
         to_merge_method_names=['crossfusion', 'mmfusion', 'polarstream'],

         task='segmentation'  # 'tracking'
         )
