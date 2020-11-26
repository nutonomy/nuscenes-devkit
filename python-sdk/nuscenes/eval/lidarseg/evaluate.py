import argparse
import json
import os
from typing import Dict

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegClassMapper, ConfusionMatrix, get_samples_in_eval_set, load_bin_file


class LidarSegEval:
    """
    This is the official nuScenes-lidarseg evaluation code.
    Results are written to the provided output_dir.

    nuScenes-lidarseg uses the following metrics:
    - Mean Intersection-over-Union (mIOU): We use the well-known IOU metric, which is defined as TP / (TP + FP + FN).
                                           The IOU score is calculated separately for each class, and then the mean is
                                           computed across classes. Note that in the challenge, index 0 is ignored in
                                           the calculation.
    - Frequency-weighted IOU (FWIOU): Instead of taking the mean of the IOUs across all the classes, each IOU is
                                      weighted by the point-level frequency of its class. Note that in the challenge,
                                      index 0 is ignored in the calculation. FWIOU is not used for the challenge.

    We assume that:
    - For each pointcloud, the prediction for every point is present in a .bin file, in the same order as that of the
      points stored in the corresponding .bin file.
    - The naming convention of the .bin files containing the predictions for a single point cloud is:
        <lidar_sample_data_token>_lidarseg.bin
    - The predictions are between 1 and 16 (inclusive); 0 is the index of the ignored class.

    Please see https://www.nuscenes.org/lidar-segmentation for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 results_folder: str,
                 eval_set: str,
                 verbose: bool = False):
        """
        Initialize a LidarSegEval object.
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param verbose: Whether to print messages during the evaluation.
        """
        # Check there are ground truth annotations.
        assert len(nusc.lidarseg) > 0, 'Error: No ground truth annotations found in {}.'.format(nusc.version)

        # Check results folder exists.
        self.results_folder = results_folder
        self.results_bin_folder = os.path.join(results_folder, 'lidarseg', eval_set)
        assert os.path.exists(self.results_bin_folder), \
            'Error: The folder containing the .bin files ({}) does not exist.'.format(self.results_bin_folder)

        self.nusc = nusc
        self.results_folder = results_folder
        self.eval_set = eval_set
        self.verbose = verbose

        self.mapper = LidarsegClassMapper(self.nusc)
        self.ignore_idx = self.mapper.ignore_class['index']
        self.id2name = {idx: name for name, idx in self.mapper.coarse_name_2_coarse_idx_mapping.items()}
        self.num_classes = len(self.mapper.coarse_name_2_coarse_idx_mapping)

        if self.verbose:
            print('There are {} classes.'.format(self.num_classes))

        self.global_cm = ConfusionMatrix(self.num_classes, self.ignore_idx)

        self.sample_tokens = get_samples_in_eval_set(self.nusc, self.eval_set)
        if self.verbose:
            print('There are {} samples.'.format(len(self.sample_tokens)))

    def evaluate(self) -> Dict:
        """
        Performs the actual evaluation.
        :return: A dictionary containing the evaluated metrics.
        """
        for sample_token in tqdm(self.sample_tokens, disable=not self.verbose):
            sample = self.nusc.get('sample', sample_token)

            # Get the sample data token of the point cloud.
            sd_token = sample['data']['LIDAR_TOP']

            # Load the ground truth labels for the point cloud.
            lidarseg_label_filename = os.path.join(self.nusc.dataroot,
                                                   self.nusc.get('lidarseg', sd_token)['filename'])
            lidarseg_label = load_bin_file(lidarseg_label_filename)

            lidarseg_label = self.mapper.convert_label(lidarseg_label)

            # Load the predictions for the point cloud.
            lidarseg_pred_filename = os.path.join(self.results_folder, 'lidarseg',
                                                  self.eval_set, sd_token + '_lidarseg.bin')
            lidarseg_pred = load_bin_file(lidarseg_pred_filename)

            # Get the confusion matrix between the ground truth and predictions.
            # Update the confusion matrix for the sample data into the confusion matrix for the eval set.
            self.global_cm.update(lidarseg_label, lidarseg_pred)

        iou_per_class = self.global_cm.get_per_class_iou()
        miou = self.global_cm.get_mean_iou()
        freqweighted_iou = self.global_cm.get_freqweighted_iou()

        # Put everything nicely into a dict.
        results = {'iou_per_class': {self.id2name[i]: class_iou for i, class_iou in enumerate(iou_per_class)},
                   'miou': miou,
                   'freq_weighted_iou': freqweighted_iou}

        # Print the results if desired.
        if self.verbose:
            print("======\nnuScenes-lidarseg evaluation for {}".format(self.eval_set))
            print(json.dumps(results, indent=4, sort_keys=False))
            print("======")

        return results


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes-lidarseg results.')
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
    args = parser.parse_args()

    result_path_ = args.result_path
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = args.verbose

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEval(nusc_, result_path_, eval_set=eval_set_, verbose=verbose_)
    evaluator.evaluate()
