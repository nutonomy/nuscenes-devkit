import argparse
import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes

from nuscenes.eval.lidarseg.utils import LidarsegChallengeAdaptor


class LidarSegEval:
    """
    This is the official nuScenes lidar segmentation evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following lidar segmentation metrics:
    - Mean Intersection-over-Union (mIOU): We use the well-known IOU metric, which is defined as TP / (TP + FP + FN).
                                           The IOU score is calculated separately for each class, and then the mean is
                                           computed across classes.

    We assume that:
    - For each pointcloud, the label for every point is present in a .bin file, in the same order as that of the points
      stored in the corresponding .bin file.
    - The naming convention of the .bin files containing the labels for a single point cloud is:
        <lidar_sample_data_token>_lidarseg.bin
    - The labels are between 0 and 16 (inclusive), where 0 is the index of the ignored class.

    Please see https://www.nuscenes.org/lidar-segmentation for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 results_folder: str,
                 ignore_idx: int = 0):
        """
        Initializr a LidarSegEval object.
        nusc: A NuScenes object.
        results_folder: Path to the folder.
        ignore_idx: Index of the class to be ignored in the evaluation.
        """
        # Check there are ground truth annotations.
        assert len(nusc.lidarseg) > 0, 'Error: No ground truth annotations found in {}.'.format(nusc.version)

        # Check results folder exists.
        self.results_folder = results_folder
        assert os.path.exists(results_folder), 'Error: The result folder ({}) does not exist.'.format(results_folder)

        self.nusc = nusc
        self.results_folder = results_folder

        self.ignore_idx = ignore_idx

        self.global_cm = None

        self.adaptor = LidarsegChallengeAdaptor(self.nusc)

        self.num_classes = len(self.adaptor.merged_name_2_merged_idx_mapping)
        print('There are {} classes.'.format(self.num_classes))

    def evaluate(self, verbose: bool = False) -> Dict:
        """
        Performs the actual evaluation.
        :param verbose: Whether to print the evaluation.
        :return: A dictionary containing the IOU of the individual classes and the mIOU.
        """
        for sample in tqdm(self.nusc.sample):
            # Get the sample data token of the point cloud.
            sd_token = sample['data']['LIDAR_TOP']

            # Load the ground truth labels for the point cloud.
            lidarseg_label_filename = os.path.join(self.nusc.dataroot,
                                                   self.nusc.get('lidarseg', sd_token)['filename'])
            lidarseg_label = self.load_bin_file(lidarseg_label_filename)

            lidarseg_label = self.adaptor.convert_label(lidarseg_label)

            # Load the predictions for the point cloud.
            lidarseg_pred_filename = os.path.join(self.results_folder, 'lidarseg',
                                                  self.nusc.version, sd_token + '_lidarseg.bin')
            lidarseg_pred = self.load_bin_file(lidarseg_pred_filename)

            # TODO remove!!
            lidarseg_pred = self.adaptor.convert_label(lidarseg_pred)

            # Get the confusion matrix between the ground truth and predictions.
            sd_cm = self._get_confusion_matrix(lidarseg_label, lidarseg_pred)

            # Update the confusion matrix for the sample data into the confusion matrix for the split.
            self._update_confusion_matrix(sd_cm)

        iou_per_class = self._get_per_class_iou(self.global_cm)
        miou = np.nanmean(iou_per_class)

        # Put everything nicely into a dict.
        id2name = {idx: name for name, idx in self.adaptor.merged_name_2_merged_idx_mapping.items()}
        results = dict()
        for i, class_iou in enumerate(iou_per_class):
            if not np.isnan(class_iou):
                results['iou_' + id2name[i]] = class_iou
        results['miou'] = miou

        # Print the results if desired.
        if verbose:
            print(json.dumps(results, indent=4, sort_keys=False))

        return results

    @staticmethod
    def load_bin_file(bin_path: str) -> np.ndarray:
        """
        Loads a .bin file containing the labels.
        :param bin_path: Path to the .bin file.
        :return: An array containing the labels.
        """
        assert os.path.exists(bin_path), 'Error: Unable to find {}.'.format(bin_path)
        bin_content = np.fromfile(bin_path, dtype=np.uint8)
        assert len(bin_content) > 0, 'Error: {} is empty.'.format(bin_path)

        return bin_content

    def _update_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Updates the global confusion matrix.
        :param cm: Confusion matrix to be updated into the global confusion matrix.
        """
        if self.global_cm is None:
            self.global_cm = cm
        else:
            self.global_cm += cm

    def _get_confusion_matrix(self, gt_array: np.ndarray, pred_array: np.ndarray) -> np.ndarray:
        """
        Obtains the confusion matrix for the segmentation of a single point cloud.
        :param gt_array: An array containing the ground truth lables.
        :param pred_array: An array containing the predicted labels.
        :return: N x N array where N is the number of classes.
        """
        assert all((gt_array >= 0) & (gt_array < self.num_classes)), \
            "Error: Array for ground truth must be between 0 and {}".format(self.num_classes - 1)
        assert all((pred_array >= 0) & (pred_array < self.num_classes)), \
            "Error: Array for predictions must be between 0 and {}".format(self.num_classes - 1)

        label = self.num_classes * gt_array.astype('int') + pred_array
        count = np.bincount(label, minlength=self.num_classes ** 2)

        # Make confusion matrix (rows = gt, cols = preds).
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        return confusion_matrix

    def _get_per_class_iou(self, confusion_matrix: np.ndarray) -> List[float]:
        """
        Gets the IOU of each class in a confusion matrix. The index of the class to be ignored is set to NaN.
        :param confusion_matrix:
        :return: An array in which the IOU of a particular class sits at the array index corresponding to the
                 class index (the index of the class ot be ignored is set to NaN).
        """
        conf = confusion_matrix.copy()

        # Get the intersection for each class.
        intersection = np.diagonal(conf)

        # Get the union for each class.
        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Get the IOU for each class.
        iou_per_class = intersection / (
                    union.astype(np.float32) + 1e-15)  # Add a small value to guard against division by zero.

        # Set the IoU for the ignored class to NaN.
        if self.ignore_idx is not None:
            iou_per_class[self.ignore_idx] = np.nan

        return iou_per_class


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes lidar segmentation results.')
    parser.add_argument('--result_path', type=str,
                        help='The path to the results folder.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = args.result_path
    dataroot_ = args.dataroot
    version_ = args.version
    verbose_ = args.verbose

    nusc_ = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)

    evaluator = LidarSegEval(nusc_, result_path_)
    evaluator.evaluate(verbose=verbose_)