import argparse
import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.lidarseg.utils import LidarsegChallengeAdaptor
from nuscenes.utils.splits import create_splits_scenes


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
                 eval_set: str,
                 ignore_idx: int = 0,
                 verbose: bool = False):
        """
        Initialize a LidarSegEval object.
        :param nusc: A NuScenes object.
        :param results_folder: Path to the folder.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param ignore_idx: Index of the class to be ignored in the evaluation.
        :param verbose: Whether to print messages during the evaluation.
        """
        # Check there are ground truth annotations.
        assert len(nusc.lidarseg) > 0, 'Error: No ground truth annotations found in {}.'.format(nusc.version)

        # Check results folder exists.
        self.results_folder = results_folder
        assert os.path.exists(results_folder), 'Error: The result folder ({}) does not exist.'.format(results_folder)

        self.nusc = nusc
        self.results_folder = results_folder
        self.eval_set = eval_set
        self.ignore_idx = ignore_idx
        self.verbose = verbose

        self.global_cm = None

        self.adaptor = LidarsegChallengeAdaptor(self.nusc)

        self.num_classes = len(self.adaptor.merged_name_2_merged_idx_mapping)
        if self.verbose:
            print('There are {} classes.'.format(self.num_classes))

        self.sample_tokens = self.get_samples_in_eval_set()
        if self.verbose:
            print('There are {} samples.'.format(len(self.sample_tokens)))

    def evaluate(self) -> Dict:
        """
        Performs the actual evaluation.
        :return: A dictionary containing the IOU of the individual classes and the mIOU.
        """
        for sample_token in tqdm(self.sample_tokens):
            sample = self.nusc.get('sample', sample_token)

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
        if self.verbose:
            print("======\nnuScenes lidar segmentation evaluation for version {}".format(self.nusc.version))
            for iou_name, iou in results.items():
                print('{:30}  {:.5f}'.format(iou_name, iou))
            # print(json.dumps(results, indent=4, sort_keys=False))
            print("======")

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
        :param gt_array: An array containing the ground truth labels.
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
        :param confusion_matrix: The confusion matrix to calculate IOU for each class on.
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

    def get_samples_in_eval_set(self) -> List[str]:
        """

        """
        # Create a dict to map from scene name to scene token for quick lookup later on.
        scene_name2tok = dict()
        for rec in self.nusc.scene:
            scene_name2tok[rec['name']] = rec['token']

        # Get scenes splits from nuScenes.
        scenes_splits = create_splits_scenes(verbose=False)

        # Collect sample tokens for each scene.
        samples = []
        for scene in scenes_splits[self.eval_set]:
            scene_record = self.nusc.get('scene', scene_name2tok[scene])
            total_num_samples = scene_record['nbr_samples']
            first_sample_token = scene_record['first_sample_token']
            last_sample_token = scene_record['last_sample_token']

            sample_token = first_sample_token
            i = 0
            while sample_token != '':
                sample_record = self.nusc.get('sample', sample_token)
                samples.append(sample_record['token'])

                if sample_token == last_sample_token:
                    sample_token = ''
                else:
                    sample_token = sample_record['next']
                i += 1

            assert total_num_samples == i, 'Error: There were supposed to be {} keyframes, ' \
                                           'but only {} keyframes were processed'.format(total_num_samples, i)

        return samples


if __name__ == '__main__':
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes lidar segmentation results.')
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
