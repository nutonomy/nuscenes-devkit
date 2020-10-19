import os

from tqdm import tqdm
import numpy as np

from nuscenes import NuScenes

from utils import LidarsegChallengeAdaptor


class LidarSegEval:
    def __init__(self,
                 nusc: NuScenes,
                 results_folder: str,
                 ignore_idx: int = 0):
        """
        """

        """
        # Check there are ground truth annotations.
        self.gt_samples = nusc.sample # check if test set has samples?? Might have to check data itself.
        assert len(self.samples) > 0, 'Error: There are no ground truth ______'

        # Check results folder exists.
        self.results_folder = results_folder
        assert os.path.exists(results_folder), 'Error: The result folder ({}) does not exist!'.format(results_folder)
        """

        self.nusc = nusc
        self.results_folder = results_folder

        self.ignore_idx = ignore_idx

        self.global_cm = None

        self.adaptor = LidarsegChallengeAdaptor(self.nusc)

        self.num_classes = len(self.adaptor.merged_name_2_merged_idx_mapping)
        print('There are {} classes.'.format(self.num_classes))

    def evaluate(self) -> None:
        for sample in tqdm(self.nusc.sample):
            sd_token = sample['data']['LIDAR_TOP']

            # Load ground truth labels for sample data.
            lidarseg_label_filename = os.path.join(self.nusc.dataroot,
                                                   self.nusc.get('lidarseg', sd_token)['filename'])
            lidarseg_label = self.load_bin_file(lidarseg_label_filename)

            # print(lidarseg_label)
            lidarseg_label = self.adaptor.convert_label(lidarseg_label)
            # print(lidarseg_label)

            # Load predictions for sample data.
            lidarseg_pred_filename = os.path.join(self.results_folder, 'lidarseg',
                                                  self.nusc.version, sd_token + '_lidarseg.bin')
            lidarseg_pred = self.load_bin_file(lidarseg_pred_filename)

            # TODO remove!!
            lidarseg_pred = self.adaptor.convert_label(lidarseg_pred)

            # Get the confusion matrix between the ground truth and predictions.
            sd_cm = self._get_confusion_matrix(lidarseg_label, lidarseg_pred)

            # Update the confusion matrix for the sample data into the confusion matrix for the split.
            self._update_confusion_matrix(sd_cm)

        """
        # Can do some printing of each class IOU with the mapping
        id2name = {label_id: label.name for label_id, label in res.labelmap.items() if label.name != 'ignore_label'}
        """

        iou_per_class = self._get_per_class_iou(self.global_cm)
        miou = np.nanmean(iou_per_class)

        print(iou_per_class, miou)
        # https://evalai.readthedocs.io/en/latest/evaluation_scripts.html

    @staticmethod
    def load_bin_file(bin_path: str) -> np.ndarray:
        assert os.path.exists(bin_path), 'Error: Unable to find {}.'.format(bin_path)
        bin_content = np.fromfile(bin_path, dtype=np.uint8)
        assert len(bin_content) > 0, 'Error: {} is empty.'.format(bin_path)

        return bin_content

    def _update_confusion_matrix(self, cm) -> None:
        """
        Updates segmentation confusion matrix.
        :param cm: Confusion matrix to be updated.
        :param res: Lidar segmentation results.
        """
        if self.global_cm is None:
            self.global_cm = cm
        else:
            self.global_cm += cm

        # TODO test that final global_cm matches counts from nusc.list_lidarseg_categories(sort_by='count')
        # TODO test that sample cm matches counts from nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')

    def _get_confusion_matrix(self, gt_array, pred_array):
        assert all((gt_array >= 0) & (
                    gt_array < self.num_classes)), "Error: Array for ground truth must be between 0 and {}".format(
            self.num_classes - 1)
        assert all((pred_array >= 0) & (
                    pred_array < self.num_classes)), "Error: Array for predictions must be between 0 and {}".format(
            self.num_classes - 1)

        """
        mask = (gt_array >= 0) & (gt_array < num_classes)
        label = num_classes * gt_array[mask].astype('int') + pred_array[mask]
        """
        label = self.num_classes * gt_array.astype('int') + pred_array
        count = np.bincount(label, minlength=self.num_classes ** 2)

        # Make confusion matrix (rows = gt, cols = preds).
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        return confusion_matrix

    def _get_per_class_iou(self, confusion_matrix):
        conf = confusion_matrix.copy()

        """
        # For the index to be ignored, remove counts for tp, fp and fn from the confusion matrix.
        conf[ignore_idx] = 0  # Remove tp and fn (row)
        conf[:, ignore_idx] = 0  # Remove fp (col)

        print(conf)
        """
        """
        np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        """

        """
        # Get tp, fp, fn
        tp = np.diagonal(conf)
        fp = np.sum(conf, axis=0) - tp
        fn = np.sum(conf, axis=1) - tp
        iou_check = tp / (tp + fp + fn + 1e-15)
        print(iou_check, '@@@##')
        """

        intersection = np.diagonal(conf)

        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        iou_per_class = intersection / (
                    union.astype(np.float32) + 1e-15)  # Add a small value to guard against division by zero

        # Set the IoU for the ignored class to NaN.
        if self.ignore_idx is not None:
            iou_per_class[self.ignore_idx] = np.nan

        return iou_per_class
