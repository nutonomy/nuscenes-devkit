from typing import Dict, List

import numpy as np

from nuscenes import NuScenes


class ConfusionMatrix:
    """
    Class for confusion matrix with various convenient methods.
    """
    def __init__(self, num_classes: int, ignore_idx: int = None):
        """
        Initialize a ConfusionMatrix object.
        :param num_classes: Number of classes in the confusion matrix.
        :param ignore_idx: Index of the class to be ignored in the confusion matrix.
        """
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx

        self.global_cm = None

    def update(self, gt_array: np.ndarray, pred_array: np.ndarray) -> None:
        """
        Updates the global confusion matrix.
        :param gt_array: An array containing the ground truth labels.
        :param pred_array: An array containing the predicted labels.
        """
        cm = self._get_confusion_matrix(gt_array, pred_array)

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

    def get_per_class_iou(self) -> List[float]:
        """
        Gets the IOU of each class in a confusion matrix. The index of the class to be ignored is set to NaN.
        :return: An array in which the IOU of a particular class sits at the array index corresponding to the
                 class index (the index of the class ot be ignored is set to NaN).
        """
        conf = self.global_cm.copy()

        # Get the intersection for each class.
        intersection = np.diagonal(conf)

        # Get the union for each class.
        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Get the IOU for each class.
        iou_per_class = intersection / (
                    union.astype(np.float32) + 1e-15)  # Add a small value to guard against division by zero.

        # Set the IOU for the ignored class to NaN.
        if self.ignore_idx is not None:
            iou_per_class[self.ignore_idx] = np.nan

        # If there are no points belonging to a certain class, then set the the IOU of that class to NaN too.
        idxs_no_ground_truth = np.where(iou_per_class == 0)
        if len(idxs_no_ground_truth) > 0:
            iou_per_class[idxs_no_ground_truth] = np.nan

        return iou_per_class

    def get_freqweighted_iou(self) -> float:
        """
        Gets the frequency-weighted IOU over the classes.
        :return: Frequency-weighted IOU over the classes.
        """
        conf = self.global_cm.copy()

        # Get the number of points per class (based on ground truth).
        num_points_per_class = conf.sum(axis=1)

        # Get the total number of points in the eval set.
        num_points_total = conf.sum()

        # Get the frequency per class.
        freq = num_points_per_class / num_points_total

        # Get the IOU per class.
        iou_per_class = self.get_per_class_iou()

        # Weight the IOU by frequency and sum across the classes.
        freqweighted_iou = np.nansum((freq[freq != np.nan] * iou_per_class[freq != np.nan]))

        return freqweighted_iou


class LidarsegChallengeAdaptor:
    """
    An adaptor to map the (raw) classes in nuScenes-lidarseg to the (merged) classes for the nuScenes lidar
    segmentation challenge.

    Example usage::
        nusc_ = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
        adaptor_ = LidarsegChallengeAdaptor(nusc_)
    """
    def __init__(self, nusc: NuScenes):
        """
        Initialize a LidarsegChallengeAdaptor object.
        :param nusc: A NuScenes object.
        """
        self.nusc = nusc

        self.raw_name_2_merged_name_mapping = self.get_raw2merged()
        self.merged_name_2_merged_idx_mapping = self.get_merged2idx()

        self.check_mapping()

        self.raw_idx_2_merged_idx_mapping = self.get_raw_idx_2_merged_idx()

    @staticmethod
    def get_raw2merged() -> Dict:
        """
        Returns the mapping from the raw classes to the merged classes.
        :return: A dictionary containing the mapping from the raw classes to the merged classes.
        """
        return {'noise': 'ignore',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.wheelchair': 'ignore',
                'human.pedestrian.stroller': 'ignore',
                'human.pedestrian.personal_mobility': 'ignore',
                'human.pedestrian.police_officer': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'animal': 'ignore',
                'vehicle.car': 'car',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.truck': 'truck',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.emergency.ambulance': 'ignore',
                'vehicle.emergency.police': 'ignore',
                'vehicle.trailer': 'trailer',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic_cone',
                'movable_object.pushable_pullable': 'ignore',
                'movable_object.debris': 'ignore',
                'static_object.bicycle_rack': 'ignore',
                'flat.driveable_surface': 'driveable_surface',
                'flat.sidewalk': 'sidewalk',
                'flat.terrain': 'terrain',
                'flat.other': 'other_flat',
                'static.manmade': 'manmade',
                'static.vegetation': 'vegetation',
                'static.other': 'ignore',
                'vehicle.ego': 'ignore'}

    @staticmethod
    def get_merged2idx() -> Dict[str, int]:
        """
        Returns the mapping from the merged class names to the merged class indices.
        :return: A dictionary containing the mapping from the merged class names to the merged class indices.
        """
        return {'ignore': 0,
                'barrier': 1,
                'bicycle': 2,
                'bus': 3,
                'car': 4,
                'construction_vehicle': 5,
                'motorcycle': 6,
                'pedestrian': 7,
                'traffic_cone': 8,
                'trailer': 9,
                'truck': 10,
                'driveable_surface': 11,
                'other_flat': 12,
                'sidewalk': 13,
                'terrain': 14,
                'manmade': 15,
                'vegetation': 16}

    def get_raw_idx_2_merged_idx(self) -> Dict[int, int]:
        """
        Returns the mapping from the the indices of the merged classes to that of the merged classes.
        :return: A dictionary containing the mapping from the the indices of the merged classes to that of the
                 merged classes.
        """
        raw_idx_2_merged_idx_mapping = dict()
        for raw_name, raw_idx in self.nusc.lidarseg_name2idx_mapping.items():
            raw_idx_2_merged_idx_mapping[raw_idx] = self.merged_name_2_merged_idx_mapping[
                self.raw_name_2_merged_name_mapping[raw_name]]
        return raw_idx_2_merged_idx_mapping

    def check_mapping(self) -> None:
        """
        Convenient method to check that the mappings for raw2merged and merged2idx are synced.
        """
        merged_set = set()
        for raw_name, merged_name in self.raw_name_2_merged_name_mapping.items():
            merged_set.add(merged_name)

        assert merged_set == set(self.merged_name_2_merged_idx_mapping.keys()), \
            'Error: Number of merged classes is not the same as the number of merged indices.'

    def convert_label(self, points_label: np.ndarray) -> np.ndarray:
        """
        Convert the labels in a single .bin file according to the provided mapping.
        :param points_label: The .bin to be converted (e.g. './i_contain_the_labels_for_a_pointcloud.bin')
        """
        counter_before = self.get_stats(points_label)  # get stats before conversion

        # Map the labels accordingly; if there are labels present in points_label but not in the map,
        # an error will be thrown
        points_label = np.vectorize(self.raw_idx_2_merged_idx_mapping.__getitem__)(points_label)

        counter_after = self.get_stats(points_label)  # get stats after conversion

        assert self.compare_stats(counter_before, counter_after), 'Error: Statistics of labels have changed ' \
                                                                  'after conversion. Pls check.'

        return points_label

    def compare_stats(self, counter_before: List[int], counter_after: List[int]) -> bool:
        """
        Compare stats for a single .bin file before and after conversion.
        :param counter_before: A numPy array which contains the counts of each class (the index of the array corresponds
                               to the class label), before conversion; e.g. np.array([0, 1, 34, ...]) --> class 0 has
                               no points, class 1 has 1 point, class 2 has 34 points, etc.
        :param counter_after: A numPy array which contains the counts of each class (the index of the array corresponds
                              to the class label) after conversion
        :return: True or False; True if the stats before and after conversion are the same, and False if otherwise.
        """
        counter_check = [0] * len(counter_after)
        for i, count in enumerate(counter_before):  # Note that it is expected that the class labels are 0-indexed.
            counter_check[self.raw_idx_2_merged_idx_mapping[i]] += count

        comparison = counter_check == counter_after

        return comparison

    def get_stats(self, points_label: np.array) -> List[int]:
        """
        Get frequency of each label in a point cloud.
        :param points_label: A numPy array which contains the labels of the point cloud;
                             e.g. np.array([2, 1, 34, ..., 38])
        :return: An array which contains the counts of each label in the point cloud. The index of the point cloud
                  corresponds to the index of the class label. E.g. [0, 2345, 12, 451] means that there are no points
                  in class 0, there are 2345 points in class 1, there are 12 points in class 2 etc.
        """
        # Create "buckets" to store the counts for each label; the number of "buckets" is the larger of the number
        # of classes in nuScenes-lidarseg and lidarseg challenge.
        lidarseg_counts = [0] * (max(max(self.raw_idx_2_merged_idx_mapping.keys()),
                                     max(self.raw_idx_2_merged_idx_mapping.values())) + 1)

        indices: np.ndarray = np.bincount(points_label)
        ii = np.nonzero(indices)[0]

        for class_idx, class_count in zip(ii, indices[ii]):
            lidarseg_counts[class_idx] += class_count  # Increment the count for the particular class name.

        return lidarseg_counts
