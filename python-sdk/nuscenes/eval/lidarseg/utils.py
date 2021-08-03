from typing import Dict, List, Tuple

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


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
            "Error: Array for ground truth must be between 0 and {} (inclusive).".format(self.num_classes - 1)
        assert all((pred_array > 0) & (pred_array < self.num_classes)), \
            "Error: Array for predictions must be between 1 and {} (inclusive).".format(self.num_classes - 1)

        label = self.num_classes * gt_array.astype('int') + pred_array
        count = np.bincount(label, minlength=self.num_classes ** 2)

        # Make confusion matrix (rows = gt, cols = preds).
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)

        # For the class to be ignored, set both the row and column to 0 (adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        if self.ignore_idx is not None:
            confusion_matrix[self.ignore_idx, :] = 0
            confusion_matrix[:, self.ignore_idx] = 0

        return confusion_matrix

    def get_per_class_iou(self) -> List[float]:
        """
        Gets the IOU of each class in a confusion matrix.
        :return: An array in which the IOU of a particular class sits at the array index corresponding to the
                 class index.
        """
        conf = self.global_cm.copy()

        # Get the intersection for each class.
        intersection = np.diagonal(conf)

        # Get the union for each class.
        ground_truth_set = conf.sum(axis=1)
        predicted_set = conf.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        # Get the IOU for each class.
        # In case we get a division by 0, ignore / hide the error(adapted from
        # https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py).
        with np.errstate(divide='ignore', invalid='ignore'):
            iou_per_class = intersection / (union.astype(np.float32))

        return iou_per_class

    def get_mean_iou(self) -> float:
        """
        Gets the mean IOU (mIOU) over the classes.
        :return: mIOU over the classes.
        """
        iou_per_class = self.get_per_class_iou()
        miou = float(np.nanmean(iou_per_class))
        return miou

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

        # Get the IOU per class.
        iou_per_class = self.get_per_class_iou()

        # Weight the IOU by frequency and sum across the classes.
        freqweighted_iou = float(np.nansum(num_points_per_class * iou_per_class) / num_points_total)

        return freqweighted_iou


class LidarsegClassMapper:
    """
    Maps the (fine) classes in nuScenes-lidarseg to the (coarse) classes for the nuScenes-lidarseg challenge.

    Example usage::
        nusc_ = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
        mapper_ = LidarsegClassMapper(nusc_)
    """
    def __init__(self, nusc: NuScenes):
        """
        Initialize a LidarsegClassMapper object.
        :param nusc: A NuScenes object.
        """
        self.nusc = nusc

        self.ignore_class = self.get_ignore_class()

        self.fine_name_2_coarse_name_mapping = self.get_fine2coarse()
        self.coarse_name_2_coarse_idx_mapping = self.get_coarse2idx()
        self.coarse_colormap = self.get_coarse2color()

        self.check_mapping()

        self.fine_idx_2_coarse_idx_mapping = self.get_fine_idx_2_coarse_idx()

    @staticmethod
    def get_ignore_class() -> Dict[str, int]:
        """
        Defines the name and index of the ignore class.
        :return: A dictionary containing the name and index of the ignore class.
        """
        return {'name': 'ignore', 'index': 0}

    def get_fine2coarse(self) -> Dict:
        """
        Returns the mapping from the fine classes to the coarse classes.
        :return: A dictionary containing the mapping from the fine classes to the coarse classes.
        """
        return {'noise': self.ignore_class['name'],
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.wheelchair': self.ignore_class['name'],
                'human.pedestrian.stroller': self.ignore_class['name'],
                'human.pedestrian.personal_mobility': self.ignore_class['name'],
                'human.pedestrian.police_officer': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'animal': self.ignore_class['name'],
                'vehicle.car': 'car',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.truck': 'truck',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.emergency.ambulance': self.ignore_class['name'],
                'vehicle.emergency.police': self.ignore_class['name'],
                'vehicle.trailer': 'trailer',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic_cone',
                'movable_object.pushable_pullable': self.ignore_class['name'],
                'movable_object.debris': self.ignore_class['name'],
                'static_object.bicycle_rack': self.ignore_class['name'],
                'flat.driveable_surface': 'driveable_surface',
                'flat.sidewalk': 'sidewalk',
                'flat.terrain': 'terrain',
                'flat.other': 'other_flat',
                'static.manmade': 'manmade',
                'static.vegetation': 'vegetation',
                'static.other': self.ignore_class['name'],
                'vehicle.ego': self.ignore_class['name']}

    def get_coarse2idx(self) -> Dict[str, int]:
        """
        Returns the mapping from the coarse class names to the coarse class indices.
        :return: A dictionary containing the mapping from the coarse class names to the coarse class indices.
        """
        return {self.ignore_class['name']: self.ignore_class['index'],
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

    def get_coarse2color(self) -> Dict[str, Tuple[int]]:
        """
        Returns the mapping from the coarse class names to the colors in RGB.
        :return: A dictionary containing the mapping from the coarse class names to the colors in RGB.
        """
        return {self.ignore_class['name']: (0, 0, 0),  # Black.
                'barrier': (112, 128, 144),  # Slategrey
                'bicycle': (220, 20, 60),  # Crimson
                'bus': (255, 127, 80),  # Coral
                'car': (255, 158, 0),  # Orange
                'construction_vehicle': (233, 150, 70),  # Darksalmon
                'motorcycle': (255, 61, 99),  # Red
                'pedestrian': (0, 0, 230),  # Blue
                'traffic_cone': (47, 79, 79),  # Darkslategrey
                'trailer': (255, 140, 0),  # Darkorange
                'truck': (255, 99, 71),  # Tomato
                'driveable_surface': (0, 207, 191),  # nuTonomy green
                'other_flat': (175, 0, 75),
                'sidewalk': (75, 0, 75),
                'terrain': (112, 180, 60),
                'manmade': (222, 184, 135),  # Burlywood
                'vegetation': (0, 175, 0)}  # Green

    def get_fine_idx_2_coarse_idx(self) -> Dict[int, int]:
        """
        Returns the mapping from the the indices of the coarse classes to that of the coarse classes.
        :return: A dictionary containing the mapping from the the indices of the coarse classes to that of the
                 coarse classes.
        """
        fine_idx_2_coarse_idx_mapping = dict()
        for fine_name, fine_idx in self.nusc.lidarseg_name2idx_mapping.items():
            fine_idx_2_coarse_idx_mapping[fine_idx] = self.coarse_name_2_coarse_idx_mapping[
                self.fine_name_2_coarse_name_mapping[fine_name]]
        return fine_idx_2_coarse_idx_mapping

    def check_mapping(self) -> None:
        """
        Convenient method to check that the mappings for fine2coarse and coarse2idx are synced.
        """
        coarse_set = set()
        for fine_name, coarse_name in self.fine_name_2_coarse_name_mapping.items():
            coarse_set.add(coarse_name)

        assert coarse_set == set(self.coarse_name_2_coarse_idx_mapping.keys()), \
            'Error: Number of coarse classes is not the same as the number of coarse indices.'

    def convert_label(self, points_label: np.ndarray) -> np.ndarray:
        """
        Convert the labels in a single .bin file according to the provided mapping.
        :param points_label: The .bin to be converted (e.g. './i_contain_the_labels_for_a_pointcloud.bin')
        """
        counter_before = self.get_stats(points_label)  # get stats before conversion

        # Map the labels accordingly; if there are labels present in points_label but not in the map,
        # an error will be thrown
        points_label = np.vectorize(self.fine_idx_2_coarse_idx_mapping.__getitem__)(points_label)

        counter_after = self.get_stats(points_label)  # Get stats after conversion.

        assert self.compare_stats(counter_before, counter_after), 'Error: Statistics of labels have changed ' \
                                                                  'after conversion. Pls check.'

        return points_label

    def compare_stats(self, counter_before: List[int], counter_after: List[int]) -> bool:
        """
        Compare stats for a single .bin file before and after conversion.
        :param counter_before: A numpy array which contains the counts of each class (the index of the array corresponds
                               to the class label), before conversion; e.g. np.array([0, 1, 34, ...]) --> class 0 has
                               no points, class 1 has 1 point, class 2 has 34 points, etc.
        :param counter_after: A numpy array which contains the counts of each class (the index of the array corresponds
                              to the class label) after conversion
        :return: True or False; True if the stats before and after conversion are the same, and False if otherwise.
        """
        counter_check = [0] * len(counter_after)
        for i, count in enumerate(counter_before):  # Note that the class labels are 0-indexed.
            counter_check[self.fine_idx_2_coarse_idx_mapping[i]] += count

        comparison = counter_check == counter_after

        return comparison

    def get_stats(self, points_label: np.array) -> List[int]:
        """
        Get frequency of each label in a point cloud.
        :param points_label: A numpy array which contains the labels of the point cloud;
                             e.g. np.array([2, 1, 34, ..., 38])
        :return: An array which contains the counts of each label in the point cloud. The index of the point cloud
                  corresponds to the index of the class label. E.g. [0, 2345, 12, 451] means that there are no points
                  in class 0, there are 2345 points in class 1, there are 12 points in class 2 etc.
        """
        # Create "buckets" to store the counts for each label; the number of "buckets" is the larger of the number
        # of classes in nuScenes-lidarseg and lidarseg challenge.
        lidarseg_counts = [0] * (max(max(self.fine_idx_2_coarse_idx_mapping.keys()),
                                     max(self.fine_idx_2_coarse_idx_mapping.values())) + 1)

        indices: np.ndarray = np.bincount(points_label)
        ii = np.nonzero(indices)[0]

        for class_idx, class_count in zip(ii, indices[ii]):
            lidarseg_counts[class_idx] += class_count  # Increment the count for the particular class name.

        return lidarseg_counts


def get_samples_in_eval_set(nusc: NuScenes, eval_set: str) -> List[str]:
    """
    Gets all the sample tokens from the split that are relevant to the eval set.
    :param nusc: A NuScenes object.
    :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
    :return: A list of sample tokens.
    """
    # Create a dict to map from scene name to scene token for quick lookup later on.
    scene_name2tok = dict()
    for rec in nusc.scene:
        scene_name2tok[rec['name']] = rec['token']

    # Get scenes splits from nuScenes.
    scenes_splits = create_splits_scenes(verbose=False)

    # Collect sample tokens for each scene.
    samples = []
    for scene in scenes_splits[eval_set]:
        scene_record = nusc.get('scene', scene_name2tok[scene])
        total_num_samples = scene_record['nbr_samples']
        first_sample_token = scene_record['first_sample_token']
        last_sample_token = scene_record['last_sample_token']

        sample_token = first_sample_token
        i = 0
        while sample_token != '':
            sample_record = nusc.get('sample', sample_token)
            samples.append(sample_record['token'])

            if sample_token == last_sample_token:
                sample_token = ''
            else:
                sample_token = sample_record['next']
            i += 1

        assert total_num_samples == i, 'Error: There were supposed to be {} keyframes, ' \
                                       'but only {} keyframes were processed'.format(total_num_samples, i)

    return samples
