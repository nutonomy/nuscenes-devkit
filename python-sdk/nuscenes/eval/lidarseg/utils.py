from typing import Dict

import numpy as np

from nuscenes import NuScenes


class LidarsegChallengeAdaptor:
    """

    """
    def __init__(self, nusc: NuScenes):
        """

        """
        self.nusc = nusc

        self.raw_name_2_merged_name_mapping = self.get_raw2merged()
        self.merged_name_2_merged_idx_mapping = self.get_merged2idx()

        self.check_mapping()

        self.raw_idx_2_merged_idx_mapping = self.get_raw_idx_2_merged_idx()

    @staticmethod
    def get_raw2merged() -> Dict:
        """
        Returns the mapping from
        :return:
        """
        return {'noise': 'void_ignore',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.wheelchair': 'void_ignore',
                'human.pedestrian.stroller': 'void_ignore',
                'human.pedestrian.personal_mobility': 'void_ignore',
                'human.pedestrian.police_officer': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'animal': 'void_ignore',
                'vehicle.car': 'car',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.truck': 'truck',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.emergency.ambulance': 'void_ignore',
                'vehicle.emergency.police': 'void_ignore',
                'vehicle.trailer': 'trailer',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic_cone',
                'movable_object.pushable_pullable': 'void_ignore',
                'movable_object.debris': 'void_ignore',
                'static_object.bicycle_rack': 'void_ignore',
                'flat.driveable_surface': 'driveable_surface',
                'flat.sidewalk': 'sidewalk',
                'flat.terrain': 'terrain',
                'flat.other': 'other_flat',
                'static.manmade': 'manmade',
                'static.vegetation': 'vegetation',
                'static.other': 'void_ignore',
                'vehicle.ego': 'void_ignore'}

    @staticmethod
    def get_merged2idx() -> Dict:
        """
        Returns the mapping from the merged class names to the merged class indices.
        :return: A dictionary containing the mapping from the merged class names to the merged class indices.
        """
        return {'void_ignore': 0,
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

    def get_raw_idx_2_merged_idx(self) -> Dict:
        """

        """
        raw_idx_2_merged_idx_mapping = dict()
        for raw_name, raw_idx in self.nusc.lidarseg_name2idx_mapping.items():
            raw_idx_2_merged_idx_mapping[raw_idx] = self.merged_name_2_merged_idx_mapping[
                self.raw_name_2_merged_name_mapping[raw_name]]
        return raw_idx_2_merged_idx_mapping

    def check_mapping(self) -> None:
        """

        """
        merged_set = set()
        for raw_name, merged_name in self.raw_name_2_merged_name_mapping.items():
            merged_set.add(merged_name)

        assert len(merged_set) == len(self.merged_name_2_merged_idx_mapping), \
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

    def compare_stats(self, counter_before: np.array, counter_after: np.array) -> bool:
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

    def get_stats(self, points_label: np.array) -> np.ndarray:
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


if __name__ == '__main__':
    nusc_ = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
    mappings_ = LidarsegChallengeAdaptor(nusc_)
