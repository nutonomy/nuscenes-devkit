# nuScenes dev-kit.
# Code written by Holger Caesar, 2021.

import os
import unittest

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud


class TestDataClasses(unittest.TestCase):

    def test_load_pointclouds(self):
        """
        Loads up lidar and radar pointclouds.
        """
        assert 'NUSCENES' in os.environ, 'Set NUSCENES env. variable to enable tests.'
        dataroot = os.environ['NUSCENES']
        nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
        sample_rec = nusc.sample[0]
        lidar_name = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])['filename']
        radar_name = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT'])['filename']
        lidar_path = os.path.join(dataroot, lidar_name)
        radar_path = os.path.join(dataroot, radar_name)
        pc1 = LidarPointCloud.from_file(lidar_path)
        pc2 = RadarPointCloud.from_file(radar_path)
        pc3, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=2)
        pc4, _ = RadarPointCloud.from_file_multisweep(nusc, sample_rec, 'RADAR_FRONT', 'RADAR_FRONT', nsweeps=2)

        # Check for valid dimensions.
        assert pc1.points.shape[0] == pc3.points.shape[0] == 4, 'Error: Invalid dimension for lidar pointcloud!'
        assert pc2.points.shape[0] == pc4.points.shape[0] == 18, 'Error: Invalid dimension for radar pointcloud!'
        assert pc1.points.dtype == pc3.points.dtype, 'Error: Invalid dtype for lidar pointcloud!'
        assert pc2.points.dtype == pc4.points.dtype, 'Error: Invalid dtype for radar pointcloud!'


if __name__ == '__main__':
    unittest.main()
