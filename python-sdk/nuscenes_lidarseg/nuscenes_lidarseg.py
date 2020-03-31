# nuScenes_lidarseg dev-kit.
# Code written by Oscar Beijbom, 2018.

import os.path as osp
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes_lidarseg dev-kit only supports Python version 3.")


class NuScenesLidarseg(NuScenes):
    """
    Database class for nuScenes-lidarseg to help query and retrieve information from the database.
    """

    def __init__(self, version: str = 'v1.0-mini', dataroot: str = '/data/sets/nuscenes-lidarseg',
                 dataroot_nuscenes: str = '/data/sets/nuscenes', verbose: bool = True):

        self.version = version
        self.dataroot = dataroot
        self.dataroot_nuscenes = dataroot_nuscenes
        self.table_names = ['categories', 'lidarseg']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes-lidarseg tables for version {}...".format(self.version))

        self.lidarseg = self.__load_table__('lidarseg')
        self.categories = self.__load_table__('lidarseg_category')
        self.colormap = [category['color'] for category in self.categories]

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))

        # Initialize NuScenesLidarsegExplorer class
        self.explorer = NuScenesLidarsegExplorer(self.version, self.dataroot, self.dataroot_nuscenes)

    def list_lidarseg_categories(self) -> None:
        """ Print categories. """
        print('There are %d categories in %s' %(len(self.categories), self.version))

        for category in self.categories:
            print (category['index'], '\t', category['label'])

    def render_sample_lidarseg_data(self, sample_data_token: str,
                                    axes_limit: float = 40, ax: Axes = None,
                                    out_path: str = None,
                                    use_flat_vehicle_coordinates: bool = True,
                                    ) -> None:
        self.explorer.render_sample_lidarseg_data(sample_data_token, colormap=self.colormap,
                                                  axes_limit=axes_limit, ax=ax,
                                                  out_path=out_path,
                                                  use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                                  )


class NuScenesLidarsegExplorer:
    """ Helper class to list and visualize NuScenes-lidarseg data. These are meant to serve as tutorials and templates
    for working with the data. """

    def __init__(self, version, dataroot, dataroot_nuscenes, verbose: bool = True, map_resolution: float = 0.1):
        self.nusc = NuScenes(version, dataroot_nuscenes, verbose, map_resolution)
        self.dataroot = dataroot

    def render_sample_lidarseg_data(self,
                                    sample_data_token: str,
                                    colormap: List[List[int]],
                                    axes_limit: float = 40,
                                    ax: Axes = None,
                                    out_path: str = None,
                                    use_flat_vehicle_coordinates: bool = True) -> None:
        """
        Render sample LIDAR segmentation data onto axis.
        :param sample_data_token: Sample_data token.
        :param colormap: Colors to plot the point cloud labels in. It should be an array of RGB values,
            like [[R1, G1, B1], [R2, G2, B2], ..., [Rn, Gn, Bn]]
        :param axes_limit: Axes limit for LIDAR (measured in meters).
        :param ax: Axes onto which to render.
        :param out_path: Optional path to save the rendered figure to disk.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        """

        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        assert sensor_modality == 'lidar', 'Please ensure that sensor data is LIDAR.'

        sample_rec = self.nusc.get('sample', sd_record['sample_token'])
        ref_chan = 'LIDAR_TOP'
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

        pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)

        # By default we render the sample_data top down in the sensor frame.
        # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
        # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
        if use_flat_vehicle_coordinates:
            # Retrieve transformation matrices for reference point cloud.
            cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
        else:
            viewpoint = np.eye(4)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        # Show point cloud.
        points = view_points(pc.points[:3, :], viewpoint, normalize=False)

        print('Loading lidarseg labels...')
        lidarseg_labels_filename = osp.join(self.dataroot, 'lidarseg', sample_data_token + '_lidarseg.bin')
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)
        assert points.shape[1] == len(points_label)
        colors = points_label

        # ---------- coloring ---------- #
        # colors in colormap should be an RGB or RGBA (red, green, blue, alpha) tuple of float values in
        # closed interval [0, 1]
        colormap = list(map(lambda x: (int(x[0] / 255), int(x[1] / 255), int(x[2] / 255)), colormap))
        colormap = np.array(colormap)
        # ---------- /coloring ---------- #

        point_scale = 0.2
        scatter = ax.scatter(points[0, :], points[1, :], c=colormap[colors], s=point_scale)

        # Show ego vehicle.
        ax.plot(0, 0, 'x', color='red')

        # Limit visible range.
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        ax.axis('off')
        ax.set_title(sd_record['channel'])
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path)

        plt.show()
