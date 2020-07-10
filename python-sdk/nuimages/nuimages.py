# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

import json
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Callable

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import cv2

from nuimages.utils.utils import annotation_name, mask_decode
from nuimages.utils.lidar import depth_map, distort_pointcloud, InvertedNormalize
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import LidarPointCloud

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")


class NuImages:
    """
    Database class for nuImages to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-train',
                 dataroot: str = '/data/sets/nuimages',
                 lazy: bool = True,
                 verbose: bool = False):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0-train", "v1.0-val").
        :param dataroot: Path to the tables and data.
        :param lazy: Whether to use lazy loading for the database tables.
        :param verbose: Whether to print status messages during load.
        """
        self.version = version
        self.dataroot = dataroot
        self.lazy = lazy
        self.verbose = verbose

        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 'log', 'object_ann', 'sample',
                            'sample_data', 'sensor', 'surface_ann']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading nuImages tables for version {}...".format(self.version))

        # Init reverse indexing.
        self._token2ind: Dict[str, Optional[dict]] = dict()
        for table in self.table_names:
            self._token2ind[table] = None

        # Load tables directly if requested.
        if not self.lazy:
            # Explicitly init tables to help the IDE determine valid class members.
            self.attribute = self.__load_table__('attribute')
            self.calibrated_sensor = self.__load_table__('calibrated_sensor')
            self.category = self.__load_table__('category')
            self.ego_pose = self.__load_table__('ego_pose')
            self.log = self.__load_table__('log')
            self.object_ann = self.__load_table__('object_ann')
            self.sample = self.__load_table__('sample')
            self.sample_data = self.__load_table__('sample_data')
            self.sensor = self.__load_table__('sensor')
            self.surface_ann = self.__load_table__('surface_ann')

        self.color_map = get_colormap()

        if verbose:
            print("Done loading in {:.1f} seconds (lazy={}).\n======".format(time.time() - start_time, self.lazy))

    # ### Internal methods. ###

    def __getattr__(self, attr_name: str) -> Any:
        """
        Implement lazy loading for the database tables. Otherwise throw the default error.
        :param attr_name: The name of the variable to look for.
        :return: The dictionary that represents that table.
        """
        if attr_name in self.table_names:
            return self._load_lazy(attr_name, lambda tab_name: self.__load_table__(tab_name))
        else:
            raise AttributeError("Error: %r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        # Lazy loading: Compute reverse indices.
        if self._token2ind[table_name] is None:
            self._token2ind[table_name] = dict()
            for ind, member in enumerate(getattr(self, table_name)):
                self._token2ind[table_name][member['token']] = ind

        return self._token2ind[table_name][token]

    @property
    def table_root(self) -> str:
        """
        Returns the folder where the tables are stored for the relevant version.
        """
        return osp.join(self.dataroot, self.version)

    def load_tables(self, table_names: List[str]) -> None:
        """
        Load tables and add them to self, if not already loaded.
        :param table_names: The names of the nuImages tables to be loaded.
        """
        for table_name in table_names:
            self._load_lazy(table_name, lambda tab_name: self.__load_table__(tab_name))

    def _load_lazy(self, attr_name: str, loading_func: Callable) -> Any:
        """
        Load an attribute and add it to self, if it isn't already loaded.
        :param attr_name: The name of the attribute to be loaded.
        :param loading_func: The function used to load it if necessary.
        :return: The loaded attribute.
        """
        if attr_name in self.__dict__.keys():
            return self.__getattribute__(attr_name)
        else:
            attr = loading_func(attr_name)
            self.__setattr__(attr_name, attr)
            return attr

    def __load_table__(self, table_name) -> List[dict]:
        """
        Load a table and return it.
        :param table_name: The name of the table to load.
        :return: The table dictionary.
        """
        start_time = time.time()
        table_path = osp.join(self.table_root, '{}.json'.format(table_name))
        assert osp.exists(table_path), 'Error: Table %s does not exist!' % table_name
        with open(table_path) as f:
            table = json.load(f)
        end_time = time.time()

        # Print a message to stdout.
        if self.verbose:
            print("Loaded {} {}(s) in {:.3f}s,".format(len(table), table_name, end_time - start_time))

        return table

    # ### List methods. ###

    def list_attributes(self) -> None:
        """
        List all attributes and the number of annotations with each attribute.
        """
        # Load data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['attribute', 'object_ann'])

        # Count attributes.
        attribute_freqs = defaultdict(lambda: 0)
        for object_ann in self.object_ann:
            for attribute_token in object_ann['attribute_tokens']:
                attribute_freqs[attribute_token] += 1

        # Print to stdout.
        format_str = '{:11} {:24.24} {:48.48}'
        print()
        print(format_str.format('Annotations', 'Name', 'Description'))
        for attribute in self.attribute:
            print(format_str.format(
                attribute_freqs[attribute['token']], attribute['name'], attribute['description']))

    def list_cameras(self) -> None:
        """
        List all cameras and the number of samples for each.
        """
        # Load data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'sample_data', 'calibrated_sensor', 'sensor'])

        # Count cameras.
        cs_freqs = defaultdict(lambda: 0)
        channel_freqs = defaultdict(lambda: 0)
        for calibrated_sensor in self.calibrated_sensor:
            sensor = self.get('sensor', calibrated_sensor['sensor_token'])
            cs_freqs[sensor['channel']] += 1
        for sample_data in self.sample_data:
            if sample_data['is_key_frame']:  # Only use keyframes (samples).
                calibrated_sensor = self.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                sensor = self.get('sensor', calibrated_sensor['sensor_token'])
                channel_freqs[sensor['channel']] += 1

        # Print to stdout.
        format_str = '{:7} {:6} {:24}'
        print()
        print(format_str.format('Cameras', 'Samples', 'Channel'))
        for channel in cs_freqs.keys():
            cs_freq = cs_freqs[channel]
            channel_freq = channel_freqs[channel]
            print(format_str.format(
                cs_freq, channel_freq, channel))

    def list_categories(self, sample_tokens: List[str] = None) -> None:
        """
        List all categories and the number of object_anns and surface_anns for them.
        :param sample_tokens: A list of sample tokens for which category stats will be shown.
        """
        # Load data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'object_ann', 'surface_ann', 'category'])

        # Count object_anns and surface_anns.
        object_freqs = defaultdict(lambda: 0)
        surface_freqs = defaultdict(lambda: 0)
        if sample_tokens is not None:
            sample_tokens = set(sample_tokens)
        for object_ann in self.object_ann:
            sample_token = self.get('sample_data', object_ann['sample_data_token'])['sample_token']
            if sample_tokens is None or sample_token in sample_tokens:
                object_freqs[object_ann['category_token']] += 1
        for surface_ann in self.surface_ann:
            sample_token = self.get('sample_data', surface_ann['sample_data_token'])['sample_token']
            if sample_tokens is None or sample_token in sample_tokens:
                surface_freqs[surface_ann['category_token']] += 1

        # Print to stdout.
        format_str = '{:11} {:12} {:24.24} {:48.48}'
        print()
        print(format_str.format('Object_anns', 'Surface_anns', 'Name', 'Description'))
        for category in self.category:
            category_token = category['token']
            object_freq = object_freqs[category_token]
            surface_freq = surface_freqs[category_token]

            # Skip empty categories.
            if object_freq == 0 and surface_freq == 0:
                continue

            name = category['name']
            description = category['description']
            print(format_str.format(
                object_freq, surface_freq, name, description))

    def list_logs(self) -> None:
        """
        List all logs and the number of samples per log.
        """
        # Load data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'log'])

        # Count samples.
        sample_freqs = defaultdict(lambda: 0)
        for sample in self.sample:
            sample_freqs[sample['log_token']] += 1

        # Print to stdout.
        format_str = '{:6} {:29} {:24}'
        print()
        print(format_str.format('Samples', 'Log', 'Location'))
        for log in self.log:
            sample_freq = sample_freqs[log['token']]
            logfile = log['logfile']
            location = log['location']
            print(format_str.format(
                sample_freq, logfile, location))

    def list_sample_content(self, sample_token: str) -> None:
        """
        List the sample_datas for a given sample.
        :param sample_token: Sample token.
        """
        # Load data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'sample_data'])

        # Print content for each modality.
        sample = self.get('sample', sample_token)
        for modality in ['camera', 'lidar']:
            sample_data_tokens = self.get_sample_content(sample_token, modality)
            timestamps = np.array([self.get('sample_data', sd_token)['timestamp'] for sd_token in sample_data_tokens])
            rel_times = (timestamps - sample['timestamp']) / 1e6

            print('\nListing sample_datas for %s...' % modality)
            print('Rel. time\tSample_data token')
            for rel_time, sample_data_token in zip(rel_times, sample_data_tokens):
                print('{:>9.1f}\t{}'.format(rel_time, sample_data_token))

    def get_sample_content(self,
                           sample_token: str,
                           modality: str) -> List[str]:
        """
        For a given sample and modality, return all the sample_datas.
        :param sample_token: Sample token.
        :param modality: Sensor modality, either camera or lidar.
        :return: A list of sample_data tokens sorted by their timestamp.
        """
        assert modality in ['camera', 'lidar']
        sample = self.get('sample', sample_token)
        key_sd = self.get('sample_data', sample['key_%s_token' % modality])

        # Go forward.
        cur_sd = key_sd
        forward = []
        while cur_sd['next'] != '':
            cur_sd = self.get('sample_data', cur_sd['next'])
            forward.append(cur_sd['token'])

        # Go backward.
        cur_sd = key_sd
        backward = []
        while cur_sd['prev'] != '':
            cur_sd = self.get('sample_data', cur_sd['prev'])
            backward.append(cur_sd['token'])

        # Combine.
        result = backward[::-1] + [key_sd['token']] + forward
        return result

    # ### Rendering methods. ###

    def render_image(self,
                     sd_token_camera: str,
                     with_annotations: bool = True,
                     with_category: bool = False,
                     with_attributes: bool = False,
                     object_tokens: List[str] = None,
                     surface_tokens: List[str] = None,
                     render_scale: float = 1.0,
                     out_path: str = None) -> None:
        """
        Renders an image (sample_data), optionally with annotations overlaid.
        :param sd_token_camera: The token of the sample_data to be rendered.
        :param with_annotations: Whether to draw all annotations.
        :param with_category: Whether to include the category name at the top of a box.
        :param with_attributes: Whether to include attributes in the label tags.
        :param object_tokens: List of object annotation tokens. If given, only these annotations are drawn.
        :param surface_tokens: List of surface annotation tokens. If given, only these annotations are drawn.
        :param render_scale: The scale at which the image will be rendered. Use 1.0 for the original image size.
        :param out_path: The path where we save the depth image, or otherwise None.
        """
        # Validate inputs.
        sample_data = self.get('sample_data', sd_token_camera)
        assert sample_data['fileformat'] == 'jpg', 'Error: Cannot use render_image() on lidar pointclouds!'
        if not sample_data['is_key_frame']:
            assert not with_annotations, 'Error: Cannot render annotations for non keyframes!'
            assert not with_attributes, 'Error: Cannot render attributes for non keyframes!'

        # Get image data.
        im_path = osp.join(self.dataroot, sample_data['filename'])
        im = Image.open(im_path)

        # Initialize drawing.
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(im, 'RGBA')

        if with_annotations:
            # Load stuff / background regions.
            surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token_camera]
            if surface_tokens is not None:
                surface_anns = [o for o in surface_anns if o['token'] in surface_tokens]

            # Draw stuff / background regions.
            for ann in surface_anns:
                # Get color and mask
                category_token = ann['category_token']
                category_name = self.get('category', category_token)['name']
                color = self.color_map[category_name]
                if ann['mask'] is None:
                    continue
                mask = mask_decode(ann['mask'])

                draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))

            # Load object instances.
            object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token_camera]
            if object_tokens is not None:
                object_anns = [o for o in object_anns if o['token'] in object_tokens]

            # Draw object instances.
            for ann in object_anns:
                # Get color, box, mask and name.
                category_token = ann['category_token']
                category_name = self.get('category', category_token)['name']
                color = self.color_map[category_name]
                bbox = ann['bbox']
                attr_tokens = ann['attribute_tokens']
                attributes = [self.get('attribute', at) for at in attr_tokens]
                name = annotation_name(attributes, category_name, with_attributes=with_attributes)
                if ann['mask'] is None:
                    continue
                mask = mask_decode(ann['mask'])

                # Draw rectangle, text and mask.
                draw.rectangle(bbox, outline=color)
                if with_category:
                    draw.text((bbox[0], bbox[1]), name, font=font)
                draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))

        # Plot the image.
        (width, height) = im.size
        pix_to_inch = 100
        figsize = (height / pix_to_inch, width / pix_to_inch)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(im)

        # Save to disk.
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch * render_scale, pad_inches=0)
            plt.close()

    def render_depth(self,
                     sd_token_camera: str,
                     mode: str = 'sparse',
                     max_depth: float = None,
                     cmap: str = 'viridis',
                     render_scale: float = 1.0,
                     out_path: str = None) -> None:
        """
        This function plots an image and its depth map, either as a set of sparse points, or with depth completion.
        Default depth colors range from yellow (close) to blue (far). Missing values are blue.
        Suitable colormaps for depth maps are viridis and magma.
        :param sd_token_camera: The sample_data token of the camera image.
        :param mode: How to render the depth, either sparse or dense.
        :param max_depth: The maximum depth used for scaling the color values. If None, the actual maximum is used.
        :param cmap: The matplotlib color map name.
        :param scale: The scaling factor applied to the depth map.
        :param n_dilate: Dilation filter size.
        :param n_gauss: Gaussian filter size.
        :param sigma_gauss: Gaussian filter sigma.
        :param render_scale: The scale at which the image will be rendered. Use 1.0 for the original image size.
        :param out_path: The path where we save the depth image, or otherwise None.
        """
        # Get depth and image.
        points, depths, _, im_size = self.get_depth(sd_token_camera)

        # Compute depth image.
        assert mode in ['sparse', 'dense']
        if mode == 'sparse':
            scale = 1 / 8
            n_dilate = None
            n_gauss = None
            sigma_gauss = None
        else:
            scale = 1 / 2
            n_dilate = 23
            n_gauss = 11
            sigma_gauss = 3
        depth_im = depth_map(points, depths, im_size, scale=scale, n_dilate=n_dilate, n_gauss=n_gauss,
                             sigma_gauss=sigma_gauss)

        # Scale depth_im to full image size.
        depth_im = cv2.resize(depth_im, im_size)

        # Determine color scaling.
        min_depth = 0
        if max_depth is None:
            max_depth = depth_im.max()
        norm = InvertedNormalize(vmin=min_depth, vmax=max_depth)

        # Plot the image.
        (width, height) = depth_im.shape[::-1]
        pix_to_inch = 100
        figsize = (height / pix_to_inch, width / pix_to_inch)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(depth_im, norm=norm, cmap=cmap)

        # Save to disk.
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch * render_scale, pad_inches=0)
            plt.close()

    def get_depth(self,
                  sd_token_camera: str,
                  min_dist: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        This function picks out the lidar pcl closest to the given image timestamp and projects it onto the image.
        :param sd_token_camera: The sample_data token of the camera image.
        :param min_dist: Distance from the camera below which points are discarded.
        :return: (
            points: Lidar points (x, y) in pixel coordinates.
            depths: Depth in meters of each lidar point.
            timestamp_deltas: Timestamp difference between each lidar point and the camera image.
            im_size: Width and height.
        )
        """
        # Find closest pointcloud.
        sd_camera = self.get('sample_data', sd_token_camera)
        sample_lidar_tokens = self.get_sample_content(sd_camera['sample_token'], 'lidar')
        timestamps = np.array([self.get('sample_data', t)['timestamp'] for t in sample_lidar_tokens])
        time_diffs = np.abs(timestamps - sd_camera['timestamp']) / 1e6
        closest_idx = int(np.argmin(time_diffs))
        closest_time_diff = time_diffs[closest_idx]
        if closest_time_diff > 0.25:
            raise Exception('Error: Cannot render depth for an image that has no associated lidar pointcloud!'
                            'This is the case for about 0.9%% of the images.')
        # TODO: revisit this number, as some of the images may also be missing
        sd_token_lidar = sample_lidar_tokens[closest_idx]
        sd_lidar = self.get('sample_data', sd_token_lidar)

        # Retrieve size from meta data.
        im_size = (sd_camera['width'], sd_camera['height'])

        # Load pointcloud.
        pcl_path = osp.join(self.dataroot, sd_lidar['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        pointsensor = sd_lidar
        cam = sd_camera

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the pointcloud.
        # Distort in camera plane (note that this only happens in nuImages, not nuScenes.
        # In nuScenes all images are undistorted, in nuImages they are not.
        sensor = self.get('sensor', cs_record['sensor_token'])
        points, depths = distort_pointcloud(pc.points, np.array(cs_record['camera_distortion']),
                                                      sensor['channel'])

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im_size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im_size[1] - 1)
        points = points[:, mask]
        depths = depths[mask]

        # Compute timestamp delta between lidar and image.
        timestamp_deltas = closest_time_diff / 1e6 * np.ones(points.shape[1])

        return points, depths, timestamp_deltas, im_size

    def render_pointcloud(self):
        pass

    def get_trajectory(self, sample_token: str):
        pass
