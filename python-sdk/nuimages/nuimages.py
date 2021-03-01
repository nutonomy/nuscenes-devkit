# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

import json
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

from nuimages.utils.utils import annotation_name, mask_decode, get_font, name_to_index_mapping
from nuscenes.utils.color_map import get_colormap

PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")


class NuImages:
    """
    Database class for nuImages to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuimages',
                 lazy: bool = True,
                 verbose: bool = False):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0-train", "v1.0-val", "v1.0-test", "v1.0-mini").
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
            print("Done loading in {:.3f} seconds (lazy={}).\n======".format(time.time() - start_time, self.lazy))

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

    def shortcut(self, src_table: str, tgt_table: str, src_token: str) -> Dict[str, Any]:
        """
        Convenience function to navigate between different tables that have one-to-one relations.
        E.g. we can use this function to conveniently retrieve the sensor for a sample_data.
        :param src_table: The name of the source table.
        :param tgt_table: The name of the target table.
        :param src_token: The source token.
        :return: The entry of the destination table corresponding to the source token.
        """
        if src_table == 'sample_data' and tgt_table == 'sensor':
            sample_data = self.get('sample_data', src_token)
            calibrated_sensor = self.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            sensor = self.get('sensor', calibrated_sensor['sensor_token'])

            return sensor
        elif (src_table == 'object_ann' or src_table == 'surface_ann') and tgt_table == 'sample':
            src = self.get(src_table, src_token)
            sample_data = self.get('sample_data', src['sample_data_token'])
            sample = self.get('sample', sample_data['sample_token'])

            return sample
        else:
            raise Exception('Error: Shortcut from %s to %s not implemented!' % (src_table, tgt_table))

    def check_sweeps(self, filename: str) -> None:
        """
        Check that the sweeps folder was downloaded if required.
        :param filename: The filename of the sample_data.
        """
        assert filename.startswith('samples') or filename.startswith('sweeps'), \
            'Error: You passed an incorrect filename to check_sweeps(). Please use sample_data[''filename''].'

        if 'sweeps' in filename:
            sweeps_dir = osp.join(self.dataroot, 'sweeps')
            if not osp.isdir(sweeps_dir):
                raise Exception('Error: You are missing the "%s" directory! The devkit generally works without this '
                                'directory, but you cannot call methods that use non-keyframe sample_datas.'
                                % sweeps_dir)

    # ### List methods. ###

    def list_attributes(self, sort_by: str = 'freq') -> None:
        """
        List all attributes and the number of annotations with each attribute.
        :param sort_by: Sorting criteria, e.g. "name", "freq".
        """
        # Preload data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['attribute', 'object_ann'])

        # Count attributes.
        attribute_freqs = defaultdict(lambda: 0)
        for object_ann in self.object_ann:
            for attribute_token in object_ann['attribute_tokens']:
                attribute_freqs[attribute_token] += 1

        # Sort entries.
        if sort_by == 'name':
            sort_order = [i for (i, _) in sorted(enumerate(self.attribute), key=lambda x: x[1]['name'])]
        elif sort_by == 'freq':
            attribute_freqs_order = [attribute_freqs[c['token']] for c in self.attribute]
            sort_order = [i for (i, _) in
                          sorted(enumerate(attribute_freqs_order), key=lambda x: x[1], reverse=True)]
        else:
            raise Exception('Error: Invalid sorting criterion %s!' % sort_by)

        # Print to stdout.
        format_str = '{:11} {:24.24} {:48.48}'
        print()
        print(format_str.format('Annotations', 'Name', 'Description'))
        for s in sort_order:
            attribute = self.attribute[s]
            print(format_str.format(
                attribute_freqs[attribute['token']], attribute['name'], attribute['description']))

    def list_cameras(self) -> None:
        """
        List all cameras and the number of samples for each.
        """
        # Preload data if in lazy load to avoid confusing outputs.
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
                sensor = self.shortcut('sample_data', 'sensor', sample_data['token'])
                channel_freqs[sensor['channel']] += 1

        # Print to stdout.
        format_str = '{:15} {:7} {:25}'
        print()
        print(format_str.format('Calibr. sensors', 'Samples', 'Channel'))
        for channel in cs_freqs.keys():
            cs_freq = cs_freqs[channel]
            channel_freq = channel_freqs[channel]
            print(format_str.format(
                cs_freq, channel_freq, channel))

    def list_categories(self, sample_tokens: List[str] = None, sort_by: str = 'object_freq') -> None:
        """
        List all categories and the number of object_anns and surface_anns for them.
        :param sample_tokens: A list of sample tokens for which category stats will be shown.
        :param sort_by: Sorting criteria, e.g. "name", "object_freq", "surface_freq".
        """
        # Preload data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'object_ann', 'surface_ann', 'category'])

        # Count object_anns and surface_anns.
        object_freqs = defaultdict(lambda: 0)
        surface_freqs = defaultdict(lambda: 0)
        if sample_tokens is not None:
            sample_tokens = set(sample_tokens)

        for object_ann in self.object_ann:
            sample = self.shortcut('object_ann', 'sample', object_ann['token'])
            if sample_tokens is None or sample['token'] in sample_tokens:
                object_freqs[object_ann['category_token']] += 1

        for surface_ann in self.surface_ann:
            sample = self.shortcut('surface_ann', 'sample', surface_ann['token'])
            if sample_tokens is None or sample['token'] in sample_tokens:
                surface_freqs[surface_ann['category_token']] += 1

        # Sort entries.
        if sort_by == 'name':
            sort_order = [i for (i, _) in sorted(enumerate(self.category), key=lambda x: x[1]['name'])]
        elif sort_by == 'object_freq':
            object_freqs_order = [object_freqs[c['token']] for c in self.category]
            sort_order = [i for (i, _) in sorted(enumerate(object_freqs_order), key=lambda x: x[1], reverse=True)]
        elif sort_by == 'surface_freq':
            surface_freqs_order = [surface_freqs[c['token']] for c in self.category]
            sort_order = [i for (i, _) in sorted(enumerate(surface_freqs_order), key=lambda x: x[1], reverse=True)]
        else:
            raise Exception('Error: Invalid sorting criterion %s!' % sort_by)

        # Print to stdout.
        format_str = '{:11} {:12} {:24.24} {:48.48}'
        print()
        print(format_str.format('Object_anns', 'Surface_anns', 'Name', 'Description'))
        for s in sort_order:
            category = self.category[s]
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

    def list_anns(self, sample_token: str, verbose: bool = True) -> Tuple[List[str], List[str]]:
        """
        List all the annotations of a sample.
        :param sample_token: Sample token.
        :param verbose: Whether to print to stdout.
        :return: The object and surface annotation tokens in this sample.
        """
        # Preload data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'object_ann', 'surface_ann', 'category'])

        sample = self.get('sample', sample_token)
        key_camera_token = sample['key_camera_token']
        object_anns = [o for o in self.object_ann if o['sample_data_token'] == key_camera_token]
        surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == key_camera_token]

        if verbose:
            print('Printing object annotations:')
            for object_ann in object_anns:
                category = self.get('category', object_ann['category_token'])
                attribute_names = [self.get('attribute', at)['name'] for at in object_ann['attribute_tokens']]
                print('{} {} {}'.format(object_ann['token'], category['name'], attribute_names))

            print('\nPrinting surface annotations:')
            for surface_ann in surface_anns:
                category = self.get('category', surface_ann['category_token'])
                print(surface_ann['token'], category['name'])

        object_tokens = [o['token'] for o in object_anns]
        surface_tokens = [s['token'] for s in surface_anns]
        return object_tokens, surface_tokens

    def list_logs(self) -> None:
        """
        List all logs and the number of samples per log.
        """
        # Preload data if in lazy load to avoid confusing outputs.
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
        # Preload data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample', 'sample_data'])

        # Print content for each modality.
        sample = self.get('sample', sample_token)
        sample_data_tokens = self.get_sample_content(sample_token)
        timestamps = np.array([self.get('sample_data', sd_token)['timestamp'] for sd_token in sample_data_tokens])
        rel_times = (timestamps - sample['timestamp']) / 1e6

        print('\nListing sample content...')
        print('Rel. time\tSample_data token')
        for rel_time, sample_data_token in zip(rel_times, sample_data_tokens):
            print('{:>9.1f}\t{}'.format(rel_time, sample_data_token))

    def list_sample_data_histogram(self) -> None:
        """
        Show a histogram of the number of sample_datas per sample.
        """
        # Preload data if in lazy load to avoid confusing outputs.
        if self.lazy:
            self.load_tables(['sample_data'])

        # Count sample_datas for each sample.
        sample_counts = defaultdict(lambda: 0)
        for sample_data in self.sample_data:
            sample_counts[sample_data['sample_token']] += 1

        # Compute histogram.
        sample_counts_list = np.array(list(sample_counts.values()))
        bin_range = np.max(sample_counts_list) - np.min(sample_counts_list)
        if bin_range == 0:
            values = [len(sample_counts_list)]
            freqs = [sample_counts_list[0]]
        else:
            values, bins = np.histogram(sample_counts_list, bin_range)
            freqs = bins[1:]  # To get the frequency we need to use the right side of the bin.

        # Print statistics.
        print('\nListing sample_data frequencies..')
        print('# images\t# samples')
        for freq, val in zip(freqs, values):
            print('{:>8d}\t{:d}'.format(int(freq), int(val)))

    # ### Getter methods. ###

    def get_sample_content(self,
                           sample_token: str) -> List[str]:
        """
        For a given sample, return all the sample_datas in chronological order.
        :param sample_token: Sample token.
        :return: A list of sample_data tokens sorted by their timestamp.
        """
        sample = self.get('sample', sample_token)
        key_sd = self.get('sample_data', sample['key_camera_token'])

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

    def get_ego_pose_data(self,
                          sample_token: str,
                          attribute_name: str = 'translation') -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the ego pose data of the <= 13 sample_datas associated with this sample.
        The method return translation, rotation, rotation_rate, acceleration and speed.
        :param sample_token: Sample token.
        :param attribute_name: The ego_pose field to extract, e.g. "translation", "acceleration" or "speed".
        :return: (
            timestamps: The timestamp of each ego_pose.
            attributes: A matrix with sample_datas x len(attribute) number of fields.
        )
        """
        assert attribute_name in ['translation', 'rotation', 'rotation_rate', 'acceleration', 'speed'], \
            'Error: The attribute_name %s is not a valid option!' % attribute_name

        if attribute_name == 'speed':
            attribute_len = 1
        elif attribute_name == 'rotation':
            attribute_len = 4
        else:
            attribute_len = 3

        sd_tokens = self.get_sample_content(sample_token)
        attributes = np.zeros((len(sd_tokens), attribute_len))
        timestamps = np.zeros((len(sd_tokens)))
        for i, sd_token in enumerate(sd_tokens):
            # Get attribute.
            sample_data = self.get('sample_data', sd_token)
            ego_pose = self.get('ego_pose', sample_data['ego_pose_token'])
            attribute = ego_pose[attribute_name]

            # Store results.
            attributes[i, :] = attribute
            timestamps[i] = ego_pose['timestamp']

        return timestamps, attributes

    def get_trajectory(self,
                       sample_token: str,
                       rotation_yaw: float = 0.0,
                       center_key_pose: bool = True) -> Tuple[np.ndarray, int]:
        """
        Get the trajectory of the ego vehicle and optionally rotate and center it.
        :param sample_token: Sample token.
        :param rotation_yaw: Rotation of the ego vehicle in the plot.
            Set to None to use lat/lon coordinates.
            Set to 0 to point in the driving direction at the time of the keyframe.
            Set to any other value to rotate relative to the driving direction (in radians).
        :param center_key_pose: Whether to center the trajectory on the key pose.
        :return: (
            translations: A matrix with sample_datas x 3 values of the translations at each timestamp.
            key_index: The index of the translations corresponding to the keyframe (usually 6).
        )
        """
        # Get trajectory data.
        timestamps, translations = self.get_ego_pose_data(sample_token)

        # Find keyframe translation and rotation.
        sample = self.get('sample', sample_token)
        sample_data = self.get('sample_data', sample['key_camera_token'])
        ego_pose = self.get('ego_pose', sample_data['ego_pose_token'])
        key_rotation = Quaternion(ego_pose['rotation'])
        key_timestamp = ego_pose['timestamp']
        key_index = [i for i, t in enumerate(timestamps) if t == key_timestamp][0]

        # Rotate points such that the initial driving direction points upwards.
        if rotation_yaw is not None:
            rotation = key_rotation.inverse * Quaternion(axis=[0, 0, 1], angle=np.pi / 2 - rotation_yaw)
            translations = np.dot(rotation.rotation_matrix, translations.T).T

        # Subtract origin to have lower numbers on the axes.
        if center_key_pose:
            translations -= translations[key_index, :]

        return translations, key_index

    def get_segmentation(self,
                         sd_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces two segmentation masks as numpy arrays of size H x W each, where H and W are the height and width
        of the camera image respectively:
            - semantic mask: A mask in which each pixel is an integer value between 0 to C (inclusive),
                             where C is the number of categories in nuImages. Each integer corresponds to
                             the index of the class in the category.json.
            - instance mask: A mask in which each pixel is an integer value between 0 to N, where N is the
                             number of objects in a given camera sample_data. Each integer corresponds to
                             the order in which the object was drawn into the mask.
        :param sd_token: The token of the sample_data to be rendered.
        :return: Two 2D numpy arrays (one semantic mask <int32: H, W>, and one instance mask <int32: H, W>).
        """
        # Validate inputs.
        sample_data = self.get('sample_data', sd_token)
        assert sample_data['is_key_frame'], 'Error: Cannot render annotations for non keyframes!'

        name_to_index = name_to_index_mapping(self.category)

        # Get image data.
        self.check_sweeps(sample_data['filename'])
        im_path = osp.join(self.dataroot, sample_data['filename'])
        im = Image.open(im_path)

        (width, height) = im.size
        semseg_mask = np.zeros((height, width)).astype('int32')
        instanceseg_mask = np.zeros((height, width)).astype('int32')

        # Load stuff / surface regions.
        surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token]

        # Draw stuff / surface regions.
        for ann in surface_anns:
            # Get color and mask.
            category_token = ann['category_token']
            category_name = self.get('category', category_token)['name']
            if ann['mask'] is None:
                continue
            mask = mask_decode(ann['mask'])

            # Draw mask for semantic segmentation.
            semseg_mask[mask == 1] = name_to_index[category_name]

        # Load object instances.
        object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token]

        # Sort by token to ensure that objects always appear in the instance mask in the same order.
        object_anns = sorted(object_anns, key=lambda k: k['token'])

        # Draw object instances.
        # The 0 index is reserved for background; thus, the instances should start from index 1.
        for i, ann in enumerate(object_anns, start=1):
            # Get color, box, mask and name.
            category_token = ann['category_token']
            category_name = self.get('category', category_token)['name']
            if ann['mask'] is None:
                continue
            mask = mask_decode(ann['mask'])

            # Draw masks for semantic segmentation and instance segmentation.
            semseg_mask[mask == 1] = name_to_index[category_name]
            instanceseg_mask[mask == 1] = i

        return semseg_mask, instanceseg_mask

    # ### Rendering methods. ###

    def render_image(self,
                     sd_token: str,
                     annotation_type: str = 'all',
                     with_category: bool = False,
                     with_attributes: bool = False,
                     object_tokens: List[str] = None,
                     surface_tokens: List[str] = None,
                     render_scale: float = 1.0,
                     box_line_width: int = -1,
                     font_size: int = None,
                     out_path: str = None) -> None:
        """
        Renders an image (sample_data), optionally with annotations overlaid.
        :param sd_token: The token of the sample_data to be rendered.
        :param annotation_type: The types of annotations to draw on the image; there are four options:
            'all': Draw surfaces and objects, subject to any filtering done by object_tokens and surface_tokens.
            'surfaces': Draw only surfaces, subject to any filtering done by surface_tokens.
            'objects': Draw objects, subject to any filtering done by object_tokens.
            'none': Neither surfaces nor objects will be drawn.
        :param with_category: Whether to include the category name at the top of a box.
        :param with_attributes: Whether to include attributes in the label tags. Note that with_attributes=True
            will only work if with_category=True.
        :param object_tokens: List of object annotation tokens. If given, only these annotations are drawn.
        :param surface_tokens: List of surface annotation tokens. If given, only these annotations are drawn.
        :param render_scale: The scale at which the image will be rendered. Use 1.0 for the original image size.
        :param box_line_width: The box line width in pixels. The default is -1.
            If set to -1, box_line_width equals render_scale (rounded) to be larger in larger images.
        :param font_size: Size of the text in the rendered image. Use None for the default size.
        :param out_path: The path where we save the rendered image, or otherwise None.
            If a path is provided, the plot is not shown to the user.
        """
        # Validate inputs.
        sample_data = self.get('sample_data', sd_token)
        if not sample_data['is_key_frame']:
            assert annotation_type == 'none', 'Error: Cannot render annotations for non keyframes!'
            assert not with_attributes, 'Error: Cannot render attributes for non keyframes!'
        if with_attributes:
            assert with_category, 'In order to set with_attributes=True, with_category must be True.'
        assert type(box_line_width) == int, 'Error: box_line_width must be an integer!'
        if box_line_width == -1:
            box_line_width = int(round(render_scale))

        # Get image data.
        self.check_sweeps(sample_data['filename'])
        im_path = osp.join(self.dataroot, sample_data['filename'])
        im = Image.open(im_path)

        # Initialize drawing.
        if with_category and font_size is not None:
            font = get_font(font_size=font_size)
        else:
            font = None
        im = im.convert('RGBA')
        draw = ImageDraw.Draw(im, 'RGBA')

        annotations_types = ['all', 'surfaces', 'objects', 'none']
        assert annotation_type in annotations_types, \
            'Error: {} is not a valid option for annotation_type. ' \
            'Only {} are allowed.'.format(annotation_type, annotations_types)
        if annotation_type is not 'none':
            if annotation_type == 'all' or annotation_type == 'surfaces':
                # Load stuff / surface regions.
                surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token]
                if surface_tokens is not None:
                    sd_surface_tokens = set([s['token'] for s in surface_anns if s['token']])
                    assert set(surface_tokens).issubset(sd_surface_tokens), \
                        'Error: The provided surface_tokens do not belong to the sd_token!'
                    surface_anns = [o for o in surface_anns if o['token'] in surface_tokens]

                # Draw stuff / surface regions.
                for ann in surface_anns:
                    # Get color and mask.
                    category_token = ann['category_token']
                    category_name = self.get('category', category_token)['name']
                    color = self.color_map[category_name]
                    if ann['mask'] is None:
                        continue
                    mask = mask_decode(ann['mask'])

                    # Draw mask. The label is obvious from the color.
                    draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))

            if annotation_type == 'all' or annotation_type == 'objects':
                # Load object instances.
                object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token]
                if object_tokens is not None:
                    sd_object_tokens = set([o['token'] for o in object_anns if o['token']])
                    assert set(object_tokens).issubset(sd_object_tokens), \
                        'Error: The provided object_tokens do not belong to the sd_token!'
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
                    if ann['mask'] is not None:
                        mask = mask_decode(ann['mask'])

                        # Draw mask, rectangle and text.
                        draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                        draw.rectangle(bbox, outline=color, width=box_line_width)
                        if with_category:
                            draw.text((bbox[0], bbox[1]), name, font=font)

        # Plot the image.
        (width, height) = im.size
        pix_to_inch = 100 / render_scale
        figsize = (height / pix_to_inch, width / pix_to_inch)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(im)

        # Save to disk.
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch, pad_inches=0)
            plt.close()

    def render_trajectory(self,
                          sample_token: str,
                          rotation_yaw: float = 0.0,
                          center_key_pose: bool = True,
                          out_path: str = None) -> None:
        """
        Render a plot of the trajectory for the clip surrounding the annotated keyframe.
        A red cross indicates the starting point, a green dot the ego pose of the annotated keyframe.
        :param sample_token: Sample token.
        :param rotation_yaw: Rotation of the ego vehicle in the plot.
            Set to None to use lat/lon coordinates.
            Set to 0 to point in the driving direction at the time of the keyframe.
            Set to any other value to rotate relative to the driving direction (in radians).
        :param center_key_pose: Whether to center the trajectory on the key pose.
        :param out_path: Optional path to save the rendered figure to disk.
            If a path is provided, the plot is not shown to the user.
        """
        # Get the translations or poses.
        translations, key_index = self.get_trajectory(sample_token, rotation_yaw=rotation_yaw,
                                                      center_key_pose=center_key_pose)

        # Render translations.
        plt.figure()
        plt.plot(translations[:, 0], translations[:, 1])
        plt.plot(translations[key_index, 0], translations[key_index, 1], 'go', markersize=10)  # Key image.
        plt.plot(translations[0, 0], translations[0, 1], 'rx', markersize=10)  # Start point.
        max_dist = translations - translations[key_index, :]
        max_dist = np.ceil(np.max(np.abs(max_dist)) * 1.05)  # Leave some margin.
        max_dist = np.maximum(10, max_dist)
        plt.xlim([translations[key_index, 0] - max_dist, translations[key_index, 0] + max_dist])
        plt.ylim([translations[key_index, 1] - max_dist, translations[key_index, 1] + max_dist])
        plt.xlabel('x in meters')
        plt.ylabel('y in meters')

        # Save to disk.
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=150, pad_inches=0)
            plt.close()
