# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import os
from typing import Dict, List, Tuple

import numpy as np
import cv2
from pyquaternion import Quaternion

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.predict import PredictHelper
from nuscenes.predict.helper import angle_of_rotation
from nuscenes.predict.input_representation.combinators import Rasterizer
from nuscenes.predict.input_representation.interface import \
    StaticLayerRepresentation
from nuscenes.predict.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords


def load_all_maps(helper: PredictHelper) -> Dict[str, NuScenesMap]:
    """
    Loads all NuScenesMap instances for all available maps.
    :param helper: Instance of PredictHelper.
    :return: Mapping from map-name to the NuScenesMap api instance.
    """
    dataroot = helper.data.dataroot

    json_files = filter(lambda f: "json" in f, os.listdir(os.path.join(dataroot, "maps")))
    maps = {}

    for map_file in json_files:
        print(map_file)
        map_name = map_file.split(".")[0]
        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps

def get_patchbox(x_in_meters: float, y_in_meters: float,
                 image_side_length: float) -> Tuple[float, float, float, float]:
    """
    Gets the patchbox representing the area to crop the base image.
    :param x_in_meters: X coordinate.
    :param y_in_meters: Y coordiante.
    :param image_side_length: Length of the image
    :return: Patch box tuple
    """

    patch_box = (x_in_meters, y_in_meters, image_side_length, image_side_length)

    return patch_box

def change_color_of_binary_mask(image: np.ndarray, color: Tuple[float, float, float]) -> np.ndarray:
    """
    Changes color of binary mask. The image has values 0 or 1 but has three channels.
    :param image: Image with either 0 or 1 values and three channels.
    :param color: RGB color tuple.
    :return: Image with color changed (type uint8).
    """

    image = image * color

    # Return as type int so cv2 can manipulate it later.
    image = image.astype("uint8")

    return image

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def draw_lanes_in_agent_frame(image_side_length,
                              agent_x: float, agent_y: float,
                              agent_yaw: float,
                              radius: float,
                              image_resolution: float,
                              discretization_resolution_meters: float,
                              map_api: NuScenesMap) -> np.ndarray:

    central_track_pixels = (image_side_length / 2, image_side_length / 2)

    base_image = np.zeros((image_side_length, image_side_length, 3))

    lanes = map_api.get_records_in_radius(agent_x, agent_y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_resolution_meters)

    for poses_along_lane in lanes.values():

        for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):

            start_pixels = convert_to_pixel_coords(start_pose[:2], (agent_x, agent_y),
                                                   central_track_pixels, image_resolution)
            end_pixels = convert_to_pixel_coords(end_pose[:2], (agent_x, agent_y),
                                                 central_track_pixels, image_resolution)

            start_pixels = (start_pixels[1], start_pixels[0])
            end_pixels = (end_pixels[1], end_pixels[0])

            # Need to flip the row coordinate and the column coordinate
            # because of cv2 convention
            cv2.line(base_image, start_pixels, end_pixels, [0, 200, 200],
                     thickness=5)

    rotation_mat = get_rotation_matrix(base_image.shape, agent_yaw)

    rotated_image = cv2.warpAffine(base_image, rotation_mat, (base_image.shape[1],
                                                              base_image.shape[0]))

    return rotated_image.astype("uint8")


class StaticLayerRasterizer(StaticLayerRepresentation):
    """
    Creates a representation of the static map layers where
    the map layers are given a color and rasterized onto a
    three channel image.
    """

    def __init__(self, helper: PredictHelper,
                 layer_names: List[str] = None,
                 colors: List[Tuple[float, float, float]] = None,
                 resolution: float = 0.1, # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25):

        self.helper = helper
        self.maps = load_all_maps(helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        if not colors:
            colors = [(255, 255, 255), (0, 255, 0), (0, 0, 255)]
        self.colors = colors

        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right
        self.combinator = Rasterizer()

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        """
        Makes rasterized representation of static map layers.
        :param instance_token: Token for instance.
        :param sample_token: Token for sample.
        :return: Three channel image.
        """

        sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_token)

        x, y = sample_annotation['translation'][:2]

        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))

        print(yaw)

        yaw_corrected = correct_yaw(yaw)

        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)

        patchbox = get_patchbox(x, y, image_side_length)

        angle_in_degrees = angle_of_rotation(yaw_corrected) * 180 / np.pi

        masks = self.maps[map_name].get_map_mask(patchbox, angle_in_degrees, self.layer_names, canvas_size=None)

        images = [change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color) for mask, color in zip(masks, self.colors)]

        lanes = draw_lanes_in_agent_frame(int(image_side_length / self.resolution), x, y, yaw, 50, self.resolution, 1, self.maps[map_name])

        images.append(lanes)

        image = self.combinator.combine(images)

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))

        return image[row_crop, col_crop, :]
