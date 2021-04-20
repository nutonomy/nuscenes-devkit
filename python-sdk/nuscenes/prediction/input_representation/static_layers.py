# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
import colorsys
import os
from typing import Dict, List, Tuple, Callable

import cv2
import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import angle_of_rotation, angle_diff
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.prediction.input_representation.interface import \
    StaticLayerRepresentation
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords

Color = Tuple[float, float, float]


def load_all_maps(helper: PredictHelper, verbose: bool = False) -> Dict[str, NuScenesMap]:
    """
    Loads all NuScenesMap instances for all available maps.
    :param helper: Instance of PredictHelper.
    :param verbose: Whether to print to stdout.
    :return: Mapping from map-name to the NuScenesMap api instance.
    """
    dataroot = helper.data.dataroot
    maps = {}

    for map_name in locations:
        if verbose:
            print(f'static_layers.py - Loading Map: {map_name}')

        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps


def get_patchbox(x_in_meters: float, y_in_meters: float,
                 image_side_length: float) -> Tuple[float, float, float, float]:
    """
    Gets the patchbox representing the area to crop the base image.
    :param x_in_meters: X coordinate.
    :param y_in_meters: Y coordiante.
    :param image_side_length: Length of the image.
    :return: Patch box tuple.
    """

    patch_box = (x_in_meters, y_in_meters, image_side_length, image_side_length)

    return patch_box


def change_color_of_binary_mask(image: np.ndarray, color: Color) -> np.ndarray:
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


def get_lanes_in_radius(x: float, y: float, radius: float,
                        discretization_meters: float,
                        map_api: NuScenesMap) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Retrieves all the lanes and lane connectors in a radius of the query point.
    :param x: x-coordinate of point in global coordinates.
    :param y: y-coordinate of point in global coordinates.
    :param radius: Any lanes within radius meters of the (x, y) point will be returned.
    :param discretization_meters: How finely to discretize the lane. If 1 is given, for example,
        the lane will be discretized into a list of points such that the distances between points
        is approximately 1 meter.
    :param map_api: The NuScenesMap instance to query.
    :return: Mapping from lane id to list of coordinate tuples in global coordinate system.
    """

    lanes = map_api.get_records_in_radius(x, y, radius, ['lane', 'lane_connector'])
    lanes = lanes['lane'] + lanes['lane_connector']
    lanes = map_api.discretize_lanes(lanes, discretization_meters)

    return lanes


def color_by_yaw(agent_yaw_in_radians: float,
                 lane_yaw_in_radians: float) -> Color:
    """
    Color the pose one the lane based on its yaw difference to the agent yaw.
    :param agent_yaw_in_radians: Yaw of the agent with respect to the global frame.
    :param lane_yaw_in_radians: Yaw of the pose on the lane with respect to the global frame.
    """

    # By adding pi, lanes in the same direction as the agent are colored blue.
    angle = angle_diff(agent_yaw_in_radians, lane_yaw_in_radians, 2*np.pi) + np.pi

    # Convert to degrees per colorsys requirement
    angle = angle * 180/np.pi

    normalized_rgb_color = colorsys.hsv_to_rgb(angle/360, 1., 1.)

    color = [color*255 for color in normalized_rgb_color]

    # To make the return type consistent with Color definition
    return color[0], color[1], color[2]


def draw_lanes_on_image(image: np.ndarray,
                        lanes: Dict[str, List[Tuple[float, float, float]]],
                        agent_global_coords: Tuple[float, float],
                        agent_yaw_in_radians: float,
                        agent_pixels: Tuple[int, int],
                        resolution: float,
                        color_function: Callable[[float, float], Color] = color_by_yaw) -> np.ndarray:
    """
    Draws lanes on image.
    :param image: Image to draw lanes on. Preferably all-black or all-white image.
    :param lanes: Mapping from lane id to list of coordinate tuples in global coordinate system.
    :param agent_global_coords: Location of the agent in the global coordinate frame.
    :param agent_yaw_in_radians: Yaw of agent in radians.
    :param agent_pixels: Location of the agent in the image as (row_pixel, column_pixel).
    :param resolution: Resolution in meters/pixel.
    :param color_function: By default, lanes are colored by the yaw difference between the pose
    on the lane and the agent yaw. However, you can supply your own function to color the lanes.
    :return: Image (represented as np.ndarray) with lanes drawn.
    """

    for poses_along_lane in lanes.values():

        for start_pose, end_pose in zip(poses_along_lane[:-1], poses_along_lane[1:]):

            start_pixels = convert_to_pixel_coords(start_pose[:2], agent_global_coords,
                                                   agent_pixels, resolution)
            end_pixels = convert_to_pixel_coords(end_pose[:2], agent_global_coords,
                                                 agent_pixels, resolution)

            start_pixels = (start_pixels[1], start_pixels[0])
            end_pixels = (end_pixels[1], end_pixels[0])

            color = color_function(agent_yaw_in_radians, start_pose[2])

            # Need to flip the row coordinate and the column coordinate
            # because of cv2 convention
            cv2.line(image, start_pixels, end_pixels, color,
                     thickness=5)

    return image


def draw_lanes_in_agent_frame(image_side_length: int,
                              agent_x: float, agent_y: float,
                              agent_yaw: float,
                              radius: float,
                              image_resolution: float,
                              discretization_resolution_meters: float,
                              map_api: NuScenesMap,
                              color_function: Callable[[float, float], Color] = color_by_yaw) -> np.ndarray:
    """
    Queries the map api for the nearest lanes, discretizes them, draws them on an image
    and rotates the image so the agent heading is aligned with the postive y axis.
    :param image_side_length: Length of the image.
    :param agent_x: Agent X-coordinate in global frame.
    :param agent_y: Agent Y-coordinate in global frame.
    :param agent_yaw: Agent yaw, in radians.
    :param radius: Draws the lanes that are within radius meters of the agent.
    :param image_resolution: Image resolution in pixels / meter.
    :param discretization_resolution_meters: How finely to discretize the lanes.
    :param map_api: Instance of NuScenesMap.
    :param color_function: By default, lanes are colored by the yaw difference between the pose
        on the lane and the agent yaw. However, you can supply your own function to color the lanes.
    :return: np array with lanes drawn.
    """

    agent_pixels = int(image_side_length / 2), int(image_side_length / 2)
    base_image = np.zeros((image_side_length, image_side_length, 3))

    lanes = get_lanes_in_radius(agent_x, agent_y, radius, discretization_resolution_meters, map_api)

    image_with_lanes = draw_lanes_on_image(base_image, lanes, (agent_x, agent_y), agent_yaw,
                                           agent_pixels, image_resolution, color_function)

    rotation_mat = get_rotation_matrix(image_with_lanes.shape, agent_yaw)

    rotated_image = cv2.warpAffine(image_with_lanes, rotation_mat, image_with_lanes.shape[:2])

    return rotated_image.astype("uint8")


class StaticLayerRasterizer(StaticLayerRepresentation):
    """
    Creates a representation of the static map layers where
    the map layers are given a color and rasterized onto a
    three channel image.
    """

    def __init__(self, helper: PredictHelper,
                 layer_names: List[str] = None,
                 colors: List[Color] = None,
                 resolution: float = 0.1, # meters / pixel
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25):

        self.helper = helper
        self.maps = load_all_maps(helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        if not colors:
            colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255)]
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

        yaw_corrected = correct_yaw(yaw)

        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)

        patchbox = get_patchbox(x, y, image_side_length)

        angle_in_degrees = angle_of_rotation(yaw_corrected) * 180 / np.pi

        canvas_size = (image_side_length_pixels, image_side_length_pixels)

        masks = self.maps[map_name].get_map_mask(patchbox, angle_in_degrees, self.layer_names, canvas_size=canvas_size)

        images = []
        for mask, color in zip(masks, self.colors):
            images.append(change_color_of_binary_mask(np.repeat(mask[::-1, :, np.newaxis], 3, 2), color))

        lanes = draw_lanes_in_agent_frame(image_side_length_pixels, x, y, yaw, radius=50,
                                          image_resolution=self.resolution, discretization_resolution_meters=1,
                                          map_api=self.maps[map_name])

        images.append(lanes)

        image = self.combinator.combine(images)

        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))

        return image[row_crop, col_crop, :]
