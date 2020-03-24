# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
from typing import Tuple

import cv2
import numpy as np

from nuscenes.prediction.helper import angle_of_rotation


def convert_to_pixel_coords(location: Tuple[float, float],
                            center_of_image_in_global: Tuple[float, float],
                            center_of_image_in_pixels: Tuple[float, float],
                            resolution: float = 0.1) -> Tuple[int, int]:
    """
    Convert from global coordinates to pixel coordinates.
    :param location: Location in global coordinates as (x, y) tuple.
    :param center_of_image_in_global: Center of the image in global coordinates (x, y) tuple.
    :param center_of_image_in_pixels: Center of the image in pixel coordinates (row_pixel, column pixel).
    :param resolution: Center of image.
    """

    x, y = location
    x_offset = (x - center_of_image_in_global[0])
    y_offset = (y - center_of_image_in_global[1])

    x_pixel = x_offset / resolution

    # Negate the y coordinate because (0, 0) is ABOVE and to the LEFT
    y_pixel = -y_offset / resolution

    row_pixel = int(center_of_image_in_pixels[0] + y_pixel)
    column_pixel = int(center_of_image_in_pixels[1] + x_pixel)

    return row_pixel, column_pixel


def get_crops(meters_ahead: float, meters_behind: float,
              meters_left: float, meters_right: float,
              resolution: float,
              image_side_length_pixels: int) -> Tuple[slice, slice]:
    """
    Crop the excess pixels and centers the agent at the (meters_ahead, meters_left)
    coordinate in the image.
    :param meters_ahead: Meters ahead of the agent.
    :param meters_behind: Meters behind of the agent.
    :param meters_left: Meters to the left of the agent.
    :param meters_right: Meters to the right of the agent.
    :param resolution: Resolution of image in pixels / meters.
    :param image_side_length_pixels: Length of the image in pixels.
    :return: Tuple of row and column slices to crop image.
    """

    row_crop = slice(0, int((meters_ahead + meters_behind) / resolution))
    col_crop = slice(int(image_side_length_pixels / 2 - (meters_left / resolution)),
                     int(image_side_length_pixels / 2 + (meters_right / resolution)))

    return row_crop, col_crop


def get_rotation_matrix(image_shape: Tuple[int, int, int], yaw_in_radians: float) -> np.ndarray:
    """
    Gets a rotation matrix to rotate a three channel image so that
    yaw_in_radians points along the positive y-axis.
    :param image_shape: (Length, width, n_channels).
    :param yaw_in_radians: Angle to rotate the image by.
    :return: rotation matrix represented as np.ndarray.
    :return: The rotation matrix.
    """

    rotation_in_degrees = angle_of_rotation(yaw_in_radians) * 180 / np.pi

    return cv2.getRotationMatrix2D((image_shape[1] / 2, image_shape[0] / 2), rotation_in_degrees, 1)
