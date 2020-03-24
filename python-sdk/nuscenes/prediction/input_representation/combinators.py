# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
from functools import reduce
from typing import List

import cv2
import numpy as np

from nuscenes.prediction.input_representation.interface import Combinator


def add_foreground_to_image(base_image: np.ndarray,
                            foreground_image: np.ndarray) -> np.ndarray:
    """
    Overlays a foreground image on top of a base image without mixing colors. Type uint8.
    :param base_image: Image that will be the background. Type uint8.
    :param foreground_image: Image that will be the foreground.
    :return: Image Numpy array of type uint8.
    """

    if not base_image.shape == foreground_image.shape:
        raise ValueError("base_image and foreground image must have the same shape."
                         " Received {} and {}".format(base_image.shape, foreground_image.shape))

    if not (base_image.dtype == "uint8" and foreground_image.dtype == "uint8"):
        raise ValueError("base_image and foreground image must be of type 'uint8'."
                         " Received {} and {}".format(base_image.dtype, foreground_image.dtype))

    img2gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(base_image, base_image, mask=mask_inv)
    img2_fg = cv2.bitwise_and(foreground_image, foreground_image, mask=mask)
    combined_image = cv2.add(img1_bg, img2_fg)
    return combined_image


class Rasterizer(Combinator):
    """
    Combines images into a three channel image.
    """

    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Combine three channel images into a single image.
        :param data: List of images to combine.
        :return: Numpy array representing image (type 'uint8')
        """
        # All images in the dict are the same shape
        image_shape = data[0].shape

        base_image = np.zeros(image_shape).astype("uint8")
        return reduce(add_foreground_to_image, [base_image] + data)
