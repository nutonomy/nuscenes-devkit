# nuScenes dev-kit. Version 0.1
# Code written by Qiang Xu, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os.path as osp
from typing import Tuple

import numpy as np
import cv2


class MapMask:
    def __init__(self, img_file: str, precision: float=0.1, foreground: int=255, background: int=0):
        """
        Init a map mask object that contains the semantic prior (drivable surface and sidewalks) mask.
        :param img_file: File path to map png file.
        :param precision: Precision in meters.
        :param foreground: Foreground value.
        :param background: Background value.
        """
        assert osp.exists(img_file), 'map mask {} does not exist'.format(img_file)
        self.img_file = img_file
        self.precision = precision
        self.foreground = foreground
        self.background = background
        self._mask = None
        self._distance_mask = None  # Distance to semantic_prior. (lazy load).
        self._transf_matrix = None  # Transformation matrix from global coords to map coords. (lazy load).

    @property
    def mask(self) -> np.ndarray:
        """
        Create binary mask from the png file.
        :return: <np.int8: image.height, image.width>. The binary mask.
        """
        if self._mask is None:
            img = cv2.imread(self.img_file, cv2.IMREAD_GRAYSCALE)
            self._mask = np.array(img)
            self._mask[self._mask < 225] = self.background
            self._mask[self._mask >= 225] = self.foreground

        return self._mask

    @property
    def transform_matrix(self) -> np.ndarray:
        """
        Generate transform matrix for this map mask.
        :return: <np.array: 4, 4>. The transformation matrix.
        """
        if self._transf_matrix is None:
            mask_shape = self.mask.shape
            self._transf_matrix = np.array([[1.0 / self.precision, 0, 0, 0],
                                            [0, -1.0 / self.precision, 0, mask_shape[0]],
                                            [0, 0, 1, 0], [0, 0, 0, 1]])
        return self._transf_matrix

    @property
    def distance_mask(self) -> np.ndarray:
        """
        Generate distance mask from self.mask which is the original mask from the png file.
        :return: <np.float32: image.height, image.width>. The distance mask.
        """
        if self._distance_mask is None:
            # distance to nearest foreground in mask
            self._distance_mask = cv2.distanceTransform((self.foreground - self.mask).astype(np.uint8), cv2.DIST_L2, 5)
            self._distance_mask = (self._distance_mask * self.precision).astype(np.float32)

        return self._distance_mask

    def export_to_png(self, filename: str='mask.png') -> None:
        """
        Export mask to png file.
        :param filename: Path to the png file.
        """
        cv2.imwrite(filename=filename, img=self.mask)

    def is_on_mask(self, x: float, y: float) -> bool:
        """
        Determine whether a point is on the semantic_prior mask.
        :param x: Global x.
        :param y: Global y.
        :return: Whether the points is on the mask.
        """

        px, py = self.get_pixel(x, y)

        if px < 0 or px >= self.mask.shape[1] or py < 0 or py >= self.mask.shape[0]:
            return False

        return self.mask[py, px] == self.foreground

    def dist_to_mask(self, x: float, y:float) -> float:
        """
        Get the distance of a point to the nearest foreground in perception GT semantic prior mask (dilated).
        :param x: Global x.
        :param y: Global y.
        :return: Distance to nearest foreground, if not in distance mask, return -1.
        """
        px, py = self.get_pixel(x, y)

        if px < 0 or px >= self.distance_mask.shape[1] or py < 0 or py >= self.distance_mask.shape[0]:
            return -1

        return self.distance_mask[py, px]

    def get_pixel(self, x: float, y: float) -> Tuple[int]:
        """
        Get the image coordinates given a x-y point.
        :param x: Global x.
        :param y: Global y.
        :return: (px <int>, py <int>). Pixel coordinates in map.
        """
        px, py = int(x / self.precision), self.mask.shape[0] - int(y / self.precision)

        return px, py
