# nuScenes dev-kit.
# Code written by Qiang Xu, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os.path as osp
from typing import Tuple

import numpy as np
import cv2


class MapMask:
    def __init__(self, img_file: str, precision: float=0.1, foreground: int=255, background: int=0,
                 dist_thresh: float=2.0):
        """
        Init a map mask object that contains the semantic prior (drivable surface and sidewalks) mask.
        :param img_file: File path to map png file.
        :param precision: Precision in meters.
        :param foreground: Foreground value.
        :param background: Background value.
        :param dist_thresh: This parameter specifies the threshold on the distance from the semantic prior mask.
            The semantic prior mask is dilated to include points which are within this distance from itself.
        """
        assert osp.exists(img_file), 'map mask {} does not exist'.format(img_file)
        self.img_file = img_file
        self.precision = precision
        self.foreground = foreground
        self.background = background
        self.dist_thresh = dist_thresh
        self._mask = None
        self._distance_mask = None  # Distance to semantic_prior (lazy load).
        self._binary_mask = None  # Binary mask of semantic prior + dilation of dist_thresh (lazy load)
        self._transf_matrix = None  # Transformation matrix from global coords to map coords (lazy load).

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
            # Distance to nearest foreground in mask.
            self._distance_mask = cv2.distanceTransform((self.foreground - self.mask).astype(np.uint8), cv2.DIST_L2, 5)
            self._distance_mask = (self._distance_mask * self.precision).astype(np.float32)

        return self._distance_mask

    @property
    def binary_mask(self) -> np.array:
        """
        Create binary mask of semantic prior plus dilation.
        :return: <np.uint8: image.height, image.width>. The binary mask.
        """
        if self._binary_mask is None:
            self._binary_mask = (self.distance_mask < self.dist_thresh).astype(np.uint8) * self.foreground

        return self._binary_mask

    def export_to_png(self, filename: str='mask.png') -> None:
        """
        Export mask to png file.
        :param filename: Path to the png file.
        """
        cv2.imwrite(filename=filename, img=self.mask)

    def is_on_mask(self, x, y) -> np.array:
        """
        Determine whether the points are on binary mask.
        :param x: Global x coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :param y: Global y coordinates. Can be a scalar, list or a numpy array of x coordinates.
        :return: <np.bool: x.shape>. Whether the points are on the mask.
        """

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        assert x.shape == y.shape

        if x.ndim == 0:
            x = np.atleast_1d(x)
        if y.ndim == 0:
            y = np.atleast_1d(y)

        assert x.ndim == y.ndim == 1

        px, py = self.get_pixel(x, y)

        on_mask = np.ones(x.size, dtype=np.bool)

        on_mask[px < 0] = False
        on_mask[px >= self.binary_mask.shape[1]] = False
        on_mask[py < 0] = False
        on_mask[py >= self.binary_mask.shape[0]] = False

        on_mask[on_mask] = (self.binary_mask[py[on_mask], px[on_mask]] == self.foreground)

        return on_mask

    def dist_to_mask(self, x: float, y: float) -> float:
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

    def get_pixel(self, x: np.array, y: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image coordinates given the x-y coordinates of points.
        :param x: Global x coordinates.
        :param y: Global y coordinates.
        :return: (px <np.uint8: x.shape>, py <np.uint8: y.shape>). Pixel coordinates in map.
        """
        px, py = (x / self.precision).astype(int), self.mask.shape[0] - (y / self.precision).astype(int)

        return px, py

    def set_dist_thresh(self, dist_thresh: float) -> None:
        """
        This function sets self.dist_thresh to a new value. This method can be used to change the threshold multiple
        times during an experiment without creating new NuScenes objects.
        :param dist_thresh: The semantic prior mask is dilated to include points which are within this distance from
            itself.
        """
        self.dist_thresh = dist_thresh

        # Update the binary mask to None since the distance threshold was changed.
        self._binary_mask = None
