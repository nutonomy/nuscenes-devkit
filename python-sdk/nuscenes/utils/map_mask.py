# nuScenes dev-kit.
# Code written by Qiang Xu, 2018.
# Licensed under the Creative Commons [see licence.txt]

import os.path as osp
from typing import Tuple

import numpy as np
import cv2
from PIL import Image

# Set the maximum loadable image size.
Image.MAX_IMAGE_PIXELS = 400000 * 400000


class MapMask:
    def __init__(self, img_file: str):
        """
        Init a map mask object that contains the semantic prior (drivable surface and sidewalks) mask.
        :param img_file: File path to map png file.
        """
        assert osp.exists(img_file), 'map mask {} does not exist'.format(img_file)
        self.img_file = img_file
        self.precision = 0.1    # Precision in meters.
        self.foreground = 255
        self.background = 0
        self._mask = None   # Binary map mask (lazy load).
        self._transf_matrix = None  # Transformation matrix from global coords to map coords (lazy load).

    @property
    def mask(self) -> np.ndarray:
        """
        Create binary mask from the png file.
        :return: <np.int8: image.height, image.width>. The binary mask.
        """
        if self._mask is None:
            # Pillow allows us to specify the maximum image size above, whereas this is more difficult in OpenCV.
            img = Image.open(self.img_file).convert('L')
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

    def dilate_mask(self, dist_thresh: float = 2.0):
        """
        Dilates the mask by a distance threshold.
        :param dist_thresh: This parameter specifies the threshold on the distance from the semantic prior mask.
            The semantic prior mask is dilated to include points which are within this distance from itself.
        :return:
        """
        # Distance to nearest foreground in mask.
        distance_mask = cv2.distanceTransform((self.foreground - self.mask).astype(np.uint8), cv2.DIST_L2, 5)
        distance_mask = (distance_mask * self.precision).astype(np.float32)

        self._mask = (distance_mask < dist_thresh).astype(np.uint8) * self.foreground

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
        on_mask[px >= self.mask.shape[1]] = False
        on_mask[py < 0] = False
        on_mask[py >= self.mask.shape[0]] = False

        on_mask[on_mask] = (self.mask[py[on_mask], px[on_mask]] == self.foreground)

        return on_mask

    def get_pixel(self, x: np.array, y: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image coordinates given the x-y coordinates of points.
        :param x: Global x coordinates.
        :param y: Global y coordinates.
        :return: (px <np.uint8: x.shape>, py <np.uint8: y.shape>). Pixel coordinates in map.
        """
        px, py = (x / self.precision).astype(int), self.mask.shape[0] - (y / self.precision).astype(int)

        return px, py

