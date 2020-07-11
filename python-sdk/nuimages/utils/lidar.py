# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

from typing import Tuple, Any

import cv2
import numpy as np
from matplotlib.colors import Normalize
from numpy.ma.core import MaskedArray


def depth_map(pts: np.ndarray,
              depths: np.ndarray,
              im_size: Tuple[int, int],
              scale: float = 1 / 8,
              n_dilate: int = None,
              n_gauss: int = None,
              sigma_gauss: float = None) -> np.ndarray:
    """
    This function computes a depth map given a lidar pointcloud projected to the camera.
    Depth completion can be used to sparsify the depth map.
    :param pts: <np.ndarray: 3, n> Lidar pointcloud in image coordinates.
    :param depths: <np.ndarray: n, 1> Depth of the points.
    :param im_size: The image width and height.
    :param scale: The scaling factor applied to the depth map.
    :param n_dilate: Dilation filter size.
    :param n_gauss: Gaussian filter size.
    :param sigma_gauss: Gaussian filter sigma.
    :return: The depth map.
    """
    # Store the minimum depth in the corresponding pixels.
    # Apply downsampling to make it more efficient and points larger to be more visible.
    pxs = (pts[0, :] * scale).astype(np.int32)
    pys = (pts[1, :] * scale).astype(np.int32)

    depth_map_size = np.array(im_size)[::-1] * scale
    depth_map_size = np.ceil(depth_map_size).astype(np.int32)
    depth_map = np.zeros(depth_map_size, dtype=np.float32)
    for x, y, depth in zip(pxs, pys, depths):
        if depth_map[y][x] == 0:
            depth_map[y][x] = depth
        else:
            depth_map[y][x] = min(depth_map[y][x], depth)

    # Set invalid pixels to max_depth.
    invalid = depth_map == 0
    depth_map[invalid] = np.max(depth_map)

    # Perform erosion to grow points
    if n_dilate is not None:
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_ERODE, np.ones((n_dilate, n_dilate), np.uint8))

    # Perform Gaussian blur to smoothen points.
    # Note that this should be used in moderation as the Gaussian filter also uses invalid depth values.
    if n_gauss is not None:
        blurred = cv2.GaussianBlur(depth_map, (n_gauss, n_gauss), sigma_gauss)
        valid = depth_map > 0
        depth_map[valid] = blurred[valid]

    return depth_map


def distort_pointcloud(points: np.ndarray, camera_distortion: np.ndarray, cam_name: str, r_sq_max: float = 1.0) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Distort the pointcloud coordinates to map into the image.
    Note: This function discards some invalid points, that do not project into the image.
          This happens if the radial distortion function is not injective, which is the case if k3 is negative.
          We also use the same mechanism to avoid float overflows in the k4 portion of CAM_BACK.
    :param r_sq_max: Hand-tuned parameter to address warping and numerical overflow.
    :param cam_name: Name of the camera.
    :param points: Lidar pointcloud.
    :param camera_distortion: Distortion coefficents of the camera.
    :return: Distorted pointcloud and depth values.
    """
    k1 = camera_distortion[0]
    k2 = camera_distortion[1]
    p1 = camera_distortion[2]
    p2 = camera_distortion[3]
    k3 = camera_distortion[4]

    # Store depth to return it.
    depths = points[2, :]

    # Normalize.
    points_x = points[0, :] / points[2, :]
    points_y = points[1, :] / points[2, :]
    r_sq = points_x ** 2 + points_y ** 2

    # Filter points from outside the frustum that are likely to map inside it.
    # This happens when the distortion function is not injective, i.e. when k3 < 0 for all cameras,
    # apart from CAM_BACK which has distortion coefficient k6, which prevents warping.
    # However, we also do it elsewhere to avoid numerical overflows.
    mask = r_sq < r_sq_max
    depths = depths[mask]
    points_x = points_x[mask]
    points_y = points_y[mask]
    r_sq = r_sq[mask]

    # Specify the basic distortion model.
    radial_distort = 1 + k1 * r_sq + k2 * r_sq ** 2 + k3 * r_sq ** 3

    # For fish-eye lenses, add another parameter to the distortion model.
    if cam_name == 'CAM_BACK':
        k4 = camera_distortion[5]
        radial_distort = radial_distort + k4 * r_sq ** 4
        assert not np.any(np.isinf(radial_distort)) and not np.any(np.isnan(radial_distort))

    # Apply distortion to points.
    x = radial_distort * points_x + 2 * p1 * points_x * points_y + p2 * (r_sq + 2 * points_x ** 2)
    y = radial_distort * points_y + p1 * (r_sq + 2 * points_y ** 2) + 2 * p2 * points_x * points_y

    # Define output.
    # Note that the third dimension is 1 as the points are already normalized above.
    points = np.ones((3, len(points_x)))
    points[0, :] = x
    points[1, :] = y
    assert points.shape[1] == len(depths), 'Error: Code is inconsistent!'

    return points, depths


class InvertedNormalize(Normalize):

    def __call__(self, value: MaskedArray, clip: Any = None) -> MaskedArray:
        """
        A custom inverted colormap that stretches the close depth values out to have more color resolution.
        :param value:
        :param clip:
        :return:
        """
        assert clip is None, 'Error: Clip option not supported!'

        # Define a non-linear mapping based on 4 keypoints.
        scaling_x = [0, 0.2, 0.5, 1]
        scaling_y = [0, 0.5, 0.95, 1]

        # Apply that scaling, taking into account the specified minimum and maximum.
        x = self.vmin + np.array(scaling_x) * (self.vmax - self.vmin)
        y = scaling_y
        colors = np.interp(value, x, y)
        return 1 - np.ma.masked_array(colors)
