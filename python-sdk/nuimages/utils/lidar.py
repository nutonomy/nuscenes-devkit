# nuScenes dev-kit.
# Code written by Asha Asvathaman & Holger Caesar, 2020.

from typing import Tuple

import cv2
import numpy as np
from matplotlib.colors import Normalize


def depth_map(pts: np.ndarray, depths: np.ndarray, im_size: Tuple[int, int], mode: str = 'sparse',
              scale: float = 0.5, n_dilate: int = 25, n_gauss: int = 11, sigma_gauss: float = 3) -> np.ndarray:
    """
    This function computes a dense depth map given a lidar pointcloud projected to the camera.
    :param pts: <np.ndarray: 3, n> Lidar point-cloud in image coordinates.
    :param depths: <np.ndarray: n, 1> Depth of the points.
    :param im_size: The image width and height.
    :param mode: How to render the depth, either sparse or dense.
    :param scale: The scaling factor applied to the depth map.
    :param n_dilate: Dilation filter size.
    :param n_gauss: Gaussian filter size.
    :param sigma_gauss: Gaussian filter sigma.
    :return The dense depth map.
    """
    # Store the minimum depth in the corresponding pixels
    # Apply downsampling to make it more efficient
    assert mode in ['sparse', 'dense']
    pxs = (pts[0, :] * scale).astype(np.int32)
    pys = (pts[1, :] * scale).astype(np.int32)

    depth_map_size = (np.array(im_size)[::-1] * scale).astype(np.int32)
    depth_map = np.zeros(depth_map_size, dtype=np.float32)
    for x, y, depth in zip(pxs, pys, depths):
        if depth_map[y][x] == 0:
            depth_map[y][x] = depth
        else:
            depth_map[y][x] = min(depth_map[y][x], depth)

    # Set invalid pixels to max_depth
    invalid = depth_map == 0
    depth_map[invalid] = depth_map.max()

    if mode == 'dense':
        # Perform erosion to grow points
        depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_ERODE, np.ones((n_dilate, n_dilate), np.uint8))

        # Perform Gaussian blur to smoothen points
        # Note that this should be used in moderation as the Gaussian filter also uses invalid depth values.
        blurred = cv2.GaussianBlur(depth_map, (n_gauss, n_gauss), sigma_gauss)
        valid = depth_map > 0
        depth_map[valid] = blurred[valid]

    return depth_map


def distort_pointcloud(pc: np.ndarray, camera_distortion: np.ndarray, cam: str, r_sq_max: float = 1) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Distort the point-cloud coordinates to map into the image.
    Note: This function discards some invalid points, that do not project into the image.
          This happens if the radial distortion function is not injective, which is the case if k3 is negative.
          We also use the same mechanism to avoid float overflows in the k4 portion of CAM_B0.
    :param r_sq_max: Hand-tuned parameter to address warping when distortion coeff, k3, < 0.
    :param cam: Name of the camera.
    :param pc: Lidar point-cloud.
    :param camera_distortion: Distortion coefficents of the camera.
    :return: Distorted point-cloud and depth values.
    """
    k1 = camera_distortion[0]
    k2 = camera_distortion[1]
    p1 = camera_distortion[2]
    p2 = camera_distortion[3]
    k3 = camera_distortion[4]

    # Store depth to return it
    depths = pc[2, :]

    # Normalize
    points_x = pc[0, :] / pc[2, :]
    points_y = pc[1, :] / pc[2, :]
    r_sq = points_x ** 2 + points_y ** 2

    # Filter points from outside the frustum that are likely to map inside it.
    # This happens when the distortion function is not injective, i.e. when k3 < 0 for all cameras,
    # apart from CAM_B0 which has distortion coefficient k6, which prevents warping.
    # However, we also do it for CAM_B0 to avoid overflows when computing r_sq ** 4.
    if k3 < 0 or cam == 'CAM_B0':
        if cam == 'CAM_B0':
            r_sq_max = 10
        mask = r_sq < r_sq_max
        depths = depths[mask]
        points_x = points_x[mask]
        points_y = points_y[mask]
        r_sq = r_sq[mask]

    radial_distort = 1 + k1 * r_sq + k2 * r_sq ** 2 + k3 * r_sq ** 3

    if cam == 'CAM_B0':  # fish-eye
        k4 = camera_distortion[5]
        radial_distort = radial_distort + k4 * r_sq ** 4
        assert not np.any(np.isinf(radial_distort)) and not np.any(np.isnan(radial_distort))

    x = radial_distort * points_x + 2 * p1 * points_x * points_y + p2 * (r_sq + 2 * points_x ** 2)
    y = radial_distort * points_y + p1 * (r_sq + 2 * points_y ** 2) + 2 * p2 * points_x * points_y

    # Define output
    points = np.ones((3, len(points_x)))
    points[0, :] = x
    points[1, :] = y

    return points, depths


class InvertedNormalize(Normalize):
    # A custom inverted colormap that stretches the close depth values out to have more color resolution.
    def __call__(self, value, clip=None):
        x = self.vmin + np.array([0, 0.2, 0.5, 1]) * (self.vmax - self.vmin)
        y = [0, 0.5, 0.95, 1]
        colors = np.interp(value, x, y)
        return 1 - np.ma.masked_array(colors)
