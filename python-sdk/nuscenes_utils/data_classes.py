# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.
# Licensed under the Creative Commons [see licence.txt]

from __future__ import annotations

import struct

import cv2
import numpy as np
from pyquaternion import Quaternion

from nuscenes_utils.geometry_utils import view_points


class PointCloud:

    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points

    @staticmethod
    def load_numpy_bin(file_name):
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: The path of the pointcloud file.
        :return: <np.float: 4, n>. Point cloud matrix (x, y, z, intensity).
        """
        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        return points.T

    @staticmethod
    def load_pcd_bin(file_name):
        """
        Loads RADAR data from a Point Cloud Data file to a list of lists (=points) and meta data.

        Example of the header fields:
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
        SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
        TYPE F F F I I F F F F F I I I I I I I I
        COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        WIDTH 125
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS 125
        DATA binary

        :param file_name: The path of the pointcloud file.
        :return: <np.float: 18, n>. Point cloud matrix.
        """
        meta = []
        with open(file_name, 'rb') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                meta.append(line)
                if line.startswith('DATA'):
                    break

            data_binary = f.read()

        # Get the header rows and check if they appear as expected.
        assert meta[0].startswith('#'), 'First line must be comment'
        assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
        sizes = meta[3].split(' ')[1:]
        types = meta[4].split(' ')[1:]
        counts = meta[5].split(' ')[1:]
        width = int(meta[6].split(' ')[1])
        height = int(meta[7].split(' ')[1])
        data = meta[10].split(' ')[1]
        feature_count = len(types)
        assert width > 0
        assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
        assert height == 1, 'Error: height != 0 not supported!'
        assert data == 'binary'

        # Lookup table for how to decode the binaries
        unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                         'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                         'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
        types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

        # Decode each point
        offset = 0
        point_count = width
        points = []
        for i in range(point_count):
            point = []
            for p in range(feature_count):
                start_p = offset
                end_p = start_p + int(sizes[p])
                assert end_p < len(data_binary)
                point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
                point.append(point_p)
                offset = end_p
            points.append(point)

        # A NaN in the first point indicates an empty pointcloud
        point = np.array(points[0])
        if np.any(np.isnan(point)):
            return np.zeros((feature_count, 0))

        # Convert to numpy matrix
        points = np.array(points).transpose()

        return points

    @classmethod
    def from_file(cls, file_name):
        """
        Instantiate from a .pcl, .pdc, .npy, or .bin file.
        :param file_name: <str>. Path of the pointcloud file on disk.
        :return: <PointCloud>.
        """

        if file_name.endswith('.bin'):
            points = cls.load_numpy_bin(file_name)
        elif file_name.endswith('.pcd'):
            points = cls.load_pcd_bin(file_name)
        else:
            raise ValueError('Unsupported filetype {}'.format(file_name))

        return cls(points)

    def nbr_points(self):
        """
        Returns the number of points.
        :return: <int>. Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio):
        """
        Sub-samples the pointcloud.
        :param ratio: <float>. Fraction to keep.
        :return: <None>.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius):
        """
        Removes point too close within a certain radius from origin.
        :param radius: <float>.
        :return: <None>.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix):
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        :return: <None>.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]

    def render_height(self, ax, view=np.eye(4), x_lim=(-20, 20), y_lim=(-20, 20), marker_size=1):
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: <matplotlib.axes.Axes>. Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min <float>, max <float>).
        :param y_lim: (min <float>, max <float>).
        :param marker_size: <float>. Marker size.
        :return: <None>.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self, ax, view=np.eye(4), x_lim=(-20, 20), y_lim=(-20, 20), marker_size=1):
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: <matplotlib.axes.Axes>. Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min <float>, max <float>).
        :param y_lim: (min <float>, max <float>).
        :param marker_size: <float>. Marker size.
        :return: <None>.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self, color_channel, ax, view, x_lim, y_lim, marker_size):
        """
        Helper function for rendering.
        :param color_channel: <int>.
        :param ax: <matplotlib.axes.Axes>. Axes on which to render the points.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min <float>, max <float>).
        :param y_lim: (min <float>, max <float>).
        :param marker_size: <float>. Marker size.
        :return: <None>.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self, center, size, orientation, label=np.nan, score=np.nan, velocity=(np.nan, np.nan, np.nan),
                 name=None):
        """
        :param center: [<float>: 3]. Center of box given as x, y, z.
        :param size: [<float>: 3]. Size of box in width, length, height.
        :param orientation: <Quaternion>. Box orientation.
        :param label: <int>. Integer label, optional.
        :param score: <float>. Classification score, optional.
        :param velocity: [<float>: 3]. Box velocity in x, y, z direction.
        :param name: <str>. Box name, optional. Can be used e.g. for denote category name.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name)

    def encode(self):
        """
        Encodes the box instance to a JSON-friendly vector representation.
        :return: [<float>: 16]. List of floats encoding the box.
        """
        return self.center.tolist() + self.wlh.tolist() + self.orientation.elements.tolist() + [
                self.label] + [self.score] + self.velocity.tolist() + [self.name]

    @classmethod
    def decode(cls, data):
        """
        Instantiates a Box instance from encoded vector representation.
        :param data: [<float>: 16]. Output from encode.
        :return: <Box>.
        """
        return Box(data[0:3], data[3:6], Quaternion(data[6:10]), label=data[10], score=data[11], velocity=data[12:15],
                   name=data[15])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: (3, 3)>.
        """
        return self.orientation.rotation_matrix

    def translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self):
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self, axis, view=np.eye(3), normalize=False, colors=('b', 'r', 'k'), linewidth=2):
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: <matplotlib.pyplot.axis>. Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: <bool>. Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: <float>. Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]], [corners.T[i][1], corners.T[i + 4][1]], color=colors[2],
                      linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self, im, view=np.eye(3), normalize=False,
                   colors=((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth=2):
        """
        Renders box using opencv2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: <bool>. Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: <float>. Linewidth for plot.
        :return:
        """

        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im, (int(prev[0]), int(prev[1])), (int(corner[0]), int(corner[1])), color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im, (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])), colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im, (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])), colors[0][::-1], linewidth)
