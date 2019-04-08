# nuScenes dev-kit.
# Code written by Sergi Adipraja Widjaja, 2019.
# Licensed under the Creative Commons [see license.txt]

"""
Export 2D annotations (xmin, ymin,xmax, ymax) from re-projections of our annotated 3D bounding boxes to a .json file.
"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import box_in_image

import numpy as np
import json
import argparse
import os

from typing import List
from pyquaternion.quaternion import Quaternion
from shapely.geometry import LineString, Point
from collections import OrderedDict
from tqdm import tqdm
from nuscenes.utils.geometry_utils import BoxVisibility


class Corner:
    """A data class implementing a 3d bounding box corner"""
    def __init__(self, point: Point, min_x: float = 0, max_x: float = 1600, min_y: float = 0, max_y: float = 900):
        """
        :param point: A shapely implementation of a point using.
        :param min_x: Minimum value of x in the image coordinate.
        :param max_x: Maximum value of x in the image coordinate.
        :param min_y: Minimum value of y in the image coordinate.
        :param max_y: Maixmum value of y in the image coordinate.
        """
        self.point = point
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.x = self.point.x
        self.y = self.point.y

    def is_inside_image(self) -> bool:
        """
        Returns `True` if the box is inside the image canvas, `False` otherwise.
        :return: Whether the box is inside the image.
        """
        return (self.min_x - 0.5 < self.x < self.max_x + 0.5) and (self.min_y - 0.5 < self.y < self.max_y + 0.5)


class Line:
    """A data class implementing a 3d bounding box line segment"""
    def __init__(self,
                 line_string: LineString,
                 min_x: float = 0,
                 max_x: float = 1600,
                 min_y: float = 0,
                 max_y: float = 900):
        """
        :param line_string: A shapely implementation of a line string.
        :param min_x: Minimum value of x in the image coordinate.
        :param max_x: Maximum value of x in the image coordinate.
        :param min_y: Minimum value of y in the image coordinate.
        :param max_y: Maximum value of y in the image coordinate.
        """
        self.line_string = line_string
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.left_boundary = LineString([(min_x, min_y), (min_x, max_y)])
        self.right_boundary = LineString([(max_x, min_y), (max_x, max_y)])
        self.top_boundary = LineString([(min_x, min_y), (max_x, min_y)])
        self.bottom_boundary = LineString([(min_x, max_y), (max_x, max_y)])

    def is_left_intersect(self) -> bool:
        """
        Returns True if the line segment intersects the left boundary of the image, False otherwise
        :return: Whether the line segment intersects the left boundary of the image
        """
        return self.line_string.intersects(self.left_boundary)

    def is_right_intersect(self) -> bool:
        """
        Returns True if the line segment intersects the right boundary of the image, False otherwise
        :return: Whether the line segment intersects the right boundary of the image
        """
        return self.line_string.intersects(self.right_boundary)

    def is_top_intersect(self) -> bool:
        """
        Returns True if the line segment intersects the top boundary of the image, False otherwise
        :return: Whether the line segment intersects the top boundary of the image
        """
        return self.line_string.intersects(self.top_boundary)

    def is_bottom_intersect(self) -> bool:
        """
        Returns True if the line segment intersects the top boundary of the image, False otherwise
        :return: Whether the line segment intersects the top boundary of the image
        """
        return self.line_string.intersects(self.bottom_boundary)

    def boundary_intersections(self) -> List[Corner]:
        """
        Retrieve all the intersections points with the image boundaries.
        :return: Intersection points with the image boundaries.
        """
        intersections = []
        if self.is_left_intersect():
            intersections.append(Corner(self.line_string.intersection(self.left_boundary)))
        if self.is_right_intersect():
            intersections.append(Corner(self.line_string.intersection(self.right_boundary)))
        if self.is_top_intersect():
            intersections.append(Corner(self.line_string.intersection(self.top_boundary)))
        if self.is_bottom_intersect():
            intersections.append(Corner(self.line_string.intersection(self.bottom_boundary)))

        return intersections


def generate_record(ann_rec: dict,
                    x1: float,
                    y1: float,
                    x2: float,
                    y2: float,
                    sample_data_token: str,
                    filename: str) -> OrderedDict:
    """
    Generate one 2D annotation record given various informations on top of the 2D bounding box coordinates.
    :param ann_rec: Original 3d annotation record.
    :param x1: Minimum value of the x coordinate.
    :param y1: Minimum value of the y coordinate.
    :param x2: Maximum value of the x coordinate.
    :param y2: Maximum value of the y coordinate.
    :param sample_data_token: Sample data tolk
    :param filename:
    :return: A sample 2D annotation record.
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    return repro_rec


def get_line_segments(corners: np.ndarray) -> List[Line]:
    """
    Get all the line segments that corresponds to the corners of the bounding box.
    :param corners: Corners of the bounding box.
    :return: Line segments of a particular bounding box.
    """

    lines = []
    front_corners = corners.T[4:]
    prev = front_corners[-1]
    for corner in front_corners:
        x1 = prev[0]
        y1 = prev[1]
        x2 = corner[0]
        y2 = corner[1]
        prev = corner

        lines.append(Line(LineString([(x1, y1), (x2, y2)])))

    rear_corners = corners.T[:4]
    prev = rear_corners[-1]
    for corner in rear_corners:
        x1 = prev[0]
        y1 = prev[1]
        x2 = corner[0]
        y2 = corner[1]
        prev = corner

        lines.append(Line(LineString([(x1, y1), (x2, y2)])))

    for i in range(4):
        x1 = corners.T[i][0]
        y1 = corners.T[i][1]
        x2 = corners.T[i + 4][0]
        y2 = corners.T[i + 4][1]

        lines.append(Line(LineString([(x1, y1), (x2, y2)])))

    return lines


def get_2d_boxes(sample_data_token: str) -> List[OrderedDict]:
    """
    Get the 2D annotation records for a given `sample_data_token`.
    :param sample_data_token: Sample data token belonging to a keyframe.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """

    # Get the sample data, and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)
    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation above a visibility threshold.
    ann_recs = [nusc.get('sample_annotation', token) for token in s_rec['anns']]
    ann_recs = [ann_rec for ann_rec in ann_recs if (ann_rec['visibility_token'] in args.visibilities)]

    repro_recs = []

    for ann_rec in ann_recs:

        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        box = nusc.get_box(ann_rec['token'])

        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        if not box_in_image(box, camera_intrinsic, (1600, 900), BoxVisibility.Any):
            continue

        corner_coords = view_points(box.corners(), camera_intrinsic, True)
        segments = get_line_segments(corner_coords)
        corners = [Corner(Point(corner_coord[0], corner_coord[1])) for corner_coord in corner_coords.T]

        for segment in segments:
            corners.extend(segment.boundary_intersections())

        corners = np.array([[corner.x, corner.y] for corner in corners if corner.is_inside_image()])

        max_x = max(corners[:, 0])
        min_x = min(corners[:, 0])
        max_y = max(corners[:, 1])
        min_y = min(corners[:, 1])

        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y, sample_data_token, sd_rec['filename'])
        repro_recs.append(repro_rec)

    return repro_recs


def main():
    if args.keyframes_only:
        sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if
                                     (s['sensor_modality'] == 'camera') and s['is_key_frame']]
    else:
        sample_data_camera_tokens = [s['token'] for s in nusc.sample_data if (s['sensor_modality'] == 'camera')]

    print("Generating 2d reprojections of the nuScenes dataset")

    reprojections = []
    for token in tqdm(sample_data_camera_tokens):
        reprojection_records = get_2d_boxes(token)
        reprojections.extend(reprojection_records)

    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    with open(os.path.join(args.dataroot, args.version, args.dest_filename), 'w') as fh:
        json.dump(reprojections, fh, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export 2D annotations from reprojections to a .json file.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--keyframes_only', type=int, default=True)
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes')
    parser.add_argument('--version', type=str, default='v1.0')
    parser.add_argument('--dest_filename', type=str, default='image_annotations.json')
    parser.add_argument('--visibilities', type=str, default=['2', '3', '4'])
    args = parser.parse_args()

    nusc = NuScenes(dataroot=args.dataroot, version=args.version)
    main()
