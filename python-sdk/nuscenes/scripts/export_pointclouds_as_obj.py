# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

"""
Export fused point clouds of a scene to a Wavefront OBJ file.
This point-cloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
"""

import argparse
import os
import os.path as osp
from typing import Tuple

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points


def export_scene_pointcloud(nusc: NuScenes,
                            out_path: str,
                            scene_token: str,
                            channel: str = 'LIDAR_TOP',
                            min_dist: float = 3.0,
                            max_dist: float = 30.0,
                            verbose: bool = True) -> None:
    """
    Export fused point clouds of a scene to a Wavefront OBJ file.
    This point-cloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
    :param nusc: NuScenes instance.
    :param out_path: Output path to write the point-cloud to.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param min_dist: Minimum distance to ego vehicle below which points are dropped.
    :param max_dist: Maximum distance to ego vehicle above which points are dropped.
    :param verbose: Whether to print messages to stdout.
    """

    # Check inputs.
    valid_channels = ['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT',
                      'RADAR_BACK_RIGHT']
    camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', start_sample_rec['data'][channel])

    # Make list of frames
    cur_sd_rec = sd_rec
    sd_tokens = []
    while cur_sd_rec['next'] != '':
        cur_sd_rec = nusc.get('sample_data', cur_sd_rec['next'])
        sd_tokens.append(cur_sd_rec['token'])

    # Write point-cloud.
    with open(out_path, 'w') as f:
        f.write("OBJ File:\n")

        for sd_token in tqdm(sd_tokens):
            if verbose:
                print('Processing {}'.format(sd_rec['filename']))
            sc_rec = nusc.get('sample_data', sd_token)
            sample_rec = nusc.get('sample', sc_rec['sample_token'])
            lidar_token = sd_rec['token']
            lidar_rec = nusc.get('sample_data', lidar_token)
            pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_rec['filename']))

            # Get point cloud colors.
            coloring = np.ones((3, pc.points.shape[1])) * -1
            for channel in camera_channels:
                camera_token = sample_rec['data'][channel]
                cam_coloring, cam_mask = pointcloud_color_from_image(nusc, lidar_token, camera_token)
                coloring[:, cam_mask] = cam_coloring

            # Points live in their own reference frame. So they need to be transformed via global to the image plane.
            # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))

            # Optional Filter by distance to remove the ego vehicle.
            dists_origin = np.sqrt(np.sum(pc.points[:3, :] ** 2, axis=0))
            keep = np.logical_and(min_dist <= dists_origin, dists_origin <= max_dist)
            pc.points = pc.points[:, keep]
            coloring = coloring[:, keep]
            if verbose:
                print('Distance filter: Keeping %d of %d points...' % (keep.sum(), len(keep)))

            # Second step: transform to the global frame.
            poserecord = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pc.translate(np.array(poserecord['translation']))

            # Write points to file
            for (v, c) in zip(pc.points.transpose(), coloring.transpose()):
                if (c == -1).any():
                    # Ignore points without a color.
                    pass
                else:
                    f.write("v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n"
                            .format(v=v, c=c/255.0))

            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])


def pointcloud_color_from_image(nusc: NuScenes,
                                pointsensor_token: str,
                                camera_token: str) -> Tuple[np.array, np.array]:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane, then retrieve the colors of the closest image pixels.
    :param nusc: NuScenes instance.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample data token.
    :return (coloring <np.float: 3, n>, mask <np.bool: m>). Returns the colors for n points that reproject into the
        image out of m total points. The mask indicates which points are selected.
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, pointsensor['filename']))
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 0)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]

    # Pick the colors of the points
    im_data = np.array(im)
    coloring = np.zeros(points.shape)
    for i, p in enumerate(points.transpose()):
        point = p[:2].round().astype(np.int32)
        coloring[:, i] = im_data[point[1], point[0], :]

    return coloring, mask


if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description='Export a scene in Wavefront point cloud format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene', default='scene-0061', type=str, help='Name of a scene, e.g. scene-0061')
    parser.add_argument('--out_dir', default='~/nuscenes-visualization/pointclouds', type=str, help='Output folder')
    parser.add_argument('--verbose', default=0, type=int, help='Whether to print outputs to stdout')

    args = parser.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    scene_name = args.scene
    verbose = bool(args.verbose)

    out_path = osp.join(out_dir, '%s.obj' % scene_name)
    if osp.exists(out_path):
        print('=> File {} already exists. Aborting.'.format(out_path))
        exit()
    else:
        print('=> Extracting scene {} to {}'.format(scene_name, out_path))

    # Create output folder
    if not out_dir == '' and not osp.isdir(out_dir):
        os.makedirs(out_dir)

    # Extract point-cloud for the specified scene
    nusc = NuScenes()
    scene_tokens = [s['token'] for s in nusc.scene if s['name'] == scene_name]
    assert len(scene_tokens) == 1, 'Error: Invalid scene %s' % scene_name

    export_scene_pointcloud(nusc, out_path, scene_tokens[0], channel='LIDAR_TOP', verbose=verbose)
