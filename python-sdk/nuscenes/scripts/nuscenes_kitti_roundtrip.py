"""
This script converts nuScenes data to KITTI format and vice versa.
"""
import os
import json
from typing import List, Tuple

import numpy as np
from pyquaternion import Quaternion
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.scripts.kitti import KittiDB
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_logs


def nuscenes_roundtrip(nusc: NuScenes, splits: Tuple[str, ...], kitti_fake_dir: str) -> None:
    """
    Check that boxes can be converted from nuScenes to KITTI and back.
    :param nusc: NuScenes instance.
    :param splits: The list of relevant splits (e.g. train, val).
    :param kitti_fake_dir: Where to write the KITTI-style annotations.
    """
    # Convert to KITTI files.
    nuscenes_to_kitti_file(nusc=nusc, splits=splits, kitti_fake_dir=kitti_fake_dir)

    # Check if the input can be reconstructed.
    kitti_file_to_nuscenes_check(nusc=nusc, splits=splits, kitti_fake_dir=kitti_fake_dir)

    print('Passed nuScenes roundtrip check!')


def kitti_roundtrip(kitti_dir: str) -> None:
    """
    Check that boxes can be converted from KITTI to nuscenes and back.
    :param kitti_dir: Original KITTI folder.
    """
    kitti = KittiDB(root=kitti_dir)
    tokens = kitti.tokens[:IMAGE_COUNT]
    for forward_itt, token in enumerate(tokens):
        print('Processing token %d of %d: %s' % (forward_itt, len(tokens), token))

        # Get KITTI boxes in nuScenes format.
        boxes = kitti.get_boxes(token=token)
        transforms = kitti.get_transforms(token=token, root=kitti_dir)
        velo_to_cam_rot = Quaternion(matrix=transforms['velo_to_cam']['R'])
        velo_to_cam_trans = np.array(transforms['velo_to_cam']['T'])
        r0_rect = Quaternion(matrix=transforms['r0_rect'])

        output_gts = []
        with open(KittiDB.get_filepath(token, 'label', root=kitti_dir), 'r') as f:
            for line in f:
                line = line.strip()
                line_tokens = line.split(' ')
                if line_tokens[0] not in ['DontCare', 'Misc']:  # Filter irrelevant classes.
                    output_gts.append(line)

        # Convert back to KITTI and check equivalence.
        for backward_itt, box_lidar_nut in enumerate(boxes):
            box_cam_kitti = KittiDB.box_nuscenes_to_kitti(box_lidar_nut, velo_to_cam_rot, velo_to_cam_trans, r0_rect)
            output_est = KittiDB.box_to_string(name=box_lidar_nut.name, box=box_cam_kitti)
            output_gt = output_gts[backward_itt]
            line_est = KittiDB._parse_label_line(output_est)
            line_gt = KittiDB._parse_label_line(output_gt)

            for field in ['name', 'xyz_camera', 'wlh', 'yaw_camera']:
                assert line_est[field] == line_gt[field]

    print('Passed KITTI roundtrip check!')


def kitti_file_to_nuscenes_check(nusc: NuScenes, splits: Tuple[str, ...], kitti_fake_dir: str) -> None:
    """
    Check whether a generated KITTI file has the same content as the original annotations.
    :param nusc: A NuScenes object.
    :param splits: The list of relevant splits (e.g. train, val).
    :param kitti_fake_dir: Where to write the KITTI-style annotations.
    """
    kitti = KittiDB(root=kitti_fake_dir)

    for split in splits:
        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, nusc)

        # Use only the samples from the current split.
        sample_tokens = split_to_samples(nusc, split_logs)
        sample_tokens = sample_tokens[:IMAGE_COUNT]

        for sample_token in sample_tokens:

            # Get sample data.
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data'][LIDAR_TOP_NAME]
            sample_annotation_tokens = sample['anns']

            # Retrieve the token from the lidar.
            sd_record_lid = nusc.get('sample_data', lidar_token)
            filename_lid_full = sd_record_lid['filename']
            token = os.path.basename(filename_lid_full).replace('.pcd.bin', '').replace('__%s__' % LIDAR_TOP_NAME, '-')

            boxes_gt = []
            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

                # Type: Only use classes from KITTI that are used in competitions.
                category = sample_annotation['category_name']
                if category.startswith('human.pedestrian.'):
                    kitti_name = 'Pedestrian'
                elif category == 'vehicle.car':
                    kitti_name = 'Car'
                elif category == 'vehicle.bicycle':
                    kitti_name = 'Cyclist'
                else:
                    # Ignore other categories.
                    continue

                # Get box in LIDAR frame.
                _, box_gt, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                    selected_anntokens=[sample_annotation_token])
                box_gt = box_gt[0]
                box_gt.name = kitti_name
                boxes_gt.append(box_gt)

            # Get the KITTI boxes we just generated in LIDAR frame.
            kitti_token = '%s_%s' % (split, token)
            boxes_verify = kitti.get_boxes(token=kitti_token)

            # Check both are equal.
            assert len(boxes_gt) == len(boxes_verify)
            for box_gt, box_verify in zip(boxes_gt, boxes_verify):
                assert ((box_gt.center - box_verify.center) ** 2).sum() < 1e-4
                assert np.sum((box_gt.orientation.rotation_matrix - box_verify.orientation.rotation_matrix) ** 2) < 0.04


def nuscenes_to_kitti_file(nusc: NuScenes, splits: Tuple[str, ...], kitti_fake_dir: str) -> None:
    """
    Convert nuScenes GT annotations to KITTI format.

    This script is used for compatibility with software that uses KITTI-style annotations. The resulting files should be
    treated with care, as KITTI has a single camera, whereas nuScenes has 6 cameras. Furthermore fields like occluded
    and truncated cannot be reproduced from nuScenes data.

    :param nusc: A NuScenes object.
    :param splits: The list of relevant splits (e.g. train, val).
    :param kitti_fake_dir: Where to write the KITTI-style annotations.

    To compare the calibration files use:
    t1 = KittiDB().get_transforms(token='train_000000', root=kitti_dir)
    t2 = KittiDB().get_transforms(token='train_n008-2018-05-21-11-06-59-0400-1526915243047392', root=kitti_fake_dir)

    velo_to_cam_trans should be:
    -    KITTI: [-0.00, -0.07, -0.27] (0 right, 7 down, 27 forward)
    - nuScenes: [ 0.01, -0.32, -0.75] (1 right, 29-32 down, 65-75 forward)
    """

    # visibility_map = {  # From nuScenes to KITTI.
    #     0: 2,  # largely occluded
    #     1: 2,  # largely occluded
    #     2: 1,  # partly occluded
    #     3: 1,  # partly occluded
    #     4: 0   # fully visible
    # }
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    kitti_to_nu_lidar_inv = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse
    image_size = [1600, 900]

    image_sizes = dict()  # Includes images from all splits.
    token_idx = 0  # Start tokens from 0.
    for split in splits:
        # Get assignment of scenes to splits.
        split_logs = create_splits_logs(split, nusc)

        # Create output folders.
        label_folder = os.path.join(kitti_fake_dir, split, 'label')
        calib_folder = os.path.join(kitti_fake_dir, split, 'calib')
        image_folder = os.path.join(kitti_fake_dir, split, 'image')
        lidar_folder = os.path.join(kitti_fake_dir, split, 'velodyne')
        for folder in [label_folder, calib_folder, image_folder, lidar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        # Use only the samples from the current split.
        sample_tokens = split_to_samples(nusc, split_logs)
        sample_tokens = sample_tokens[:IMAGE_COUNT]

        tokens = []
        for sample_token in sample_tokens:

            # Get sample data.
            sample = nusc.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][CAM_FRONT_NAME]
            lidar_token = sample['data'][LIDAR_TOP_NAME]

            # Retrieve sensor records.
            sd_record_cam = nusc.get('sample_data', cam_front_token)
            sd_record_lid = nusc.get('sample_data', lidar_token)
            cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            token = os.path.basename(filename_lid_full).replace('.pcd.bin', '').replace('__%s__' % LIDAR_TOP_NAME, '-')
            # token = '%06d' % token_idx # Alternative to use KITTI names.
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, token + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, token + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            scan = np.fromfile(src_lid_path, dtype=np.float32)
            points = scan.reshape((-1, 5))[:, :4].T
            pcl = LidarPointCloud(points)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # in KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to image_sizes.
            image_sizes[token] = image_size

            # Add to tokens.
            tokens.append(token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
            kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, token + '.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))

            # Write label file.
            label_path = os.path.join(label_folder, token + '.txt')
            if os.path.exists(label_path):
                print('Skipping existing file: %s' % label_path)
                continue
            else:
                print('Writing file: %s' % label_path)
            with open(label_path, "w") as label_file:

                for sample_annotation_token in sample_annotation_tokens:
                    sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

                    # Type: Only use classes from KITTI that are used in competitions.
                    category = sample_annotation['category_name']
                    if category.startswith('human.pedestrian.'):
                        kitti_name = 'Pedestrian'
                    elif category == 'vehicle.car':
                        kitti_name = 'Car'
                    elif category == 'vehicle.bicycle':
                        kitti_name = 'Cyclist'
                    else:
                        # Ignore other categories.
                        continue

                    # Get box in LIDAR frame.
                    _, box_lidar_nut, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                               selected_anntokens=[sample_annotation_token])
                    box_lidar_nut = box_lidar_nut[0]

                    # Truncated: Set all objects to 0 which means untruncated.
                    truncated = 0.0

                    # Occluded: Map from percent (0-20, 20-40, 40-60, 60-80, 80-100 to textually defined states).
                    occluded = 0  # Hard-coded: Full visibility.
                    # Alternatively use: visibility_map[int(sample_annotation['visibility_token'])]

                    # Convert to KITTI 3d and 2d box and KITTI output format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(box_lidar_nut, Quaternion(matrix=velo_to_cam_rot),
                                                                  velo_to_cam_trans, r0_rect)
                    output = KittiDB.box_to_string(name=kitti_name, box=box_cam_kitti, truncation=truncated,
                                                   occlusion=occluded)

                    # Write to stdout and disk.
                    print(output)
                    label_file.write(output + '\n')

        # Write tokens.txt for each split.
        tok_path = os.path.join(kitti_fake_dir, split, 'tokens.txt')
        with open(tok_path, "w") as tok_file:
            for tok in tokens:
                tok_file.write(tok + '\n')

    # Write image_sizes.json for all splits.
    image_sizes_path = os.path.join(kitti_fake_dir, 'image_sizes.json')
    with open(image_sizes_path, 'w') as image_sizes_file:
        json.dump(image_sizes, image_sizes_file)


def split_to_samples(nusc: NuScenes, split_logs: List[str]):
    """
    Convenience function to get the samples in a particular split.
    :param nusc: NuScenes instance.
    :param split_logs: A list of the log names in this split.
    :return: The list of samples.
    """
    samples = []
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            samples.append(sample['token'])
    return samples


if __name__ == '__main__':

    # Settings.
    _kitti_dir = '/data/sets/kitti'
    _kitti_fake_dir = os.path.expanduser('~/kitti_fake_splits')
    IMAGE_COUNT = 10
    is_mini = False
    CAM_FRONT_NAME = 'CAM_FRONT'
    LIDAR_TOP_NAME = 'LIDAR_TOP'

    # Select subset of the data to look at.
    if is_mini:
        _nusc = NuScenes(version='v1.0-mini')
        _splits = ('mini_train', 'mini_val')
    else:
        _nusc = NuScenes(version='v1.0-trainval')
        _splits = ('train', 'val')

    # nuScenes roundtrip.
    nuscenes_roundtrip(_nusc, _splits, _kitti_fake_dir)

    # KITTI roundtrip.
    kitti_roundtrip(_kitti_dir)
