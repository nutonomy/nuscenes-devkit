# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We don't map to KITTI category names or back.
- Attributes are not part of KITTI and therefore not output in the nuScenes result format.

This script includes three main functions:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.
- render_kitti(): Render the annotations of the (generated or real) KITTI dataset.
- kitti_res_to_nuscenes(): Converts a KITTI detection result to the nuScenes detection results format.

To launch these scripts run:
- python export_kitti.py nuscenes_gt_to_kitti --nusc_kitti_dir ~/nusc_kitti
- python export_kitti.py render_kitti --nusc_kitti_dir ~/nusc_kitti
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_dir ~/nusc_kitti

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""
import os
import json
from typing import List, Tuple, Dict, Any

from pyquaternion import Quaternion
import numpy as np
import fire
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.kitti import KittiDB


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '~/nusc_kitti',
                 is_mini: bool = True,
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 10,
                 image_size: Tuple[int, int] = (1600, 900)):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param is_mini: Whether to use only the mini split

        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.is_mini = is_mini
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.image_count = image_count
        self.image_size = image_size

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        if self.is_mini:
            self.nusc = NuScenes(version='v1.0-mini')
            self.splits = ('mini_train', 'mini_val')
        else:
            self.nusc = NuScenes(version='v1.0-trainval')
            self.splits = ('train', 'val')

    def nuscenes_gt_to_kitti(self) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse

        image_sizes = dict()  # Includes images from all splits.
        token_idx = 0  # Start tokens from 0.
        for split in self.splits:
            # Get assignment of scenes to splits.
            split_logs = create_splits_logs(split, self.nusc)

            # Create output folders.
            label_folder = os.path.join(self.nusc_kitti_dir, split, 'label')
            calib_folder = os.path.join(self.nusc_kitti_dir, split, 'calib')
            image_folder = os.path.join(self.nusc_kitti_dir, split, 'image')
            lidar_folder = os.path.join(self.nusc_kitti_dir, split, 'velodyne')
            for folder in [label_folder, calib_folder, image_folder, lidar_folder]:
                if not os.path.isdir(folder):
                    os.makedirs(folder)

            # Use only the samples from the current split.
            sample_tokens = self._split_to_samples(split_logs)
            sample_tokens = sample_tokens[:self.image_count]

            tokens = []
            for sample_token in sample_tokens:

                # Get sample data.
                sample = self.nusc.get('sample', sample_token)
                sample_annotation_tokens = sample['anns']
                cam_front_token = sample['data'][self.cam_name]
                lidar_token = sample['data'][self.lidar_name]

                # Retrieve sensor records.
                sd_record_cam = self.nusc.get('sample_data', cam_front_token)
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                cs_record_cam = self.nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
                cs_record_lid = self.nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

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
                token = os.path.basename(filename_lid_full).replace('.pcd.bin', '') \
                    .replace('__%s__' % self.lidar_name, '-')
                # token = '%06d' % token_idx # Alternative to use KITTI names.
                token_idx += 1

                # Convert image (jpg to png).
                src_im_path = os.path.join(self.nusc.dataroot, filename_cam_full)
                dst_im_path = os.path.join(image_folder, token + '.png')
                if not os.path.exists(dst_im_path):
                    im = Image.open(src_im_path)
                    im.save(dst_im_path, "PNG")

                # Convert lidar.
                # Note that we are only using a single sweep, instead of the commonly used n sweeps.
                src_lid_path = os.path.join(self.nusc.dataroot, filename_lid_full)
                dst_lid_path = os.path.join(lidar_folder, token + '.bin')
                assert not dst_lid_path.endswith('.pcd.bin')
                scan = np.fromfile(src_lid_path, dtype=np.float32)
                points = scan.reshape((-1, 5))[:, :4].T
                pcl = LidarPointCloud(points)
                pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
                with open(dst_lid_path, "w") as lid_file:
                    pcl.points.T.tofile(lid_file)

                # Add to image_sizes.
                image_sizes[token] = self.image_size

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
                        sample_annotation = self.nusc.get('sample_annotation', sample_annotation_token)

                        # Get box in LIDAR frame.
                        _, box_lidar_nusc, _ = self.nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE,
                                                                         selected_anntokens=[sample_annotation_token])
                        box_lidar_nusc = box_lidar_nusc[0]

                        # Truncated: Set all objects to 0 which means untruncated.
                        truncated = 0.0

                        # Occluded: Hard-coded: Full visibility.
                        occluded = 0

                        # Convert to KITTI 3d and 2d box and KITTI output format.
                        box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                            box_lidar_nusc, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)
                        box_cam_kitti.score = 0 # Set dummy score so we can use this file as result.
                        category = sample_annotation['category_name']
                        output = KittiDB.box_to_string(name=category, box=box_cam_kitti, truncation=truncated,
                                                       occlusion=occluded)

                        # Write to disk.
                        label_file.write(output + '\n')

            # Write tokens.txt for each split.
            tok_path = os.path.join(self.nusc_kitti_dir, split, 'tokens.txt')
            with open(tok_path, "w") as tok_file:
                for tok in tokens:
                    tok_file.write(tok + '\n')

        # Write image_sizes.json for all splits.
        image_sizes_path = os.path.join(self.nusc_kitti_dir, 'image_sizes.json')
        with open(image_sizes_path, 'w') as image_sizes_file:
            json.dump(image_sizes, image_sizes_file, indent=2)

    def render_kitti(self) -> None:
        """
        Renders the annotations in the KITTI dataset from a lidar and a camera view.
        """
        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=self.splits)

        # Create output folder.
        render_dir = os.path.join(self.nusc_kitti_dir, 'render')
        if ~os.path.isdir(render_dir):
            os.makedirs(render_dir)

        # Render each image.
        for token in kitti.tokens:
            for sensor in ['lidar', 'camera']:
                out_path = os.path.join(render_dir, '%s_%s.png' % (token, sensor))
                kitti.render_sample_data(token, sensor_modality=sensor, out_path=out_path)

    def kitti_res_to_nuscenes(self) -> None:
        """
        Converts a KITTI detection result to the nuScenes detection results format.
        """
        # Dummy meta data, please adjust accordingly.
        meta = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        # Init.
        results = {}

        # Load the KITTI dataset.
        kitti = KittiDB(root=self.nusc_kitti_dir, splits=self.splits)

        for split in self.splits:
            # Get assignment of scenes to splits.
            split_logs = create_splits_logs(split, self.nusc)

            # Use only the samples from the current split.
            sample_tokens = self._split_to_samples(split_logs)
            sample_tokens = sample_tokens[:self.image_count]

            for sample_token in sample_tokens:
                # Get sample data.
                sample = self.nusc.get('sample', sample_token)
                lidar_token = sample['data'][self.lidar_name]

                # Retrieve the token from the lidar.
                sd_record_lid = self.nusc.get('sample_data', lidar_token)
                filename_lid_full = sd_record_lid['filename']
                token = os.path.basename(filename_lid_full) \
                    .replace('.pcd.bin', '') \
                    .replace('__%s__' % self.lidar_name, '-')

                # Get the KITTI boxes we just generated in LIDAR frame.
                kitti_token = '%s_%s' % (split, token)
                boxes = kitti.get_boxes(token=kitti_token)

                # Convert KITTI boxes to nuScenes detection challenge result format.
                sample_results = [self._box_to_sample_result(sample_token, box) for box in boxes]

                # Store all results for this image.
                results[sample_token] = sample_results

        # Store submission file to disk.
        submission = {
            'meta': meta,
            'results': results
        }
        submission_path = os.path.join(self.nusc_kitti_dir, 'submission.json')
        print('Writing submission to: %s' % submission_path)
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)

    def _box_to_sample_result(self, sample_token: str, box: Box, attribute_name: str = '') -> Dict[str, Any]:
        # Prepare data
        translation = box.center
        size = box.wlh
        rotation = box.orientation.q
        velocity = box.velocity
        detection_name = box.name
        detection_score = box.score

        # Create result dict
        sample_result = dict()
        sample_result['sample_token'] = sample_token
        sample_result['translation'] = translation.tolist()
        sample_result['size'] = size.tolist()
        sample_result['rotation'] = rotation.tolist()
        sample_result['velocity'] = velocity.tolist()[:2]  # Only need vx, vy.
        sample_result['detection_name'] = detection_name
        sample_result['detection_score'] = detection_score
        sample_result['attribute_name'] = attribute_name

        return sample_result

    def _split_to_samples(self, split_logs: List[str]) -> List[str]:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        samples = []
        for sample in self.nusc.sample:
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            logfile = log['logfile']
            if logfile in split_logs:
                samples.append(sample['token'])
        return samples


if __name__ == '__main__':
    fire.Fire(KittiConverter)
