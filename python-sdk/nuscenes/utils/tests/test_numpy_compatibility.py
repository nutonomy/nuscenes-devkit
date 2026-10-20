"""Output regressions for NumPy 2 scalar promotion and default integer changes."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
from PIL import Image

from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.lidarseg.utils import ConfusionMatrix
from nuscenes.eval.panoptic import get_panoptic_from_seg_det_or_track as panoptic
from nuscenes.eval.panoptic.panoptic_seg_evaluator import PanopticEval
from nuscenes.utils.map_mask import MapMask


class TestNumpyCompatibility(unittest.TestCase):

    def generate_labels(self, labels, boxes=()):
        """Use real point/label files and geometry; replace only dataset metadata loading."""
        with tempfile.TemporaryDirectory() as root:
            points = np.zeros((len(labels), 5), dtype=np.float32)
            points[:, 0] = np.arange(len(labels))
            points.tofile(os.path.join(root, 'points.bin'))
            label_dir = os.path.join(root, 'lidarseg', 'mini')
            os.makedirs(label_dir)
            np.array(labels, dtype=np.uint8).tofile(os.path.join(label_dir, 'lidar_lidarseg.bin'))

            identity = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}
            records = {
                ('sample', 'sample'): {'scene_token': 'scene', 'data': {'LIDAR_TOP': 'lidar'}},
                ('sample_data', 'lidar'): {
                    'token': 'lidar', 'filename': 'points.bin',
                    'calibrated_sensor_token': 'sensor', 'ego_pose_token': 'pose'
                },
                ('calibrated_sensor', 'sensor'): identity,
                ('ego_pose', 'pose'): identity
            }
            nusc = SimpleNamespace(
                dataroot=root, version='v1.0-mini',
                get=Mock(side_effect=lambda table, token: records[table, token])
            )
            predictions = {'sample': list(boxes)}
            with patch.object(panoptic, 'get_samples_in_eval_set', return_value=['sample']), \
                    patch.object(panoptic, 'PanopticClassMapper') as mapper, \
                    patch.object(panoptic, 'load_prediction', return_value=(predictions, {})), \
                    patch.object(panoptic, 'add_center_dist', return_value=predictions):
                mapper.return_value.get_coarse2idx.return_value = {'car': 4}
                panoptic.generate_panoptic_labels(nusc, root, 'unused.json', 'mini', out_dir=root)

            with np.load(os.path.join(root, 'panoptic', 'mini', 'lidar_panoptic.npz')) as saved:
                result = saved['data']
            self.assertEqual(result.dtype, np.dtype(np.uint16))
            self.assertEqual(result.shape, (len(labels),))
            return result

    def test_panoptic_stuff_uint8_multiplication(self):
        """Python integer 1000 must not be coerced into the uint8 input dtype."""
        result = self.generate_labels([0, 4, 11, 12, 13, 14, 15, 16])
        np.testing.assert_array_equal(result, [0, 0, 11000, 12000, 13000, 14000, 15000, 16000])

    def test_panoptic_mixed_thing_and_stuff(self):
        box = DetectionBox(
            translation=(0, 0, 0), size=(1, 1, 1), rotation=(1, 0, 0, 0),
            detection_name='car', detection_score=0.9
        )
        result = self.generate_labels([4, 4, 11, 16], [box])
        np.testing.assert_array_equal(result, [4001, 0, 11000, 16000])

    def test_panoptic_float32_overlap_threshold(self):
        """Exercise float32 count + Python epsilon at and above the overlap threshold."""
        first = DetectionBox(
            translation=(0.5, 0, 0), size=(1, 1.2, 1), rotation=(1, 0, 0, 0),
            detection_name='car', detection_score=0.9
        )
        for center, expected in [(1.5, [4001, 4001, 4002]), (0.5, [4001, 4001, 0])]:
            with self.subTest(center=center):
                second = DetectionBox(
                    translation=(center, 0, 0), size=(1, 1.2, 1), rotation=(1, 0, 0, 0),
                    detection_name='car', detection_score=0.8
                )
                np.testing.assert_array_equal(self.generate_labels([4, 4, 4], [first, second]), expected)

    def test_panoptic_empty_box_mask(self):
        """A zero float32 count plus epsilon must keep an empty box from producing NaNs."""
        box = DetectionBox(
            translation=(100, 0, 0), size=(1, 1, 1), rotation=(1, 0, 0, 0),
            detection_name='car', detection_score=0.9
        )
        with np.errstate(divide='raise', invalid='raise'):
            result = self.generate_labels([4, 11], [box])
        np.testing.assert_array_equal(result, [0, 11000])

    def test_panoptic_large_instance_offset(self):
        """The 2**32 Python offset must be applied in a sufficiently wide integer dtype."""
        evaluator = PanopticEval(n_classes=2, ignore=[0], min_points=1)
        semantics = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        predicted = np.array([[1000, 1000, 32000, 32000]], dtype=np.uint16)
        target = np.array([[2000, 2000, 31000, 31000]], dtype=np.uint16)
        evaluator.addBatch(semantics, predicted, semantics, target)
        np.testing.assert_array_equal(evaluator.pan_tp, [0, 2])
        np.testing.assert_array_equal(evaluator.pan_fp, [0, 0])
        np.testing.assert_array_equal(evaluator.pan_fn, [0, 0])
        np.testing.assert_allclose(evaluator.getPQ()[:3], [1, 1, 1], rtol=0, atol=0)

    def test_confusion_matrix_uint8_label_promotion(self):
        """Default int width must not let flattened class-pair indices wrap at 255."""
        evaluator = ConfusionMatrix(num_classes=32)
        evaluator.update(
            np.array([31, 31, 1, 1], dtype=np.uint8),
            np.array([31, 1, 1, 31], dtype=np.uint8)
        )
        expected = np.zeros((32, 32), dtype=np.int64)
        expected[31, 31] = expected[31, 1] = expected[1, 1] = expected[1, 31] = 1
        np.testing.assert_array_equal(evaluator.global_cm, expected)
        self.assertEqual(evaluator.global_cm.dtype.kind, 'i')
        np.testing.assert_allclose(
            evaluator.get_per_class_iou()[[1, 31]], [1 / 3, 1 / 3], rtol=1e-6, atol=0
        )

    def test_map_dilation_float32_scalar_promotion(self):
        """OpenCV float32 distances multiplied by Python/NumPy scalars must agree."""
        with tempfile.TemporaryDirectory() as root:
            filename = os.path.join(root, 'map.png')
            mask = np.zeros((5, 5), dtype=np.uint8)
            mask[2, 2] = 255
            Image.fromarray(mask).save(filename)
            expected = np.zeros_like(mask)
            expected[2, 2] = expected[1, 2] = expected[3, 2] = 255
            expected[2, 1] = expected[2, 3] = 255
            for resolution in (0.1, np.float32(0.1), np.float64(0.1)):
                with self.subTest(resolution_type=type(resolution).__name__):
                    # Isolate dilation from float32 rounding in the image resize dimensions.
                    with patch.object(MapMask, '_base_mask', new_callable=PropertyMock, return_value=mask):
                        result = MapMask(filename, resolution=resolution).mask(dilation=0.11)
                    self.assertEqual(result.dtype, np.dtype(np.uint8))
                    np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
