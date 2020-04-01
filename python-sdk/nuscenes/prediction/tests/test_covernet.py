# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

import math
import unittest

import torch
from torch.nn.functional import cross_entropy

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.covernet import mean_pointwise_l2_distance, ConstantLatticeLoss, CoverNet


class TestCoverNet(unittest.TestCase):

    def test_shapes_in_forward_pass_correct(self):
        resnet = ResNetBackbone('resnet50')

        covernet = CoverNet(resnet, 5, n_hidden_layers=[4096], input_shape=(3, 100, 100))

        image = torch.zeros(4, 3, 100, 100)
        asv = torch.empty(4, 3).random_(12)

        logits = covernet(image, asv)
        self.assertTupleEqual(logits.shape, (4, 5))


class TestConstantLatticeLoss(unittest.TestCase):

    def test_l1_distance(self):

        lattice = torch.zeros(3, 6, 2)
        lattice[0] = torch.arange(1, 13).reshape(6, 2)
        lattice[1] = torch.arange(1, 13).reshape(6, 2) * 3
        lattice[2] = torch.arange(1, 13).reshape(6, 2) * 6

        # Should select the first mode
        ground_truth = torch.arange(1, 13).reshape(6, 2).unsqueeze(0) + 2
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 0)

        # Should select the second mode
        ground_truth = torch.arange(1, 13).reshape(6, 2).unsqueeze(0) * 3 + 4
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 1)

        # Should select the third mode
        ground_truth = torch.arange(1, 13).reshape(6, 2).unsqueeze(0) * 6 + 10
        self.assertEqual(mean_pointwise_l2_distance(lattice, ground_truth), 2)

    def test_constant_lattice_loss(self):


        def generate_trajectory(theta: float) -> torch.Tensor:
            trajectory = torch.zeros(6, 2)
            trajectory[:, 0] = torch.arange(6) * math.cos(theta)
            trajectory[:, 1] = torch.arange(6) * math.sin(theta)
            return trajectory

        lattice = torch.zeros(3, 6, 2)
        lattice[0] = generate_trajectory(math.pi / 2)
        lattice[1] = generate_trajectory(math.pi / 4)
        lattice[2] = generate_trajectory(3 * math.pi / 4)

        ground_truth = torch.zeros(5, 1, 6, 2)
        ground_truth[0, 0] = generate_trajectory(0.2)
        ground_truth[1, 0] = generate_trajectory(math.pi / 3)
        ground_truth[2, 0] = generate_trajectory(5 * math.pi / 6)
        ground_truth[3, 0] = generate_trajectory(6 * math.pi / 11)
        ground_truth[4, 0] = generate_trajectory(4 * math.pi / 9)

        logits = torch.Tensor([[2, 10, 5],
                               [-3, 4, 5],
                               [-4, 2, 7],
                               [8, -2, 3],
                               [10, 3, 6]])

        answer = cross_entropy(logits, torch.LongTensor([1, 1, 2, 0, 0]))

        loss = ConstantLatticeLoss(lattice, mean_pointwise_l2_distance)
        loss_value = loss(logits, ground_truth)

        self.assertAlmostEqual(float(loss_value.detach().numpy()), float(answer.detach().numpy()))
