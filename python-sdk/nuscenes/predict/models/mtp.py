# nuScenes dev-kit.
# Code written by Freddy Boulton, Elena Corina Grigore 2020.

"""
Implementation of Multiple-Trajectory Prediction (MTP) model,
based on  https://arxiv.org/pdf/1809.10732.pdf.
"""

import math
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f
from torchvision.models import (mobilenet_v2, resnet18, resnet34, resnet50,
                                resnet101, resnet152)

from nuscenes.eval.predict.config import PredictionConfig


def trim_network_at_index(network: nn.Module, index: int) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim
    :param index: Where to trim the network. Counted from the last layer.
    """
    return nn.Sequential(*list(network.children())[:-1])

RESNET_VERSION_TO_MODEL = {'resnet18': resnet18, 'resnet34': resnet34,
                           'resnet50': resnet50, 'resnet101': resnet101,
                           'resnet152': resnet152}

class ResNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: resnet18, resnet34, resnet50, resnet101, resnet152.
    """

    def __init__(self, version: str):
        """
        Inits ResNetBackbone
        :param vesion: resnet version to use.
        """
        super().__init__()

        if version not in RESNET_VERSION_TO_MODEL:
            raise ValueError(f'Parameter version must be one of {list(RESNET_VERSION_TO_MODEL.keys())}'
                             f'. Received {version}.')

        self.backbone = trim_network_at_index(RESNET_VERSION_TO_MODEL[version](), -1)

    def forward(self, input_tensor):
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width]
        :return: Tensor of shape [batch_size, n_convolution_filters]. For resnet50,
            the shape is [batch_size, 2048]
        """
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)

class MobileNetBackbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.

    Allowed versions: mobilenet_v2.
    """

    def __init__(self, version: str):
        """
        Inits MobileNetBackbone
        :param version: mobilenet version to use.
        """
        super().__init__()

        if version != 'mobilenet_v2':
            raise NotImplementedError(f'Only mobilenet_v2 has been implemented. Received {version}.')

        self.backbone = trim_network_at_index(mobilenet_v2(), -1)

    def forward(self, input_tensor):
        """
        Outputs features after last convolution.
        :param input_tensor:  Shape [batch_size, n_channels, length, width]
        :return: Tensor of shape [batch_size, n_convolution_filters]. For mobilenet_v2,
            the shape is [batch_size, 1280]
        """
        backbone_features = self.backbone(input_tensor)
        return backbone_features.mean([2, 3])


class MTP(nn.Module):
    """Implements the MTP network."""

    def __init__(self, backbone: nn.Module, num_modes: int,
                 seconds: int = 6, frequency_in_hz: int = 2,
                 n_hidden_layers: int = 4096, input_shape: Tuple[int, int, int] = (3, 500, 500)):
        """
        Inits the MTP network.
        :param backbone: CNN Backbone to use.
        :param num_modes: Number of predicted paths to estimate for each agent.
        :param seconds: Number of seconds into the future to predict.
            Default for the challenge is 6.
        :param frequency: Frequency between timesteps in the prediction (in HZ).
            Highest frequency is nuScenes is 2 hz.
        :param n_hidden_layers: Size of fully connected layer after the CNN
            backbone processes the image.
        :param input_shape: Shape of the input expected by the network.
            This is needed because the size of the fully connected layer after
            the backbone depends on the backbone and its version.
        """

        super().__init__()

        self.backbone = backbone
        self.num_modes = num_modes
        backbone_feature_dim = self._calculate_backbone_feature_dim(input_shape)
        self.fc1 = nn.Linear(backbone_feature_dim + 3, n_hidden_layers)
        predictions_per_mode = seconds * frequency_in_hz * 2

        self.fc2 = nn.Linear(n_hidden_layers, int(num_modes * predictions_per_mode + num_modes))

    def _calculate_backbone_feature_dim(self, input_shape: Tuple[int, int, int]):
        """Helper to calculate the shape of the fully-connected regression layer."""
        tensor = torch.ones(1, *input_shape)
        output_feat = self.backbone.forward(tensor)
        return output_feat.shape[-1]

    def forward(self, image_tensor, agent_state_vector):
        """
        Forward pass of the model.
        """

        backbone_features = self.backbone(image_tensor)

        features = torch.cat([backbone_features, agent_state_vector], axis=1)

        predictions = self.fc2(self.fc1(features))

        # Normalize the probabilities to sum to 1 for inference
        mode_probabilities = predictions[:, -self.num_modes:].clone()
        if self.training is False:
            mode_probabilities = f.softmax(mode_probabilities, dim=-1)

        predictions = predictions[:, :-self.num_modes]

        return torch.cat((predictions, mode_probabilities), 1)


class MTPLoss:
    """
    Computes the loss for the MTP model.
    """

    def __init__(self, num_modes, regression_loss_weight,
                 angle_threshold_degrees: float = 5.):
        """
        Inits MTP loss
        :param num_modes: How many modes are being predicted for each agent.
        :param regression_loss_weight: Coefficient applied to the regression loss to
            balance classification and regression performance.
        :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
            and the ground to consider it a match.
        """
        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2 # We predict x, y coordinates at each timestep
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    def _get_trajectory_and_modes(self,
                                  model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes]
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes]
        """
        mode_probabilities = model_prediction[:, -self.num_modes:].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities

    def _angle_between(self, ref_traj: torch.Tensor,
                       traj_to_compare: torch.Tensor) -> float:
        """
        Computes the angle between the last points of the two trajectories.
        The resulting angle is in degrees and is an angle in the [0; 180) interval.
        :param ref_traj: Tensor of shape [n_timesteps, 2]
        :param traj_to_compare: Tensor of shape [n_timesteps, 2]
        """

        EPSILON = 1e-5

        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
            ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')

        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON

        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

        # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
        if math.isclose(traj_norms_product, 0):
            return 0.

        # We apply the max and min operations below to ensure there is no value
        # returned for cos_angle that is greater than 1 or less than -1.
        # This should never be the case, but the check is in place for cases where
        # we might encounter numerical instability.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

        if angle >= 180:
            return angle - EPSILON

        return angle

    def _compute_ave_l2_norms(self, tensor: torch.Tensor) -> float:
        """
        Compute the average of l2 norms of each row in the tensor
        :param tensor: Shape [1, n_timesteps, 2]
        """
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance

    def _compute_angles_from_ground_truth(self, target, trajectories) -> List[float]:
        """
        Compute angle between the target trajectory (ground truth) and
        the predicted trajectories.
        :param target: Shape [1, n_timesteps, 2]
        :param trajectories: Shape [n_modes, n_timesteps, 2
        :return: List of angles.
        """
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            # For each mode, we compute the angle between the last point of the predicted trajectory for that
            # mode and the last point of the ground truth trajectory.
            angle = self._angle_between(target[0], mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self, angles_from_ground_truth, target, trajectories):
        """
        Finds the index of the best mode given the angles from the ground truth.
        :param angles_from_ground_truth: List of angles
        :param target: Shape [1, n_timesteps, 2]
        :param trajectories: Shape [n_modes, n_timesteps, 2]
        """

        # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
        # the index corresponding to the biggest angle that is still smaller than a threshold value.
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        # We choose the best mode at random IF there are no modes with an angle less than the threshold.
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)

        # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
        # predicted trajectory and the ground truth, taking into account only the modes with an angle
        # less than the threshold IF there is at least one mode with an angle less than the threshold.
        else:
            # Out of the selected modes above, we choose the final best mode as that which returns the
            # smallest ave of l2 norms between the predicted and ground truth trajectories.
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :]).item()

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
        trajectories, modes = self._get_trajectory_and_modes(predictions)

        for batch_idx in range(predictions.shape[0]):

            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])

            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

            regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss