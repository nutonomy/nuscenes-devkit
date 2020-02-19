# nuScenes dev-kit.
# Code written by Freddy Boulton, Elena Corina Grigore 2020.

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as f
import numpy as np
from nuscenes.predict.models.backbone import calculate_backbone_feature_dim


class CoverNet(nn.Module):

    def __init__(self, backbone: nn.Module, num_modes: int,
                 n_hidden_layers: List[int] = None,  input_shape: Tuple[int, int, int] = (3, 500, 500)):

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [2051, 4096]

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + 3] + n_hidden_layers + [num_modes]

        linear_layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.head = nn.ModuleList(linear_layers)


    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:

        backbone_features = self.backbone(image_tensor)

        features = torch.cat([backbone_features, agent_state_vector], dim=1)

        logits = self.head(features)

        return logits


class CoverNetLoss:
    pass

