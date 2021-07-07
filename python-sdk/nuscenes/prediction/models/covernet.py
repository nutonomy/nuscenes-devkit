# nuScenes dev-kit.
# Code written by Freddy Boulton, Tung Phan 2020.
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim

#尝试是西夏

# Number of entries in Agent State Vector（应该是指考虑了agent的三类状态：速度、加速度、朝向）
ASV_DIM = 3

class CoverNet(nn.Module):
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    def __init__(self, backbone: nn.Module, num_modes: int,
                 n_hidden_layers: List[int] = None,
                 input_shape: Tuple[int, int, int] = (3, 500, 500)):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone（使用的CNN主干网络）
        :param num_modes: Number of modes in the lattice（考虑的模态数）
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]（隐藏层神经元个数）
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.（输入语义地图的形状）
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):#可以有单个或多个隐藏层，但需要保证格式是list
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]#默认值，参考论文

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)#提取backbone输出的维数
        n_hidden_layers = [backbone_feature_dim + ASV_DIM] + n_hidden_layers + [num_modes]#在隐藏层基础上加入输入层和输出层
        #输入层注意包含语义图和agent的三个状态

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]#建立全连接各层的列表

        self.head = nn.ModuleList(linear_layers)

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)#先用CNN处理语义地图获取特征，之后再与agent特征结合

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)#特征concat

        for linear in self.head:
            logits = linear(logits)#特征再经过全连接层以获取最终轨迹集的概率

        return logits#输出轨迹集的各项概率


def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.（计算lattice中最接近轨迹的索引）
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()


class ConstantLatticeLoss:
    """
    这个函数应该就是计算loss的
    Computes the loss for a constant lattice CoverNet model.（计算一个常数晶格CoverNet模型的损失）
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)#应该是选取了与gt最近的点对应轨迹后用其概率来计算交叉熵

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()
