# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

"""
Regression test to see if CoverNet implementation can overfit on a single example.
"""

import argparse
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from nuscenes.prediction.models.backbone import MobileNetBackbone
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss


def generate_trajectory(theta: float) -> torch.Tensor:
    trajectory = torch.zeros(6, 2)
    trajectory[:, 0] = torch.arange(6) * math.cos(theta)
    trajectory[:, 1] = torch.arange(6) * math.sin(theta)
    return trajectory


class Dataset(IterableDataset):
    """ Implements an infinite dataset of the same input image, agent state vector and ground truth label. """

    def __iter__(self,):

        while True:
            image = torch.zeros((3, 100, 100))
            agent_state_vector = torch.ones(3)
            ground_truth = generate_trajectory(math.pi / 2)

            yield image, agent_state_vector, ground_truth.unsqueeze(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run CoverNet to make sure it overfits on a single test case.')
    parser.add_argument('--use_gpu', type=int, help='Whether to use gpu', default=0)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    backbone = MobileNetBackbone('mobilenet_v2')
    model = CoverNet(backbone, num_modes=3, input_shape=(3, 100, 100))
    model = model.to(device)

    lattice = torch.zeros(3, 6, 2)
    lattice[0] = generate_trajectory(math.pi / 2)
    lattice[1] = generate_trajectory(math.pi / 4)
    lattice[2] = generate_trajectory(3 * math.pi / 4)

    loss_function = ConstantLatticeLoss(lattice)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    n_iter = 0

    minimum_loss = 0

    for img, agent_state_vector, ground_truth in dataloader:

        img = img.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth = ground_truth.to(device)

        optimizer.zero_grad()

        logits = model(img, agent_state_vector)
        loss = loss_function(logits, ground_truth)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.2f}")
        if np.allclose(current_loss, minimum_loss, atol=1e-2):
            print(f"Achieved near-zero loss after {n_iter} iterations.")
            break

        n_iter += 1

