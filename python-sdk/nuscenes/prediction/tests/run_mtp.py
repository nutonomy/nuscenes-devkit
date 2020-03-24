# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.

"""
Regression test to see if MTP can overfit on a single example.
"""

import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


class Dataset(IterableDataset):
    """
    Implements an infinite dataset where the input data
    is always the same and the target is a path going
    forward with 75% probability, and going backward
    with 25% probability.
    """

    def __init__(self, num_modes: int = 1):
        self.num_modes = num_modes

    def __iter__(self,):

        while True:
            image = torch.zeros((3, 100, 100))
            agent_state_vector = torch.ones(3)
            ground_truth = torch.ones((1, 12, 2))

            if self.num_modes == 1:
                going_forward = True
            else:
                going_forward = np.random.rand() > 0.25

            if going_forward:
                ground_truth[:, :, 1] = torch.arange(0, 6, step=0.5)
            else:
                ground_truth[:, :, 1] = -torch.arange(0, 6, step=0.5)

            yield image, agent_state_vector, ground_truth


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MTP to make sure it overfits on a single test case.')
    parser.add_argument('--num_modes', type=int, help='How many modes to learn.', default=1)
    parser.add_argument('--use_gpu', type=bool, help='Whether to use gpu', default=False)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    dataset = Dataset(args.num_modes)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)

    backbone = ResNetBackbone('resnet18')
    model = MTP(backbone, args.num_modes)
    model = model.to(device)

    loss_function = MTPLoss(args.num_modes, 1, 5)

    current_loss = 10000

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    n_iter = 0

    minimum_loss = 0

    if args.num_modes == 2:

        # We expect to see 75% going_forward and
        # 25% going backward. So the minimum
        # classification loss is expected to be
        # 0.56234

        minimum_loss += 0.56234

    for img, agent_state_vector, ground_truth in dataloader:

        img = img.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth = ground_truth.to(device)

        optimizer.zero_grad()

        prediction = model(img, agent_state_vector)
        loss = loss_function(prediction, ground_truth)
        loss.backward()
        optimizer.step()

        current_loss = loss.cpu().detach().numpy()

        print(f"Current loss is {current_loss:.4f}")
        if np.allclose(current_loss, minimum_loss, atol=1e-4):
            print(f"Achieved near-zero loss after {n_iter} iterations.")
            break

        n_iter += 1

