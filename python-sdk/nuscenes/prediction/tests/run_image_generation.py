import argparse
from typing import List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss


class TestDataset(Dataset):

    def __init__(self, tokens: List[str], helper: PredictHelper):
        self.tokens = tokens
        self.static_layer_representation = StaticLayerRasterizer(helper)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):

        token = self.tokens[index]
        instance_token, sample_token = token.split("_")

        image = self.static_layer_representation.make_representation(instance_token, sample_token)
        image = torch.Tensor(image).permute(2, 0, 1)
        agent_state_vector = torch.ones((3))
        ground_truth = torch.ones((1, 12, 2))

        ground_truth[:, :, 1] = torch.arange(0, 6, step=0.5)

        return image, agent_state_vector, ground_truth


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Makes sure image generation code can run on gpu "
                                                 "with multiple workers")
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--use_gpu', type=bool, help='Whether to use gpu', default=False)
    args = parser.parse_args()

    NUM_MODES = 1

    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    tokens = ['bd26c2cdb22d4bb1834e808c89128898_ca9a282c9e77460f8360f564131a8af5',
              '085fb7c411914888907f7198e998a951_ca9a282c9e77460f8360f564131a8af5',
              'bc38961ca0ac4b14ab90e547ba79fbb6_ca9a282c9e77460f8360f564131a8af5',
              '56a71c208ac6472f90b6a82529a6ce61_ca9a282c9e77460f8360f564131a8af5',
              '85246a44cc6340509e3882e2ff088391_ca9a282c9e77460f8360f564131a8af5',
              '42641eb6adcb4f8f8def8ef129d9e843_ca9a282c9e77460f8360f564131a8af5',
              '4080c30aa7104d91ad005a50b18f6108_ca9a282c9e77460f8360f564131a8af5',
              'c1958768d48640948f6053d04cffd35b_ca9a282c9e77460f8360f564131a8af5',
              '4005437c730645c2b628dc1da999e06a_39586f9d59004284a7114a68825e8eec',
              'a017fe4e9c3d445784aae034b1322006_356d81f38dd9473ba590f39e266f54e5',
              'a0049f95375044b8987fbcca8fda1e2b_c923fe08b2ff4e27975d2bf30934383b',
              '61dd7d03d7ad466d89f901ed64e2c0dd_e0845f5322254dafadbbed75aaa07969',
              '86ed8530809d4b1b8fbc53808f599339_39586f9d59004284a7114a68825e8eec',
              '2a80b29c0281435ca4893e158a281ce0_2afb9d32310e4546a71cbe432911eca2',
              '8ce4fe54af77467d90c840465f69677f_de7593d76648450e947ba0c203dee1b0',
              'f4af7fd215ee47aa8b64bac0443d7be8_9ee4020153674b9e9943d395ff8cfdf3']

    tokens = tokens * 32

    nusc = NuScenes('v1.0-trainval', dataroot=args.data_root)
    helper = PredictHelper(nusc)

    dataset = TestDataset(tokens, helper)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=16)

    backbone = ResNetBackbone('resnet18')
    model = MTP(backbone, NUM_MODES)
    model = model.to(device)

    loss_function = MTPLoss(NUM_MODES, 1, 5)

    current_loss = 10000

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    n_iter = 0

    minimum_loss = 0

    while True:

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

            if n_iter % 32 == 0:
                print(f"Number of iterations: {n_iter}.")
            
            n_iter += 1


