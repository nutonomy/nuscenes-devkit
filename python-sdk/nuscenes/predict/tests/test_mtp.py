import unittest
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from nuscenes.predict.models import mtp


class TestBackBones(unittest.TestCase):

    def count_layers(self, model):
        if isinstance(model[4][0], BasicBlock):
            n_convs = 2
        elif isinstance(model[4][0], Bottleneck):
            n_convs = 3
        return sum([len(model[i]) for i in range(4, 8)]) * n_convs + 2

    def test_resnet(self):

        rn_18 = mtp.ResNetBackbone('resnet18')
        rn_34 = mtp.ResNetBackbone('resnet34')
        rn_50 = mtp.ResNetBackbone('resnet50')
        rn_101 = mtp.ResNetBackbone('resnet101')
        rn_152 = mtp.ResNetBackbone('resnet152')

        tensor = torch.ones((1, 3, 100, 100))

        self.assertEqual(rn_18(tensor).shape[1], 512)
        self.assertEqual(rn_34(tensor).shape[1], 512)
        self.assertEqual(rn_50(tensor).shape[1], 2048)
        self.assertEqual(rn_101(tensor).shape[1], 2048)
        self.assertAlmostEqual(rn_152(tensor).shape[1], 2048)

        self.assertEqual(self.count_layers(list(rn_18.backbone.children())), 18)
        self.assertEqual(self.count_layers(list(rn_34.backbone.children())), 34)
        self.assertEqual(self.count_layers(list(rn_50.backbone.children())), 50)
        self.assertEqual(self.count_layers(list(rn_101.backbone.children())), 101)
        self.assertEqual(self.count_layers(list(rn_152.backbone.children())), 152)

        with self.assertRaises(ValueError):
            mtp.ResNetBackbone('resnet51')

    def test_mobilenet(self):

        mobilenet = mtp.MobileNetBackbone('mobilenet_v2')

        tensor = torch.ones((1, 3, 100, 100))

        self.assertEqual(mobilenet(tensor).shape[1], 1280)


class TestMTP(unittest.TestCase):

    def setUp(self):
        self.image = torch.ones((1, 3, 100, 100))
        self.agent_state_vector = torch.ones((1, 3))
        self.image_5 = torch.ones((5, 3, 100, 100))
        self.agent_state_vector_5 = torch.ones((5, 3))

    def _run(self, model):
        pred = model(self.image, self.agent_state_vector)
        pred_5 = model(self.image_5, self.agent_state_vector_5)

        self.assertTupleEqual(pred.shape, (1, 75))
        self.assertTupleEqual(pred_5.shape, (5, 75))

        model.training = False
        pred = model(self.image, self.agent_state_vector)
        self.assertTrue(torch.allclose(pred[:, -3:].sum(axis=1), torch.ones(pred.shape[0])))

    def test_works_with_resnet_18(self,):
        rn_18 = mtp.ResNetBackbone('resnet18')
        model = mtp.MTP(rn_18, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_34(self,):
        rn_34 = mtp.ResNetBackbone('resnet34')
        model = mtp.MTP(rn_34, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_50(self,):
        rn_50 = mtp.ResNetBackbone('resnet50')
        model = mtp.MTP(rn_50, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_101(self,):
        rn_101 = mtp.ResNetBackbone('resnet101')
        model = mtp.MTP(rn_101, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_152(self,):
        rn_152 = mtp.ResNetBackbone('resnet152')
        model = mtp.MTP(rn_152, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_mobilenet_v2(self,):
        mobilenet = mtp.MobileNetBackbone('mobilenet_v2')
        model = mtp.MTP(mobilenet, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)



