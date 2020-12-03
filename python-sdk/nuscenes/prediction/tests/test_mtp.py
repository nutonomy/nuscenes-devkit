import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as torch was not found!')

from nuscenes.prediction.models import backbone
from nuscenes.prediction.models import mtp


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
        rn_18 = backbone.ResNetBackbone('resnet18')
        model = mtp.MTP(rn_18, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_34(self,):
        rn_34 = backbone.ResNetBackbone('resnet34')
        model = mtp.MTP(rn_34, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_50(self,):
        rn_50 = backbone.ResNetBackbone('resnet50')
        model = mtp.MTP(rn_50, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_101(self,):
        rn_101 = backbone.ResNetBackbone('resnet101')
        model = mtp.MTP(rn_101, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_resnet_152(self,):
        rn_152 = backbone.ResNetBackbone('resnet152')
        model = mtp.MTP(rn_152, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)

    def test_works_with_mobilenet_v2(self,):
        mobilenet = backbone.MobileNetBackbone('mobilenet_v2')
        model = mtp.MTP(mobilenet, 3, 6, 2, input_shape=(3, 100, 100))
        self._run(model)



