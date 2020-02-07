from typing import List, Dict, Tuple
import os
import numpy as np
from pyquaternion import Quaternion

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.predict import PredictHelper
from nuscenes.predict.helper import angle_of_rotation
from nuscenes.predict.input_representation.interface import StaticLayerRepresentation
from nuscenes.predict.input_representation.combinators import Rasterizer
from nuscenes.eval.common.utils import quaternion_yaw

def load_all_maps(helper: PredictHelper) -> Dict[str, NuScenesMap]:
    """Loads all NuScenesMap instances for all available maps."""
    dataroot = helper.data.dataroot

    json_files = filter(lambda f: "json" in f, os.listdir(os.path.join(dataroot, "maps")))

    maps = {}
    for map_file in json_files:
        map_name = map_file.split(".")[0]
        maps[map_name] = NuScenesMap(dataroot, map_name=map_name)

    return maps

def get_patchbox(x: float, y: float,
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25) -> Tuple[Tuple[float, float, float, float], float]:

    # Get the most data possible before the rotation
    buffer = max(meters_ahead, meters_behind, meters_left, meters_right)

    patch_box = (x, y, 2 * buffer, 2 * buffer)

    return patch_box

def change_color_of_binary_mask(image, color):
    image = image*color
    image = image.astype("uint8")
    return image

class StaticLayerRasterizer(StaticLayerRepresentation):

    def __init__(self, helper: PredictHelper,
                 layer_names: List[str] = None,
                 colors: List[Tuple[float, float, float]] = None,
                 resolution: float = 0.1,
                 meters_ahead: float = 40, meters_behind: float = 10,
                 meters_left: float = 25, meters_right: float = 25):

        self.helper = helper
        self.maps = load_all_maps(helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        if not colors:
            colors = [(255, 255, 255), (0, 255, 0), (0, 0, 255)]
        self.colors = colors

        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right
        self.combinator = Rasterizer()

    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:

        sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_token)

        x, y = sample_annotation['translation'][:2]

        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))

        if yaw <= 0:
            yaw = -np.pi - yaw
        else:
            yaw = np.pi - yaw

        patchbox = get_patchbox(x, y, self.meters_ahead,
                                self.meters_behind, self.meters_left, self.meters_right)

        angle = (angle_of_rotation(yaw) * 180/np.pi)

        masks = self.maps[map_name].get_map_mask(patchbox, angle, self.layer_names, canvas_size=None)
        images = [change_color_of_binary_mask(np.repeat(mask[:, :, np.newaxis], 3, 2), color) for mask, color in zip(masks, self.colors)]

        image = self.combinator.combine(images)

        return image[:500, 150:650, :]