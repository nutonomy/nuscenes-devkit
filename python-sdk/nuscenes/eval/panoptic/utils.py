"""
Panoptic nuScenes utils.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""
from typing import Dict

from nuscenes.eval.lidarseg.utils import LidarsegClassMapper, get_samples_in_eval_set
from nuscenes.nuscenes import NuScenes

get_samples_in_panoptic_eval_set = get_samples_in_eval_set


class PanopticClassMapper(LidarsegClassMapper):
    """
    Maps the general (fine) classes to the challenge (coarse) classes in the Panoptic nuScenes challenge.

    Example usage::
        nusc_ = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
        mapper_ = PanopticClassMapper(nusc_)
    """

    def __init__(self, nusc: NuScenes):
        """
        Initialize a PanopticClassMapper object.
        :param nusc: A NuScenes object.
        """
        super(PanopticClassMapper, self).__init__(nusc)
        self.things = self.get_things()
        self.stuff = self.get_stuff()

    def get_stuff(self) -> Dict[str, int]:
        """
        Returns the mapping from the challenge (coarse) class names to the challenge class indices for stuff.
        :return: A dictionary containing the mapping from the challenge class names to the challenge class indices for
            stuff.
            {
              'driveable_surface': 11,
              'other_flat': 12,
              'sidewalk': 13,
              'terrain': 14,
              'manmade': 15,
              'vegetation': 16
            }
        """
        stuff_names = {'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'}
        coarse_name_to_id = self.get_coarse2idx()
        assert stuff_names <= set(coarse_name_to_id.keys()), 'Invalid stuff names, pls check !'
        stuff_name_to_id = {name: coarse_name_to_id[name] for name in stuff_names}

        return stuff_name_to_id

    def get_things(self) -> Dict[str, int]:
        """
        Returns the mapping from the challenge (coarse) class names to the challenge class indices for things.
        :return: A dictionary containing the mapping from the challenge class names to the challenge class indices for
            things.
            {
              'barrier': 1,
              'bicycle': 2,
              'bus': 3,
              'car': 4,
              'construction_vehicle': 5,
              'motorcycle': 6,
              'pedestrian': 7,
              'traffic_cone': 8,
              'trailer': 9,
              'truck': 10
            }
        """
        thing_names = {'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
                       'traffic_cone', 'trailer', 'truck'}
        coarse_name_to_id = self.get_coarse2idx()
        assert thing_names <= set(coarse_name_to_id.keys()), 'Invalid thing names, pls check !'
        thing_name_to_id = {name: coarse_name_to_id[name] for name in thing_names}

        return thing_name_to_id
