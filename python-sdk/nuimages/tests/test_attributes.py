# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import os
import unittest
from typing import Any

from nuimages.nuimages import NuImages


class TestAttributes(unittest.TestCase):

    def __init__(self, _: Any = None, version: str = 'v1.0-mini', dataroot: str = None):
        """
        Initialize TestAttributes.
        Note: The second parameter is a dummy parameter required by the TestCase class.
        :param version: The NuImages version.
        :param dataroot: The root folder where the dataset is installed.
        """
        super().__init__()

        self.version = version
        if dataroot is None:
            self.dataroot = os.environ['NUIMAGES']
        else:
            self.dataroot = dataroot
        self.nuim = NuImages(version=self.version, dataroot=self.dataroot, verbose=False)
        self.valid_attributes = {
            'animal': ['pedestrian', 'vertical_position'],
            'human.pedestrian.adult': ['pedestrian'],
            'human.pedestrian.child': ['pedestrian'],
            'human.pedestrian.construction_worker': ['pedestrian'],
            'human.pedestrian.personal_mobility': ['cycle'],
            'human.pedestrian.police_officer': ['pedestrian'],
            'human.pedestrian.stroller': [],
            'human.pedestrian.wheelchair': [],
            'movable_object.barrier': [],
            'movable_object.debris': [],
            'movable_object.pushable_pullable': [],
            'movable_object.trafficcone': [],
            'static_object.bicycle_rack': [],
            'vehicle.bicycle': ['cycle'],
            'vehicle.bus.bendy': ['vehicle'],
            'vehicle.bus.rigid': ['vehicle'],
            'vehicle.car': ['vehicle'],
            'vehicle.construction': ['vehicle'],
            'vehicle.ego': [],
            'vehicle.emergency.ambulance': ['vehicle', 'vehicle_light.emergency'],
            'vehicle.emergency.police': ['vehicle', 'vehicle_light.emergency'],
            'vehicle.motorcycle': ['cycle'],
            'vehicle.trailer': ['vehicle'],
            'vehicle.truck': ['vehicle']
        }

    def runTest(self) -> None:
        """
        Dummy function required by the TestCase class.
        """
        pass

    def test_object_anns(self, print_only: bool = False) -> None:
        """
        For every object_ann, check that all the required attributes for that class are present.
        :param print_only: Whether to throw assertion errors or just print a warning message.
        """
        att_token_to_name = {att['token']: att['name'] for att in self.nuim.attribute}
        cat_token_to_name = {cat['token']: cat['name'] for cat in self.nuim.category}
        for object_ann in self.nuim.object_ann:
            # Collect the attribute names used here.
            category_name = cat_token_to_name[object_ann['category_token']]
            sample_token = self.nuim.get('sample_data', object_ann['sample_data_token'])['sample_token']

            cur_att_names = []
            for attribute_token in object_ann['attribute_tokens']:
                attribute_name = att_token_to_name[attribute_token]
                cur_att_names.append(attribute_name)

            # Compare to the required attribute name prefixes.
            # Check that the length is correct.
            required_att_names = self.valid_attributes[category_name]
            condition = len(cur_att_names) == len(required_att_names)
            if not condition:
                debug_output = {
                    'sample_token': sample_token,
                    'category_name': category_name,
                    'cur_att_names': cur_att_names,
                    'required_att_names': required_att_names
                }
                error_msg = 'Error: ' + str(debug_output)
                if print_only:
                    print(error_msg)
                else:
                    self.assertTrue(condition, error_msg)

                # Skip next check if we already saw an error.
                continue

            # Check that they are really the same.
            for required in required_att_names:
                condition = any([cur.startswith(required + '.') for cur in cur_att_names])
                if not condition:
                    error_msg = 'Errors: Required attribute ''%s'' not in %s for class %s! (sample %s)' \
                                % (required, cur_att_names, category_name, sample_token)
                    if print_only:
                        print(error_msg)
                    else:
                        self.assertTrue(condition, error_msg)


if __name__ == '__main__':
    # Runs the tests without aborting on error.
    for nuim_version in ['v1.0-train', 'v1.0-val', 'v1.0-test', 'v1.0-mini']:
        print('Running TestAttributes for version %s...' % nuim_version)
        test = TestAttributes(version=nuim_version)
        test.test_object_anns(print_only=True)
