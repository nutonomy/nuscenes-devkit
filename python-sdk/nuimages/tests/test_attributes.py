import unittest
import os

from nuimages.nuimages import NuImages


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.nuim = NuImages(version='v1.0-val', dataroot=os.environ['NUIMAGES'], verbose=False)

        self.valid_attributes = {
            'animal': ['pedestrian', 'vertical_position'],
            'human.pedestrian.adult': ['pedestrian'],
            'human.pedestrian.child': ['pedestrian'],
            'human.pedestrian.construction_worker': ['pedestrian'],
            'human.pedestrian.personal_mobility': ['has_rider'],
            'human.pedestrian.police_officer': ['pedestrian'],
            'human.pedestrian.stroller': [],
            'human.pedestrian.wheelchair': [],
            'movable_object.barrier': [],
            'movable_object.debris': [],
            'movable_object.pushable_pullable': [],
            'movable_object.trafficcone': [],
            'static_object.bicycle_rack': [],
            'vehicle.bicycle': ['has_rider'],
            'vehicle.bus.bendy': ['vehicle'],
            'vehicle.bus.rigid': ['vehicle'],
            'vehicle.car': ['vehicle'],
            'vehicle.construction': ['vehicle'],
            'vehicle.ego': [],
            'vehicle.emergency.ambulance': ['vehicle', 'vehicle_light.emergency'],
            'vehicle.emergency.police': ['vehicle', 'vehicle_light.emergency'],
            'vehicle.motorcycle': ['has_rider'],
            'vehicle.trailer': ['vehicle'],
            'vehicle.truck': ['vehicle']
        }

    def test_object_anns(self, print_only: bool = True) -> None:
        """
        For every object_ann, check that all the required attributes for that class are present.
        """
        att_token_to_name = {att['token']: att['name'] for att in self.nuim.attribute}
        cat_token_to_name = {cat['token']: cat['name'] for cat in self.nuim.category}
        for object_ann in self.nuim.object_ann:
            # Collect the attribute names used here.
            category_name = cat_token_to_name[object_ann['category_token']]

            cur_att_names = []
            for attribute_token in object_ann['attribute_tokens']:
                attribute_name = att_token_to_name[attribute_token]
                cur_att_names.append(attribute_name)

            # Compare to the required attribute name prefixes.
            # Check that the length is correct.
            required_att_names = self.valid_attributes[category_name]
            condition = len(cur_att_names) == len(required_att_names)
            if not condition:
                sample_token = self.nuim.get('sample_data', object_ann['sample_data_token'])['sample_token']
                debug_output = {
                    'sample_token': sample_token,
                    'category_name': category_name,
                    'cur_att_names': cur_att_names,
                    'required_att_names': required_att_names
                }
                error_msg = 'Error in test_object_anns: ' + str(debug_output)
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
                    error_msg = 'Error in test_object_anns: Required attribute ''%s'' not in %s for class %s!' \
                                % (required, cur_att_names, category_name)
                    if print_only:
                        print(error_msg)
                    else:
                        self.assertTrue(condition, error_msg)


if __name__ == '__main__':
    # Runs the tests without throwing errors
    test = TestAttributes()
    test.setUp()
    test.test_object_anns(print_only=True)
