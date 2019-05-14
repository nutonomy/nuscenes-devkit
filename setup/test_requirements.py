import importlib
import os
import unittest
import warnings


class TestRequirements(unittest.TestCase):

    def setUp(self):

        this_dir = os.path.dirname(os.path.abspath(__file__))
        requirements = os.path.join(this_dir, 'requirements.txt')
        with open(requirements, 'r') as f:
            requirements = f.readlines()

        specifiers = ['==', '>=', '<=', '>', '<']  # types of specifiers allowed by pip

        for specifier in specifiers:
            requirements = [requirement.split(specifier)[0] for requirement in requirements]

        requirements = [requirement.strip() for requirement in requirements]

        self.requirements = requirements

        self.module_renamings = {
            'opencv-python': 'cv2',
            'Pillow': 'PIL',
            'scikit-learn': 'sklearn',
            'Shapely': 'shapely',
        }

    def test_requirements(self):
        """ Verify code is being run in a properly configured Python environment. """

        for requirement in self.requirements:
            if requirement in self.module_renamings.keys():
                mod_name = self.module_renamings[requirement]
            else:
                mod_name = requirement

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                warnings.filterwarnings("ignore", category=ImportWarning)

                try:
                    mod = importlib.import_module(mod_name)
                    self.assertIsNotNone(mod)
                except:
                    msg = 'Requirement {} is incorrectly installed and therefore {} cannot be imported'.format(requirement, mod_name)
                    self.assertTrue(False, msg=msg)


if __name__ == '__main__':
    unittest.main()
