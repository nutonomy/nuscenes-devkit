import unittest

from nuscenes.tests.assert_download import _check_file_exists


class TestAssertDownloadHelpers(unittest.TestCase):

    def test_check_file_exists_passes_for_existing_file(self):
        _check_file_exists(__file__, 'test file')

    def test_check_file_exists_raises_for_missing_file(self):
        missing_path = '/tmp/this_file_should_not_exist_123456789.txt'
        with self.assertRaises(AssertionError) as context:
            _check_file_exists(missing_path, 'sample_data')
        self.assertIn('Missing sample_data at', str(context.exception))


if __name__ == '__main__':
    unittest.main()
