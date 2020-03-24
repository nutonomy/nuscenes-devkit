import unittest

from nuscenes.prediction.input_representation import utils


class Test_convert_to_pixel_coords(unittest.TestCase):

    def test_above_and_to_the_right(self):

        location = (55, 60)
        center_of_image_in_global = (50, 50)
        center_of_image_in_pixels = (400, 250)

        pixels = utils.convert_to_pixel_coords(location,
                                               center_of_image_in_global,
                                               center_of_image_in_pixels)

        answer = (300, 300)
        self.assertTupleEqual(pixels, answer)

        pixels = utils.convert_to_pixel_coords(location,
                                               center_of_image_in_global,
                                               center_of_image_in_pixels,
                                               resolution=0.2)
        answer = (350, 275)
        self.assertTupleEqual(pixels, answer)

    def test_above_and_to_the_left(self):

        location = (40, 70)
        center_of_image_in_global = (50, 50)
        center_of_image_in_pixels = (300, 300)

        pixels = utils.convert_to_pixel_coords(location, center_of_image_in_global,
                                               center_of_image_in_pixels)
        answer = (100, 200)
        self.assertTupleEqual(pixels, answer)

        pixels = utils.convert_to_pixel_coords(location, center_of_image_in_global,
                                                center_of_image_in_pixels, resolution=0.2)
        answer = (200, 250)
        self.assertTupleEqual(answer, pixels)

    def test_below_and_to_the_right(self):

        location = (60, 45)
        center_of_image_in_global = (50, 50)
        center_of_image_in_pixels = (400, 250)

        pixels = utils.convert_to_pixel_coords(location, center_of_image_in_global, center_of_image_in_pixels)
        answer = (450, 350)
        self.assertTupleEqual(pixels, answer)

    def test_below_and_to_the_left(self):

        location = (30, 40)
        center_of_image_in_global = (50, 50)
        center_of_image_in_pixels = (400, 250)

        pixels = utils.convert_to_pixel_coords(location, center_of_image_in_global, center_of_image_in_pixels)
        answer = (500, 50)
        self.assertTupleEqual(pixels, answer)

    def test_same_location(self):

        location = (50, 50)
        center_of_image_in_global = (50, 50)
        center_of_image_in_pixels = (400, 250)

        pixels = utils.convert_to_pixel_coords(location, center_of_image_in_global, center_of_image_in_pixels)
        self.assertTupleEqual(pixels, (400, 250))

class Test_get_crops(unittest.TestCase):

    def test(self):

        row_crop, col_crop = utils.get_crops(40, 10, 25, 25, 0.1, 800)

        self.assertEqual(row_crop, slice(0, 500))
        self.assertEqual(col_crop, slice(150, 650))
