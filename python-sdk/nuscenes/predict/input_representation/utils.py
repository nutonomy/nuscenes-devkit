from typing import Tuple




def convert_to_pixel_coords(x: float, y: float,
                            image_shape: Tuple[int, int], resolution: float = 0.1,
                            bottom_left: Tuple[int, int] = (0, 0)):
    """
    Convert from global coordinates to
    :param angles_from_ground_truth: List of angles
    :param target: Shape [1, n_timesteps, 2]
    :param trajectories: Shape [n_modes, n_timesteps, 2]
        """
    x_pixel = (x - bottom_left[0])/resolution
    y_pixel = image_shape[0] - (bottom_left[1] - y)/resolution
    return x_pixel, y_pixel