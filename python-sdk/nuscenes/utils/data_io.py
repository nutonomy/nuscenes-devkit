import numpy as np
import os


def load_bin_file(bin_path: str, type: str = 'lidarseg') -> np.ndarray:
    """
    Loads a .bin file containing the lidarseg or lidar panoptic labels.
    :param bin_path: Path to the .bin file.
    :param type: semantic type, 'lidarseg': stored in 8-bit format, 'panoptic': store in 32-bit format.
    :return: An array containing the labels, with dtype of np.uint8 for lidarseg and np.int32 for panoptic.
    """
    assert os.path.exists(bin_path), 'Error: Unable to find {}.'.format(bin_path)
    if type == 'lidarseg':
        bin_content = np.fromfile(bin_path, dtype=np.uint8)
    elif type == 'panoptic':
        bin_content = np.load(bin_path)['data']
    else:
        raise TypeError(f"Only lidarseg/panoptic type is supported, received {type}")
    assert len(bin_content) > 0, 'Error: {} is empty.'.format(bin_path)

    return bin_content


def panoptic_to_lidarseg(panoptic_labels: np.ndarray) -> np.ndarray:
    """
    Convert panoptic label array to lidarseg label array
    :param panoptic_labels: <np.array, HxW, np.uint16>, encoded in (instance_id + 1000 * category_idx), note instance_id
    for stuff points is 0.
    :return: lidarseg semantic labels, <np.array, HxW, np.uint8>.
    """
    return (panoptic_labels // 1000).astype(np.uint8)
