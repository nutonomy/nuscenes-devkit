# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import fire
import os
import json
import tarfile
from typing import List


def export_release(dataroot='/data/sets/nuimages', version: str = 'v1.0') -> None:
    """
    This script tars the image and metadata files for release on https://www.nuscenes.org/download.
    :param dataroot: The nuImages folder.
    :param version: The nuImages dataset version.
    """
    # Create export folder.
    export_dir = os.path.join(dataroot, 'export')
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    # Determine the images from the mini split.
    mini_src = os.path.join(dataroot, version + '-mini')
    with open(os.path.join(mini_src, 'sample_data.json'), 'r') as f:
        sample_data = json.load(f)
    file_names = [sd['filename'] for sd in sample_data]

    # Hard-code the mapping from archive names to their relative folder paths.
    archives = {
        'all-metadata': [version + '-train', version + '-val', version + '-test', version + '-mini'],
        'all-samples': ['samples'],
        'all-sweeps-cam-back': ['sweeps/CAM_BACK'],
        'all-sweeps-cam-back-left': ['sweeps/CAM_BACK_LEFT'],
        'all-sweeps-cam-back-right': ['sweeps/CAM_BACK_RIGHT'],
        'all-sweeps-cam-front': ['sweeps/CAM_FRONT'],
        'all-sweeps-cam-front-left': ['sweeps/CAM_FRONT_LEFT'],
        'all-sweeps-cam-front-right': ['sweeps/CAM_FRONT_RIGHT'],
        'mini': [version + '-mini'] + file_names
    }

    # Pack each folder.
    for key, folder_list in archives.items():
        out_path = os.path.join(export_dir, 'nuimages-%s-%s.tgz' % (version, key))
        if os.path.exists(out_path):
            print('Warning: Skipping export for file as it already exists: %s' % out_path)
            continue
        print('Compressing archive %s...' % out_path)
        pack_folder(out_path, dataroot, folder_list)


def pack_folder(out_path: str, dataroot: str, folder_list: List[str], tar_format: str = 'w:gz') -> None:
    """
    :param out_path: The output path where we write the tar file.
    :param dataroot: The nuImages folder.
    :param folder_list: List of files or folders to include in the archive.
    :param tar_format: The compression format to use. See tarfile package for more options.
    """
    tar = tarfile.open(out_path, tar_format)
    for name in folder_list:
        folder_path = os.path.join(dataroot, name)
        tar.add(folder_path, arcname=name)
    tar.close()


if __name__ == '__main__':
    fire.Fire(export_release)
