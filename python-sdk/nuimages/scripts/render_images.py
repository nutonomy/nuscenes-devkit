import argparse
import gc
import os
import random
from typing import List
from collections import defaultdict

import cv2
import tqdm

from nuimages.nuimages import NuImages


def render_images(nuim: NuImages,
                  mode: str = 'all',
                  cam_name: str = None,
                  log_name: str = None,
                  sample_limit: int = 100,
                  filter_categories: List[str] = None,
                  out_type: str = 'image',
                  out_dir: str = '~/Downloads/nuImages',
                  cleanup: bool = True) -> None:
    """
    Render a random selection of images and save them to disk.
    Note: The images rendered here are keyframes only.
    :param nuim: NuImages instance.
    :param mode: What to render:
      "annotated" for the image with annotations,
      "raw" for the image without annotations,
      "dept_dense" for dense depth image,
      "dept_sparse" for sparse depth image,
      "pointcloud" for a birds-eye view of the pointcloud,
      "trajectory" for a rendering of the trajectory of the vehice,
      "all" to render all of the above separately.
    :param cam_name: Only render images from a particular camera, e.g. "CAM_BACK'.
    :param log_name: Only render images from a particular log, e.g. "n013-2018-09-04-13-30-50+0800".
    :param sample_limit: Maximum number of samples (images) to render.
    :param filter_categories: Specify a list of object_ann category names. Every sample that is rendered must
        contain annotations of any of those categories.
    :param out_type: The output type as one of the following:
        'image': Renders a single image for the image keyframe of each sample.
        'video': Renders a video for all images/pcls in the clip associated with each sample.
    :param out_dir: Folder to render the images to.
    :param cleanup: Whether to delete images after rendering the video. Not relevant for out_type == 'image'.
    """
    # Check and convert inputs.
    assert out_type in ['image', 'video'], ' Error: Unknown out_type %s!' % out_type
    all_modes = ['annotated', 'image', 'depth_dense', 'depth_sparse', 'pointcloud', 'trajectory']
    assert mode in all_modes + ['all'], 'Error: Unknown mode %s!' % mode
    assert not (out_type == 'video' and mode == 'trajectory'), 'Error: Cannot render "trajectory" for videos!'

    if mode == 'all':
        if out_type == 'image':
            modes = all_modes
        elif out_type == 'video':
            modes = [m for m in all_modes if m not in ['annotated', 'trajectory']]
        else:
            raise Exception('Error" Unknown mode %s!' % mode)
    else:
        modes = [mode]

    if filter_categories is not None:
        category_names = [c['name'] for c in nuim.category]
        for category_name in filter_categories:
            assert category_name in category_names, 'Error: Invalid object_ann category %s!' % category_name

    # Create output folder.
    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Filter by camera.
    sample_tokens = [s['token'] for s in nuim.sample]
    if cam_name is not None:
        sample_tokens_cam = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            sd_token_camera = sample['key_camera_token']
            sensor = nuim.shortcut('sample_data', 'sensor', sd_token_camera)
            if sensor['channel'] == cam_name:
                sample_tokens_cam.append(sample_token)
        sample_tokens = sample_tokens_cam

    # Filter by log.
    if log_name is not None:
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            log = nuim.get('log', sample['log_token'])
            if log['logfile'] == log_name:
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Filter samples by category.
    if filter_categories is not None:
        # Get categories in each sample.
        sd_to_object_cat_names = defaultdict(lambda: set())
        for object_ann in nuim.object_ann:
            category = nuim.get('category', object_ann['category_token'])
            sd_to_object_cat_names[object_ann['sample_data_token']].add(category['name'])

        # Filter samples.
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            sd_token_camera = sample['key_camera_token']
            category_names = sd_to_object_cat_names[sd_token_camera]
            if any([c in category_names for c in filter_categories]):
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Get a random selection of samples.
    random.shuffle(sample_tokens)

    # Limit number of samples.
    sample_tokens = sample_tokens[:sample_limit]

    print('Rendering %s for mode %s to folder %s...' % (out_type, mode, out_dir))
    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nuim.get('sample', sample_token)
        sd_token_camera = sample['key_camera_token']
        sensor = nuim.shortcut('sample_data', 'sensor', sd_token_camera)
        sample_cam_name = sensor['channel']
        sd_tokens_camera = nuim.get_sample_content(sample_token, modality='camera')

        # We cannot render a video if there are missing camera sample_datas.
        if len(sd_tokens_camera) < 13 and out_type == 'video':
            continue

        for mode in modes:
            out_path_prefix = os.path.join(out_dir, '%s_%s_%s' % (sample_token, sample_cam_name, mode))
            if out_type == 'image':
                write_image(nuim, sd_token_camera, mode, '%s.jpg' % out_path_prefix)
            elif out_type == 'video':
                write_video(nuim, sd_tokens_camera, mode, out_path_prefix, cleanup=cleanup)


def write_video(nuim: NuImages,
                sd_tokens_camera: List[str],
                mode: str,
                out_path_prefix: str,
                cleanup: bool = True) -> None:
    """
    Render a video by combining all the images of type mode for each sample_data.
    :param nuim: NuImages instance.
    :param sd_tokens_camera: All camera sample_data tokens in chronological order.
    :param mode: The mode - see render_images().
    :param out_path_prefix: The file prefix used for the images and video.
    :param cleanup: Whether to delete images after rendering the video.
    """
    # Loop through each frame to create the video.
    out_paths = []
    for i, sd_token_camera in enumerate(sd_tokens_camera):
        out_path = '%s_%d.jpg' % (out_path_prefix, i)
        out_paths.append(out_path)
        write_image(nuim, sd_token_camera, mode, out_path)

    # Create video.
    first_im = cv2.imread(out_paths[0])
    freq = 2  # Display frequency (Hz).
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_path = '%s.avi' % out_path_prefix
    out = cv2.VideoWriter(video_path, fourcc, freq, first_im.shape[1::-1])

    # Load each image and add to the video.
    for out_path in out_paths:
        im = cv2.imread(out_path)
        out.write(im)

        # Delete temporary image if requested.
        if cleanup:
            os.remove(out_path)

    # Finalize video.
    out.release()


def write_image(nuim: NuImages, sd_token_camera: str, mode: str, out_path: str) -> None:
    """
    Render a single image of type mode for the given sample_data.
    :param nuim: NuImages instance.
    :param sd_token_camera: The sample_data token of the camera.
    :param mode: The mode - see render_images().
    :param out_path: The file to write the image to.
    """
    if mode == 'annotated':
        nuim.render_image(sd_token_camera, with_annotations=True, out_path=out_path)
    elif mode == 'image':
        nuim.render_image(sd_token_camera, with_annotations=False, out_path=out_path)
    elif mode == 'depth_dense':
        nuim.render_depth(sd_token_camera, mode='dense', out_path=out_path)
    elif mode == 'depth_sparse':
        nuim.render_depth(sd_token_camera, mode='sparse', out_path=out_path)
    elif mode == 'pointcloud':
        nuim.render_pointcloud(sd_token_camera, out_path=out_path)
    elif mode == 'trajectory':
        sd_camera = nuim.get('sample_data', sd_token_camera)
        nuim.render_trajectory(sd_camera['sample_token'], out_path=out_path)
    else:
        raise Exception('Error: Unknown mode %s!' % mode)

    # Trigger garbage collection to avoid memory overflow from the render functions.
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--seed', type=int, default=42)  # Set to 0 to disable.
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=100)
    parser.add_argument('--filter_categories', action='append')
    parser.add_argument('--out_type', type=str, default='image')
    parser.add_argument('--out_dir', type=str, default='~/Downloads/nuImages')
    args = parser.parse_args()

    # Set random seed for reproducible image selection.
    if args.seed != 0:
        random.seed(args.seed)

    # Initialize NuImages class.
    nuim_ = NuImages(version=args.version, dataroot=args.dataroot, verbose=bool(args.verbose), lazy=False)

    # Render images.
    render_images(nuim_, mode=args.mode, cam_name=args.cam_name, log_name=args.log_name, sample_limit=args.sample_limit,
                  filter_categories=args.filter_categories, out_type=args.out_type, out_dir=args.out_dir)
