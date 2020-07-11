import argparse
import os
import random
from typing import List

import cv2
import tqdm

from nuimages.nuimages import NuImages


class ImageRenderer:

    def __init__(self, version: str = 'v1.0-val', dataroot: str = '/data/sets/nuimages', verbose: bool = False):
        """
        Initialize ImageRenderer.
        :param version: The NuImages version.
        :param dataroot: The root folder where the dataset is installed.
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.nuim = NuImages(version=self.version, dataroot=self.dataroot, verbose=self.verbose, lazy=False)

    def render_images(self,
                      mode: str = 'all',
                      cam_name: str = None,
                      sample_limit: int = 100,
                      out_type: str = 'image',
                      out_dir: str = '~/Downloads/nuImages',
                      cleanup: bool = True) -> None:
        """
        Render a random selection of images and save them to disk.
        Note: The images rendered here are keyframes only.
        :param mode: What to render:
          "annotated" for the image with annotations,
          "raw" for the image without annotations,
          "dept_dense" for dense depth image,
          "dept_sparse" for sparse depth image,
          "pointcloud" for a birds-eye view of the pointcloud,
          "trajectory" for a rendering of the trajectory of the vehice,
          "all" to render all of the above separately.
        :param cam_name: Only render images from a particular camera, e.g. "CAM_BACK'.
        :param sample_limit: Maximum number of samples (images) to render.
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
        assert not(out_type == 'video' and mode == 'trajectory'), 'Error" Cannot render "trajectory" for videos!'

        if mode == 'all':
            if out_type == 'image':
                modes = all_modes
            elif out_type == 'video':
                modes = [m for m in all_modes if m not in ['annotated', 'trajectory']]
            else:
                raise Exception('Error" Unknown mode %s!' % mode)
        else:
            modes = [mode]

        # Create output folder.
        out_dir = os.path.expanduser(out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Get a random selection of samples.
        sample_tokens = [s['token'] for s in self.nuim.sample]
        random.shuffle(sample_tokens)

        # Filter by camera.
        if cam_name is not None:
            sample_tokens_cam = []
            for sample_token in sample_tokens:
                sample = self.nuim.get('sample', sample_token)
                sd_token_camera = sample['key_camera_token']
                sensor = self.nuim.shortcut('sample_data', 'sensor', sd_token_camera)
                if sensor['channel'] == cam_name:
                    sample_tokens_cam.append(sample_token)
            sample_tokens = sample_tokens_cam

        # Limit number of samples.
        sample_tokens = sample_tokens[:sample_limit]

        print('Rendering %s for mode %s to folder %s...' % (out_type, mode, out_dir))
        for sample_token in tqdm.tqdm(sample_tokens):
            sample = self.nuim.get('sample', sample_token)
            sd_token_camera = sample['key_camera_token']
            sensor = self.nuim.shortcut('sample_data', 'sensor', sd_token_camera)
            sample_cam_name = sensor['channel']
            sd_tokens_camera = self.nuim.get_sample_content(sample_token, modality='camera')

            for mode in modes:
                out_path_prefix = os.path.join(out_dir, '%s_%s_%s' % (sample_token, sample_cam_name, mode))
                if out_type == 'image':
                    self.write_image(sd_token_camera, mode, '%s.jpg' % out_path_prefix)
                elif out_type == 'video':
                    self.write_video(sd_tokens_camera, mode, out_path_prefix, cleanup=cleanup)

    def write_video(self, sd_tokens_camera: List[str], mode: str, out_path_prefix: str, cleanup: bool = True) -> None:
        """
        Render a video by combining all the images of type mode for each sample_data.
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
            self.write_image(sd_token_camera, mode, out_path)

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

    def write_image(self, sd_token_camera: str, mode: str, out_path: str) -> None:
        """
        Render a single image of type mode for the given sample_data.
        :param sd_token_camera: The sample_data token of the camera.
        :param mode: The mode - see render_images().
        :param out_path: The file to write the image to.
        """
        if mode == 'annotated':
            self.nuim.render_image(sd_token_camera, with_annotations=True, out_path=out_path)
        elif mode == 'image':
            self.nuim.render_image(sd_token_camera, with_annotations=False, out_path=out_path)
        elif mode == 'depth_dense':
            self.nuim.render_depth(sd_token_camera, mode='dense', out_path=out_path)
        elif mode == 'depth_sparse':
            self.nuim.render_depth(sd_token_camera, mode='sparse', out_path=out_path)
        elif mode == 'pointcloud':
            self.nuim.render_pointcloud(sd_token_camera, out_path=out_path)
        elif mode == 'trajectory':
            sd_camera = self.nuim.get('sample_data', sd_token_camera)
            self.nuim.render_trajectory(sd_camera['sample_token'], out_path=out_path)
        else:
            raise Exception('Error: Unknown mode %s!' % mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--seed', type=int, default=42)  # Set to 0 to disable.
    parser.add_argument('--version', type=str, default='v1.0-val')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=100)
    parser.add_argument('--out_type', type=str, default='image')
    parser.add_argument('--out_dir', type=str, default='~/Downloads/nuImages')
    args = parser.parse_args()

    # Set random seed for reproducible image selection.
    if args.seed != 0:
        random.seed(args.seed)

    # Render images.
    renderer = ImageRenderer(args.version, args.dataroot, bool(args.verbose))
    renderer.render_images(mode=args.mode, cam_name=args.cam_name, sample_limit=args.sample_limit,
                           out_type=args.out_type, out_dir=args.out_dir)
