import argparse
import os
import random

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
                      image_limit: int = 100,
                      out_dir: str = '~/Downloads/nuImages') -> None:
        """
        Render a random selection of images and save them to disk.
        Note: The images rendered here are keyframes only.
        :param mode: What to render:
          "annotated" for the image with annotations,
          "raw" for the image without annotations,
          "depth" for depth image,
          "all" to render all of the above separately.
        :param cam_name: Only render images from a particular camera, e.g. "CAM_BACK'.
        :param image_limit: Maximum number of images to render.
        :param out_dir: Folder to render the images to.
        """
        # Check and convert inputs.
        all_modes = ['annotated', 'image', 'depth_dense', 'depth_sparse', 'pointcloud', 'trajectory']
        assert mode in all_modes + ['all']
        out_dir = os.path.expanduser(out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if mode == 'all':
            modes = all_modes
        else:
            modes = [mode]

        random.seed(42)  # TODO: remove later

        # Get a random selection of samples.
        sample_tokens = [s['token'] for s in self.nuim.sample]
        random.shuffle(sample_tokens)

        # sample_tokens = ['749d065e9bba4d19ae4030eab81970b3']  # TODO: remove this debug cmd

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
        sample_tokens = sample_tokens[:image_limit]

        print('Rendering images for mode %s to folder %s...' % (mode, out_dir))
        for sample_token in tqdm.tqdm(sample_tokens):
            sample = self.nuim.get('sample', sample_token)
            sd_token_camera = sample['key_camera_token']
            sensor = self.nuim.shortcut('sample_data', 'sensor', sd_token_camera)
            sample_cam_name = sensor['channel']

            for mode in modes:
                out_path = os.path.join(out_dir, '%s_%s_%s.jpg' % (sample_token, sample_cam_name, mode))
                if mode == 'annotated':
                    self.nuim.render_image(sd_token_camera, with_annotations=True, out_path=out_path)
                elif mode == 'image':
                    self.nuim.render_image(sd_token_camera, with_annotations=False, out_path=out_path)
                elif mode == 'depth_dense':
                    self.nuim.render_depth(sd_token_camera, mode='dense', out_path=out_path)
                elif mode == 'depth_sparse':
                    self.nuim.render_depth(sd_token_camera, mode='sparse', out_path=out_path)
                elif mode == 'pointcloud':
                    sd_token_lidar = sample['key_lidar_token']
                    self.nuim.render_pointcloud(sd_token_lidar, out_path=out_path)
                elif mode == 'trajectory':
                    self.nuim.render_trajectory(sample_token, out_path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--version', type=str, default='v1.0-val')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--image_limit', type=int, default=100)
    parser.add_argument('--out_dir', type=str, default='~/Downloads/nuImages')
    args = parser.parse_args()

    renderer = ImageRenderer(args.version, args.dataroot, bool(args.verbose))
    renderer.render_images(mode=args.mode, cam_name=args.cam_name, image_limit=args.image_limit, out_dir=args.out_dir)
