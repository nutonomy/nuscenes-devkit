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
        self.nuim = NuImages(version=self.version, dataroot=self.dataroot, verbose=self.verbose)

    def render_images(self,
                      mode: str = 'annotated',
                      image_limit: int = 100,
                      out_dir: str = '~/Downloads/nuImages') -> None:
        """
        Render a random selection of images and save them to disk.
        Note: The images rendered here are keyframes only.
        :param mode: What to render, "annotated" for the annotated image, "raw" for just the image and "depth" for depth
            image.
        :param image_limit: Maximum number of images to render.
        :param out_dir: Folder to render the images to.
        """
        # Check and convert inputs.
        assert mode in ['annotated', 'raw', 'depth']
        out_dir = os.path.expanduser(out_dir)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # Get a random selection of samples.
        sample_tokens = [s['token'] for s in self.nuim.samples]
        random.shuffle(sample_tokens)
        sample_tokens = sample_tokens[:image_limit]

        print('Rendering images for mode %s to folder %s...' % (mode, out_dir))
        for sample_token in tqdm.tqdm(sample_tokens):
            sample = self.nuim.get('sample', sample_token)
            sd_token_camera = sample['key_camera_token']
            out_path = os.path.join(out_dir, '%s_%s.jpg' % (sample_token, mode))

            if mode == 'annotated':
                self.nuim.render_image(sd_token_camera, with_annotations=True, out_path=out_path)
            elif mode == 'raw':
                self.nuim.render_image(sd_token_camera, with_annotations=False, out_path=out_path)
            elif mode == 'depth':
                self.nuim.render_depth(sd_token_camera, out_path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--version', type=str, default='v1.0-val',
                        help='Which version of the dataset to render, e.g. v1.0-val.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print details to stdout.')
    parser.add_argument('--mode', type=str, default='annotated')
    args = parser.parse_args()

    renderer = ImageRenderer(args.version, args.dataroot, bool(args.verbose))
    renderer.render_images(mode=args.mode)
