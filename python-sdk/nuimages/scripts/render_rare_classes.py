# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import argparse
import random
from collections import defaultdict
from typing import Dict, Any, List

from nuimages.nuimages import NuImages
from nuimages.scripts.render_images import render_images


def render_rare_classes(nuim: NuImages,
                        render_args: Dict[str, Any],
                        filter_categories: List[str] = None,
                        max_frequency: float = 0.1) -> None:
    """
    Wrapper around render_images() that renders images with rare classes.
    :param nuim: NuImages instance.
    :param render_args: The render arguments passed on to the render function. See render_images().
    :param filter_categories: Specify a list of object_ann category names.
        Every sample that is rendered must contain annotations of any of those categories.
        Filter_categories are a applied on top of the frequency filering.
    :param max_frequency: The maximum relative frequency of the categories, at least one of which is required to be
        present in the image. E.g. 0.1 indicates that one of the classes that account for at most 10% of the annotations
        is present.
    """
    # Checks.
    assert 'filter_categories' not in render_args.keys(), \
        'Error: filter_categories is a separate argument and should not be part of render_args!'
    assert 0 <= max_frequency <= 1, 'Error: max_frequency must be a ratio between 0 and 1!'

    # Compute object class frequencies.
    object_freqs = defaultdict(lambda: 0)
    for object_ann in nuim.object_ann:
        category = nuim.get('category', object_ann['category_token'])
        object_freqs[category['name']] += 1

    # Find rare classes.
    total_freqs = len(nuim.object_ann)
    filter_categories_freq = sorted([k for (k, v) in object_freqs.items() if v / total_freqs <= max_frequency])
    assert len(filter_categories_freq) > 0, 'Error: No classes found with the specified max_frequency!'
    print('The rare classes are: %s' % filter_categories_freq)

    # If specified, additionally filter these categories by what was requested.
    if filter_categories is None:
        filter_categories = filter_categories_freq
    else:
        filter_categories = list(set(filter_categories_freq).intersection(set(filter_categories)))
        assert len(filter_categories) > 0, 'Error: No categories left after applying filter_categories!'

    # Call render function.
    render_images(nuim, filter_categories=filter_categories, **render_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--seed', type=int, default=42)  # Set to 0 to disable.
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuimages')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=100)
    parser.add_argument('--max_frequency', type=float, default=0.1)
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
    _render_args = {
        'mode': args.mode,
        'cam_name': args.cam_name,
        'sample_limit': args.sample_limit,
        'out_type': args.out_type,
        'out_dir': args.out_dir
    }
    render_rare_classes(nuim_, _render_args, filter_categories=args.filter_categories, max_frequency=args.max_frequency)
