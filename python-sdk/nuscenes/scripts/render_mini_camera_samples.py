"""
Render camera samples from the nuScenes v1.0-mini dataset.

This script provides a small, beginner-friendly example for loading the
nuScenes mini split, selecting a sample, and rendering all available camera
views for that sample.

Example usage:

    python python-sdk/nuscenes/scripts/render_mini_camera_samples.py \
        --dataroot /data/sets/nuscenes \
        --version v1.0-mini \
        --output-dir /tmp/nuscenes-mini-cameras

The output directory will contain one rendered image per camera channel.
"""

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes


CAMERA_CHANNELS: List[str] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def render_camera_samples(
    dataroot: str,
    version: str,
    output_dir: str,
    sample_index: int,
) -> None:
    """
    Render all available camera channels for a selected nuScenes sample.

    :param dataroot: Path to the nuScenes dataset root.
    :param version: Dataset version, for example 'v1.0-mini'.
    :param output_dir: Directory where rendered images will be saved.
    :param sample_index: Index of the sample to render.
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    if sample_index < 0 or sample_index >= len(nusc.sample):
        raise ValueError(
            f"sample_index must be between 0 and {len(nusc.sample) - 1}, "
            f"but got {sample_index}."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample = nusc.sample[sample_index]
    print(f"Rendering sample index: {sample_index}")
    print(f"Sample token: {sample['token']}")
    print(f"Scene token: {sample['scene_token']}")
    print(f"Timestamp: {sample['timestamp']}")

    for channel in CAMERA_CHANNELS:
        if channel not in sample["data"]:
            print(f"Skipping missing camera channel: {channel}")
            continue

        sample_data_token = sample["data"][channel]
        sample_data = nusc.get("sample_data", sample_data_token)

        out_file = output_path / f"{sample_index:04d}_{channel}.png"

        print(f"Rendering {channel}")
        print(f"  sample_data token: {sample_data_token}")
        print(f"  source file: {os.path.join(dataroot, sample_data['filename'])}")
        print(f"  output file: {out_file}")

        nusc.render_sample_data(sample_data_token, out_path=str(out_file))
        plt.close("all")

    print(f"Done. Rendered camera images are saved in: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render camera samples from the nuScenes mini dataset."
    )

    parser.add_argument(
        "--dataroot",
        default="/data/sets/nuscenes",
        help="Path to the nuScenes dataset root.",
    )

    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes dataset version to load.",
    )

    parser.add_argument(
        "--output-dir",
        default="/tmp/nuscenes-mini-cameras",
        help="Directory where rendered camera images will be saved.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the sample to render.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    render_camera_samples(
        dataroot=args.dataroot,
        version=args.version,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
    )