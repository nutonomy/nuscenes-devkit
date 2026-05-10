"""
Validate the local structure of a nuScenes dataset directory.

This script checks whether the expected metadata files for a nuScenes version
exist under the provided dataroot. It is intended as a lightweight helper for
users who want to verify that the dataset was extracted into the expected
folder layout before using the devkit.

Example usage:

    python python-sdk/nuscenes/scripts/validate_dataset_structure.py \
        --dataroot /data/sets/nuscenes \
        --version v1.0-mini
"""

import argparse
from pathlib import Path
from typing import List


REQUIRED_METADATA_FILES: List[str] = [
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
]


def validate_dataset_structure(dataroot: str, version: str) -> None:
    """
    Validate that required nuScenes metadata files exist.

    :param dataroot: Path to the nuScenes dataset root.
    :param version: Dataset version, for example 'v1.0-mini'.
    """
    root = Path(dataroot).expanduser().resolve()
    version_dir = root / version

    print(f"Checking nuScenes dataset structure")
    print(f"Dataroot: {root}")
    print(f"Version: {version}")
    print(f"Expected metadata directory: {version_dir}")

    if not root.exists():
        raise FileNotFoundError(f"Dataroot does not exist: {root}")

    if not version_dir.exists():
        raise FileNotFoundError(
            f"Version directory does not exist: {version_dir}\n"
            f"Expected layout example: {root}/{version}/sample.json"
        )

    missing_files = [
        filename for filename in REQUIRED_METADATA_FILES
        if not (version_dir / filename).exists()
    ]

    if missing_files:
        print("\nMissing metadata files:")
        for filename in missing_files:
            print(f"  - {version_dir / filename}")

        raise FileNotFoundError(
            "\nDataset structure validation failed. "
            "Check that the dataset archive was fully extracted and that the "
            "version folder is not nested one level too deep."
        )

    print("\nAll required metadata files were found.")
    print("Dataset structure looks valid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local structure of a nuScenes dataset directory."
    )

    parser.add_argument(
        "--dataroot",
        default="/data/sets/nuscenes",
        help="Path to the nuScenes dataset root.",
    )

    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes dataset version to validate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dataset_structure(dataroot=args.dataroot, version=args.version)