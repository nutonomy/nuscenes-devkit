# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for running baseline models on a given nuscenes-split. """
import argparse
import json
import os

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.predict.splits import get_prediction_challenge_split
from nuscenes.predict import PredictHelper
from nuscenes.predict.models.physics import ConstantVelocityHeading, PhysicsOracle


def main(version: str, data_root: str,
         split_name: str, output_dir: str, config_name: str = 'predict_2020_icra') -> None:
    """
    Performs inference for all of the baseline models defined in models.py.
    :param version: Nuscenes dataset version.
    :param split_name: Nuscenes data split name, e.g. train, val, mini_train, etc.
    :param output_dir: Directory where predictions should be stored.
    :param config_name: Name of config file.
    :return: None.
    """

    nusc = NuScenes(version=version, dataroot=data_root)
    helper = PredictHelper(nusc)
    dataset = get_prediction_challenge_split(split_name)
    config = config_factory(config_name)
    oracle = PhysicsOracle(config.seconds, helper)
    cv_heading = ConstantVelocityHeading(config.seconds, helper)

    cv_preds = []
    oracle_preds = []
    for token in dataset:
        cv_preds.append(cv_heading(token).serialize())
        oracle_preds.append(oracle(token).serialize())

    json.dump(cv_preds, open(os.path.join(output_dir, "cv_preds.json"), "w"))
    json.dump(oracle_preds, open(os.path.join(output_dir, "oracle_preds.json"), "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Perform Inference with baseline models.')
    parser.add_argument('--version', help='NuScenes version number.')
    parser.add_argument('--data_root', help='Directory storing NuScenes data.', default='/data/sets/nuscenes')
    parser.add_argument('--split_name', help='Data split to run inference on.')
    parser.add_argument('--output_dir', help='Directory to store output files.')
    parser.add_argument('--config_name', help='Config file to use.', default='predict_2020_icra')

    args = parser.parse_args()
    main(args.version, args.split_name, args.output_dir, args.config_name)
