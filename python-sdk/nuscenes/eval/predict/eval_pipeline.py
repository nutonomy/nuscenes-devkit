# nuScenes dev-kit.
# Code written by Freddy Boulton, 2020.
""" Script for generating and evaluating a submission to the nuscenes prediction challenge. """
import argparse

from nuscenes.eval.predict import do_inference, compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Inference with baseline models.')
    parser.add_argument('--version', help='NuScenes version number.')
    parser.add_argument('--data_root', help='Root directory for NuScenes json files.', default='/data/sets/nuscenes')
    parser.add_argument('--split_name', help='Data split to run inference on.')
    parser.add_argument('--output_dir', help='Directory to store output file.')
    parser.add_argument('--submission_name', help='Name of the submission to use for the results file.')
    parser.add_argument('--config_name', help='Name of the config file to use', default='predict_2020_icra.json')

    args = parser.parse_args()
    do_inference.main(args.version, args.data_root, args.split_name, args.output_dir, args.submission_name,
                      args.config_name)
    compute_metrics.main(args.version, args.data_root, args.output_dir, args.submission_name, args.config_name)
