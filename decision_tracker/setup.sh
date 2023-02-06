#! /bin/bash

export PYTHONPATH=
export PYTHONPATH="${PYTHONPATH}:/home/ahtchow/github/nuscenes-devkit/python-sdk"
python3 -m pip install motmetrics==1.1.3
python3 -m pip install pandas==1.1.0

# Sim link data
ln -s /home/dataset/nuscenes /dataset/nuscenes

#python3 evaluate_nuscenes.py tracks/predicted_tracks_mini.json --output_dir results --dataroot dataset/nuscenes --version v1.0-mini --eval_set mini_train