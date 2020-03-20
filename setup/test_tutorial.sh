#!/bin/bash
set -ex

# This script is to be executed inside a Docker container
source activate nuscenes

# Generate python script from Jupyter notebook and then copy into Docker image.
jupyter nbconvert --to python python-sdk/tutorials/nuscenes_basics_tutorial.ipynb || { echo "Failed to convert nuscenes_basics_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/can_bus_tutorial.ipynb || { echo "Failed to convert can_bus_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/map_expansion_tutorial.ipynb || { echo "Failed to convert map_expansion_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/prediction_tutorial.ipynb || { echo "Failed to convert prediction notebook to python script"; exit 1; }

# Remove extraneous matplot inline command and comment out any render* methods.
sed -i.bak "/get_ipython.*/d; s/\(nusc\.render.*\)/#\1/" python-sdk/tutorials/nuscenes_basics_tutorial.py || { echo "error in sed command"; exit 1; }

# Run tutorial
xvfb-run python python-sdk/tutorials/nuscenes_basics_tutorial.py
xvfb-run python python-sdk/tutorials/can_bus_tutorial.py
xvfb-run python python-sdk/tutorials/map_expansion_tutorial.py
xvfb-run python python-sdk/tutorials/prediction_tutorial.py
