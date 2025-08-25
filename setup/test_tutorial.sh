#!/bin/bash
set -ex

RUNNING_IN_CI=False

# Parse arguments
for arg in "$@"; do
    case $arg in
        --ci)
        RUNNING_IN_CI=True
        ;;
    esac
done

# Generate python script from Jupyter notebook and then copy into Docker image.
jupyter nbconvert --to python python-sdk/tutorials/nuscenes_tutorial.ipynb || { echo "Failed to convert nuscenes_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/nuimages_tutorial.ipynb || { echo "Failed to convert nuimages_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/can_bus_tutorial.ipynb || { echo "Failed to convert can_bus_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/map_expansion_tutorial.ipynb || { echo "Failed to convert map_expansion_tutorial notebook to python script"; exit 1; }
jupyter nbconvert --to python python-sdk/tutorials/prediction_tutorial.ipynb || { echo "Failed to convert prediction notebook to python script"; exit 1; }

# Remove extraneous matplot inline command and comment out any render* methods.
sed -i.bak "/get_ipython.*/d; s/\(nusc\.render.*\)/#\1/" python-sdk/tutorials/nuscenes_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(nusc\.render.*\)/#\1/" python-sdk/tutorials/nuimages_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(nusc_can.plot.*\)/#\1/"  python-sdk/tutorials/can_bus_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(^plt.*\)/#\1/"  python-sdk/tutorials/can_bus_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(fig, ax.*\)/#\1/"  python-sdk/tutorials/map_expansion_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(nusc_map.render.*\)/#\1/"  python-sdk/tutorials/map_expansion_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(ego_poses = .*\)/#\1/"  python-sdk/tutorials/map_expansion_tutorial.py || { echo "error in sed command"; exit 1; }
sed -i.bak "/get_ipython.*/d; s/\(plt.imshow.*\)/#\1/"  python-sdk/tutorials/prediction_tutorial.py || { echo "error in sed command"; exit 1; }

# Replace dataset path for running on CI
# Use "data/sets/nuscenes" instead of "/data/sets/nuscenes"
# Use "data/sets/nuimages" instead of "/data/sets/nuimages"
if [[ ${RUNNING_IN_CI} == "True" ]]; then
    echo "Running in CI. Replacing path to dataroot as: data/sets/nuimages"
    sed -i 's/\/data\/sets\/nuscenes/data\/sets\/nuscenes/g' python-sdk/tutorials/nuscenes_tutorial.py
    sed -i 's/\/data\/sets\/nuscenes/data\/sets\/nuscenes/g' python-sdk/tutorials/nuimages_tutorial.py
    sed -i 's/\/data\/sets\/nuimages/data\/sets\/nuimages/g' python-sdk/tutorials/nuimages_tutorial.py
    sed -i 's/\/data\/sets\/nuscenes/data\/sets\/nuscenes/g' python-sdk/tutorials/can_bus_tutorial.py
    sed -i 's/\/data\/sets\/nuscenes/data\/sets\/nuscenes/g' python-sdk/tutorials/map_expansion_tutorial.py
    sed -i 's/\/data\/sets\/nuscenes/data\/sets\/nuscenes/g' python-sdk/tutorials/prediction_tutorial.py
fi

# Run tutorial
xvfb-run python python-sdk/tutorials/nuscenes_tutorial.py
# xvfb-run python python-sdk/tutorials/nuimages_tutorial.py # skip until PR-440 merged
xvfb-run python python-sdk/tutorials/can_bus_tutorial.py
xvfb-run python python-sdk/tutorials/map_expansion_tutorial.py
xvfb-run python python-sdk/tutorials/prediction_tutorial.py
