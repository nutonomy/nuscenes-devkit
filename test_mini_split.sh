#!/bin/bash
dockerfile=$1
set -eux

data_image=registry-local.nutonomy.team:5000/nuscenes/mini_split_data
data_container=mini_split_data
data_volume=mini_split_volume

function clean_up(){
    echo "Cleaning up docker containers and volumes if they already exist"
    docker container stop ${data_container} || { echo "container does not exist"; }
    docker container rm ${data_container} || { echo "container does not exist"; }
    
    docker rm -f test_container || { echo "test container does not exist"; }
    docker volume rm ${data_volume} || { echo "volume does not already exist"; }
}

trap clean_up EXIT

clean_up

echo "Pulling image containing mini split data from registry"
docker pull ${data_image} || { echo "error during docker pull;" exit 1; }

echo "Creating Docker volume from container"
docker run -d --name=${data_container} -v ${data_volume}:/data:ro ${data_image} || { echo "container already running";}

echo "Building image containing nuscenes-devkit"
docker build -t test_mini_split -f ${dockerfile} . || { echo "Failed to build main Docker image"; exit 1; }

docker run --name=test_container -v ${data_volume}:/data \
    -e NUSCENES=/data/nuscenes-v1.0 test_mini_split \
    /bin/bash -c "source activate nuenv && cd python-sdk && python -m unittest"

clean_up
