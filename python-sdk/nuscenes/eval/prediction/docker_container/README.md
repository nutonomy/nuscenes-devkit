# nuScenes Prediction Challenge Docker Submission Process

We will ask at least the top five teams ranked on the leader board to submit their code to us so we can
evaluate their model on the private test set. To ensure reproducibility, we will run their code
in a Docker container. This document explains how you can run your model inside the Docker container we
will use. If you follow these steps, then if it runs on your machine, it will run on ours. 

## Requirements

- Docker version >= 19 (We tested with 19.03.7)
- machine with GPU card, nvidia drivers and CUDA 10.1 (for GPU support)
- nvidia-docker https://github.com/NVIDIA/nvidia-docker. You can use generic docker image if you don't need GPU support
- nuScenes dataset
- cloned nuScenes-devkit repo https://github.com/nutonomy/nuscenes-devkit

## Usage
- Pull docker image. For CUDA 10.1 use:
```
docker pull nuscenes/dev-challenge:10.1
```
- Pull docker image. For CUDA 9.2 use:
```
docker pull nuscenes/dev-challenge:9.2
```


- Create directory for output data
```
mkdir -p ~/Documents/submissions
```

- Create home directory for the image (needed if you need to install extra packages).
```
mkdir -p ~/Desktop/home_directory
```

- Modify `do_inference.py` in `nuscenes/eval/prediction/submission` to 
run your model. Place your model weights in
`nuscenes/eval/prediction/submission` as well. If you need to install any
extra packages, add them (along with the **exact** version number) to
`nuscenes/eval/prediction/submission/extra_packages.txt`.

- Run docker container
```
cd <NUSCENES ROOT DIR>
docker run [ --gpus all ] -ti --rm \
   -v <PATH TO NUSCENES DATASET>:/data/sets/nuscenes \
   -v <PATH TO nuScenes-devkit ROOT DIR>/python-sdk:/nuscenes-dev/python-sdk \
   -v <PATH TO nuscenes/eval/prediction/submission>:/nuscenes-dev/prediction \
   -v ~/Documents/:/nuscenes-dev/Documents \
   -v ~/Desktop/home_directory:/home/<username>
   <name of image>
```

NOTE: The docker image uses 1000:1000 uid:gid
If this is different from your local setup, you may want to add this options into `docker run` command
```
--user `id -u`:`id -g` -v /etc/passwd:/etc/passwd -v /etc/group:/etc/group
```

- Execute your script inside docker container
```
source activate /home/nuscenes/.conda/envs/nuscenes

pip install -r submission/extra_packages.txt

# Use v1.0-trainval and split_name val to run on the entire val set

python do_inference.py --version v1.0-mini \
    --data_root /data/sets/nuscenes \
    --split_name mini_val \
    --output_dir /nuscenes-dev/Documents/submissions \
    --submission_name <submission-name>
```
