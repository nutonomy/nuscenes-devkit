# nuScenes devkit
Welcome to the devkit of the [nuScenes](https://www.nuscenes.org) dataset.
![](https://www.nuscenes.org/public/images/road.jpg)

## Overview
- [Changelog](#changelog)
- [Dataset download](#dataset-download)
- [Map expansion](#map-expansion)
- [Devkit setup](#devkit-setup)
- [Getting started](#getting-started)
- [Citation](#citation)

## Changelog
- Jul. 1, 2019: Map expansion pack released.
- Apr. 30, 2019: Devkit v1.0.1: loosen PIP requirements, refine detection challenge, export 2d annotation script. 
- Mar. 26, 2019: Full dataset, paper, & devkit v1.0.0 released. Support dropped for teaser data.
- Dec. 20, 2018: Initial evaluation code released. Devkit folders restructured, which breaks backward compatibility.
- Nov. 21, 2018: RADAR filtering and multi sweep aggregation.
- Oct. 4, 2018: Code to parse RADAR data released.
- Sep. 12, 2018: Devkit for teaser dataset released.

## Dataset download
To download nuScenes you need to go to the [Download page](https://www.nuscenes.org/download), 
create an account and agree to the nuScenes [Terms of Use](https://www.nuscenes.org/terms-of-use).
After logging in you will see multiple archives. 
For the devkit to work you will need to download *all* archives.
Please unpack the archives to the `/data/sets/nuscenes` folder \*without\* overwriting folders that occur in multiple archives.
Eventually you should have the following folder structure:
```
/data/sets/nuscenes
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```
If you want to use another folder, specify the `dataroot` parameter of the NuScenes class (see tutorial).

## Map expansion
In July 2019 we published a map expansion pack with 11 semantic layers (crosswalk, sidewalk, traffic lights, stop lines, lanes, etc.).
To install this expansion, please follow these steps:
- Download the expansion pack from the [Download page](https://www.nuscenes.org/download),
- Move the four .json files to your nuScenes maps folder (e.g. `/data/sets/nuscenes/maps`).
- Get the latest version of the nuscenes-devkit.
- If you already have a previous version of the devkit, update the pip requirements (see [details](https://github.com/nutonomy/nuscenes-devkit/blob/master/setup/installation.md)): `pip install -r setup/requirements.txt`

## Devkit setup
The devkit is tested for Python 3.6 and Python 3.7.
To install Python, please check [here](https://github.com/nutonomy/nuscenes-devkit/blob/master/installation.md#install-python).

Our devkit is available and can be installed via [pip](https://pip.pypa.io/en/stable/installing/) :
```
pip install nuscenes-devkit
```
For an advanced installation, see [installation](https://github.com/nutonomy/nuscenes-devkit/blob/master/setup/installation.md) for detailed instructions.

## Getting started
Please follow these steps to make yourself familiar with the nuScenes dataset:
- Read the [dataset description](https://www.nuscenes.org/overview).
- [Explore](https://www.nuscenes.org/explore/scene-0011/0) the lidar viewer and videos.
- [Download](https://www.nuscenes.org/download) the dataset. 
- Get the [nuscenes-devkit code](https://github.com/nutonomy/nuscenes-devkit).
- Read the [online tutorial](https://www.nuscenes.org/tutorial) or run it yourself using:
```
jupyter notebook $HOME/nuscenes-devkit/python-sdk/tutorial.ipynb
```
- Read the [nuScenes paper](https://www.nuscenes.org/publications) for a detailed analysis of the dataset.
- Run the [map expansion tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_demo.ipynb).
- Take a look at the [experimental scripts](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/scripts).
- For instructions related to the object detection task (results format, classes and evaluation metrics), please refer to [this readme](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md).
- See the [database schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/schema.md) and [annotator instructions](https://github.com/nutonomy/nuscenes-devkit/blob/master/instructions.md).
- See the [FAQs](https://github.com/nutonomy/nuscenes-devkit/blob/master/faqs.md).

## Citation
Please use the following citation when referencing [nuScenes](https://arxiv.org/abs/1903.11027):
```
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```

![](https://www.nuscenes.org/public/images/nuscenes-example.png)
