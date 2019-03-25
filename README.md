# nuScenes devkit
Welcome to the devkit of the [nuScenes](https://www.nuscenes.org) dataset.
 
![](https://www.nuscenes.org/public/images/road.jpg)

## Overview
- [Changelog](#changelog)
- [Dataset download](#dataset-download)
- [Devkit setup](#devkit-setup)
- [Tutorial](#tutorial)
- [Object detection task](#object-detection-task)
- [Backward compatibility](#backward-compatibility)

## Changelog
- Mar. 26, 2019: Full dataset & devkit v1.0.0 released. Support dropped for teaser data.
- Dec. 20, 2018: Initial evaluation code released. Devkit folders restructured. 
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
    maps	-	Large image files (~500 Gigapixel) that depict the drivable surface and sidewalks in the scene.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```
If you want to use another folder, specify the `dataroot` parameter of the NuScenes class (see tutorial).

## Devkit setup
The devkit is tested for Python 3.6 and Python 3.7. To install python, please check [here](https://github.com/nutonomy/nuscenes-devkit/blob/master/installation.md#install-python).

Our devkit is available and can be installed via pip:
```
pip install nuscenes-devkit
```
If you don't have pip, please check [here](https://pip.pypa.io/en/stable/installing/) to install pip.

For an advanced installation, see [installation](https://github.com/nutonomy/nuscenes-devkit/blob/master/setup/installation.md) for detailed instructions.

## Tutorial
To get started with the nuScenes devkit, please run the tutorial as an IPython notebook:
```
jupyter notebook $HOME/nuscenes-devkit/python-sdk/tutorial.ipynb
```
In case you want to avoid downloading and setting up the data, you can also take a look at the [rendered notebook on nuScenes.org](https://www.nuscenes.org/tutorial).
To learn more about the dataset, go to [nuScenes.org](https://www.nuscenes.org) or take a look at the [database schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/schema.md) and [annotator instructions](https://github.com/nutonomy/nuscenes-devkit/blob/master/instructions.md).

## Object detection task
For instructions related to the object detection task (results format, classes and evaluation metrics), please refer to [this readme](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md).

## Backward compatibility
- Mar. 26, 2019: With the full dataset release we drop support for the code and data of the teaser release. Several changes to the map table and map files break backward compatibility.
- Dec. 20, 2018: We restructured the nuscenes-devkit code, which breaks backward compatibility.
  The new structure has a top-level package `nuscenes` which contains packages `eval`, `export` and `utils`.
  Therefore, existing imports from `nuscenes_utils` should be replaced by `nuscenes.nuscenes`.


![](https://www.nuscenes.org/public/images/nuscenes-example.png)
