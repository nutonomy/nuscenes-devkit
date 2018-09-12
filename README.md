# nuScenes dev-kit
Welcome to the dev-kit of the [nuScenes](https://nuscenes.org) dataset. 

## Dataset download
To download nuScenes you need to go to [the Download page](https://nuscenes.org/download), 
create an account and confirm the nuScenes [Terms of Use](https://nuscenes.org/terms-of-use).
After logging in you will see multiple archives for images, pointclouds and meta data. 
For the devkit to work you will need to download *all* archives.
Please unpack the archives to the `/data/nuscenes` folder \*without\* overwriting folders that occur in multiple archives.
Eventually you should have the following folder structure:
```
/data/nuscenes
    maps	-	Large image files (~500 Gigapixel) that depict the drivable surface of the scene.
    samples	-	Sensor data for keyframes
    sweeps	-	Sensor data for intermediate frames
    v0.1	-	JSON tables that include all the meta data and annotations
```
If you want to use another folder, specify the `dataroot` parameter of the NuScenes class below.

## Devkit setup
Download the devkit to your home directory using:
```
cd && git clone https://github.com/nutonomy/nuscenes-devkit.git
```
The dev-kit is tested for Python 3.7. 
We may add backward compatibility in future releases.
To install the required packages, run the following in your favourite virtual environment:
```
pip install -r requirements.txt
```
Also add the `nuscenes_python` directory to your `PYTHONPATH` environmental variable, e.g. by adding the following to your `~/.bashrc`:
```
export PYTHONPATH="${PYTHONPATH}:$HOME/nuscenes-devkit/nuscenes_python"
```

## Getting started
To get started with the nuScenes devkit, please run the tutorial as an IPython notebook:
```
jupyter notebook $HOME/nuscenes-devkit/nuscenes_python/nuscenes_tutorial.ipynb
```
In case you want to avoid downloading and setting up the data, you can also take a look at the same notebook on [Github](https://github.com/nutonomy/nuscenes-devkit/nuscenes_python/nuscenes_tutorial.ipynb).