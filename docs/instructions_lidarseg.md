# nuScenes-lidarseg Annotator Instructions

# Overview
- [Introduction](#introduction)
- [General Instructions](#general-instructions)
- [Detailed Instructions](#detailed-instructions)
- [Classes](#classes)

# Introduction
In nuScenes-lidarseg, we annotate every point in the lidar pointcloud with a semantic label. 
All the labels from nuScenes are carried over into nuScenes-lidarseg; in addition, more ["stuff" (background) classes](#classes) have been included.
Thus, nuScenes-lidarseg contains both foreground classes (pedestrians, vehicles, cyclists, etc.) and background classes (driveable surface, nature, buildings, etc.).


# General Instructions
 - Label each point with a class. 
 - Use the camera images to facilitate, check and validate the labels.
 - Each point belongs to only one class, i.e., one class per point.

 
# Detailed Instructions  
+ **Extremities** such as vehicle doors, car mirrors and human limbs should be assigned the same label as the object. 
Note that in contrast to the nuScenes 3d cuboids, the lidarseg labels include car mirrors and antennas.
+ **Minimum number of points** 
    + An object can have as little as **one** point. 
    In such cases, that point should only be labeled if it is certain that the point belongs to a class 
    (with additional verification by looking at the corresponding camera frame). 
    Otherwise, the point should be labeled as `static.other`.  
+ **Other static object vs noise.**
    + **Other static object:** Points that belong to some physical object, but are not defined in our taxonomy.  
    + **Noise:** Points that do not correspond to physical objects or surfaces in the environment
    (e.g. noise, reflections, dust, fog, raindrops or smoke).
+ **Terrain vs other flat.**
    + **Terrain:** Grass, all kinds of horizontal vegetation, soil or sand. These areas are not meant to be driven on. 
    This label includes a possibly delimiting curb. 
    Single grass stalks do not need to be annotated and get the label of the region they are growing on.
    + Short bushes / grass with **heights of less than 20cm**, should be labeled as terrain. 
    Similarly, tall bushes / grass which are higher than 20cm should be labeled as vegetation.
    + **Other flat:** Horizontal surfaces which cannot be classified as ground plane / sidewalk / terrain, e.g., water.
+ **Terrain vs sidewalk**
    + **Terrain:** See above.
    + **Sidewalk:** A sidewalk is a walkway designed for pedestrians and / or cyclists. Sidewalks are always paved.


# Classes
The following classes are in **addition** to the existing ones in nuScenes:  

| Label ID |  Label | Short Description |
| --- | --- | --- |
| 0 | [`noise`](#1-noise-class-0) | Any lidar return that does not correspond to a physical object, such as dust, vapor, noise, fog, raindrops, smoke and reflections. |
| 24 | [`flat.driveable_surface`](#2-flatdriveable_surface-class-24) | All paved or unpaved surfaces that a car can drive on with no concern of traffic rules. |
| 25 | [`flat.sidewalk`](#3-flatsidewalk-class-25) | Sidewalk, pedestrian walkways, bike paths, etc. Part of the ground designated for pedestrians or cyclists. Sidewalks do **not** have to be next to a road. |
| 26 | [`flat.terrain`](#4-flatterrain-class-26) | Natural horizontal surfaces such as ground level horizontal vegetation (< 20 cm tall), grass, rolling hills, soil, sand and gravel. |
| 27 | [`flat.other`](#5-flatother-class-27) | All other forms of horizontal ground-level structures that do not belong to any of driveable_surface, curb, sidewalk and terrain. Includes elevated parts of traffic islands, delimiters, rail tracks, stairs with at most 3 steps and larger bodies of water (lakes, rivers). |
| 28 | [`static.manmade`](#6-staticmanmade-class-28) | Includes man-made structures but not limited to: buildings, walls, guard rails, fences, poles, drainages, hydrants, flags, banners, street signs, electric circuit boxes, traffic lights, parking meters and stairs with more than 3 steps.  |
| 29 | [`static.vegetation`](#7-staticvegetation-class-29) | Any vegetation in the frame that is higher than the ground, including bushes, plants, potted plants, trees, etc. Only tall grass (> 20cm) is part of this, ground level grass is part of `flat.terrain`.|
| 30 | [`static.other`](#8-staticother-class-30) | Points in the background that are not distinguishable. Or objects that do not match any of the above labels. |
| 31 | [`vehicle.ego`](#9-vehicleego-class-31) | The vehicle on which the cameras, radar and lidar are mounted, that is sometimes visible at the bottom of the image. |

## Examples of classes
Below are examples of the classes added in nuScenes-lidarseg.
For simplicity, we only show lidar points which are relevant to the class being discussed.


### 1. noise (class 0)
![noise_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/0_scene-0053_CAM_FRONT_LEFT_1532402428104844_crop.jpg)
![noise_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/0_scene-0163_CAM_FRONT_LEFT_1526915289904917_crop.jpg) 
![noise_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/0_scene-0207_CAM_BACK_LEFT_1532621922197405_crop.jpg)
![noise_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/0_scene-0635_CAM_FRONT_1537296086862404_crop.jpg)

[Top](#classes)


### 2. flat.driveable_surface (class 24)
![driveable_surface_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/24_206_CAM_BACK.jpg)
![driveable_surface_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/24_250_CAM_FRONT.jpg)
![driveable_surface_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/24_9750_CAM_FRONT.jpg)
![driveable_surface_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/24_10000_CAM_BACK.jpg)

[Top](#classes)


### 3. flat.sidewalk (class 25)
![sidewalk_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/25_90_CAM_FRONT_LEFT.jpg)
![sidewalk_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/25_13250_CAM_FRONT_LEFT.jpg)
![sidewalk_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/25_280_CAM_FRONT_LEFT.jpg)
![sidewalk_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/25_680_CAM_FRONT_LEFT.jpg)

[Top](#classes)


### 4. flat.terrain (class 26)
![terrain_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/26_11750_CAM_BACK_RIGHT.jpg)
![terrain_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/26_10700_CAM_BACK_LEFT.jpg)
![terrain_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/26_886_CAM_BACK_LEFT.jpg)
![terrain_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/26_1260_CAM_BACK_LEFT.jpg)

[Top](#classes)


### 5. flat.other (class 27)
![flat_other_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/27_2318_CAM_FRONT.jpg)
![flat_other_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/27_3750_CAM_FRONT_RIGHT.jpg)
![flat_other_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/27_1230_CAM_FRONT.jpg)
![flat_other_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/27_1380_CAM_FRONT.jpg)

[Top](#classes)


### 6. static.manmade (class 28)
![manmade_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/28_13850_CAM_FRONT.jpg)
![manmade_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/28_15550_CAM_FRONT.jpg)
![manmade_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/28_5009_CAM_FRONT.jpg)
![manmade_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/28_5501_CAM_BACK.jpg)

[Top](#classes)


### 7. static.vegetation (class 29)
![vegetation_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/29_650_CAM_FRONT_LEFT.jpg)
![vegetation_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/29_3650_CAM_FRONT.jpg)
![vegetation_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/29_5610_CAM_BACK_RIGHT.jpg)
![vegetation_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/29_5960_CAM_FRONT_RIGHT.jpg)

[Top](#classes)


### 8. static.other (class 30)
![static_other_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/30_scene-0031_CAM_BACK_LEFT_1531886230947423.jpg)
![static_other_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/30_scene-0032_CAM_BACK_RIGHT_1531886262027893.jpg)
![static_other_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/30_scene-0160_CAM_BACK_LEFT_1533115303947423.jpg)
![static_other_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/30_scene-0166_CAM_BACK_RIGHT_1526915380527813.jpg)

[Top](#classes)


### 9. vehicle.ego (class 31)
Points on the ego vehicle generally arise due to self-occlusion, in which some lidar beams hit the ego vehicle.
When the pointcloud is projected into a chosen camera image, the devkit removes points which are less than 
1m in front of the camera to prevent such points from cluttering the image. Thus, users will not see points
belonging to `vehicle.ego` projected onto the camera images when using the devkit. To give examples, of the
`vehicle.ego` class, the bird's eye view (BEV) is used instead:

![ego_1](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/31_479_BEV.jpg)
![ego_2](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/31_11200_BEV.jpg)
![ego_3](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/31_14500_BEV.jpg)
![ego_4](https://www.nuscenes.org/public/images/taxonomy_imgs/lidarseg/31_24230_BEV.jpg)

[Top](#classes)
