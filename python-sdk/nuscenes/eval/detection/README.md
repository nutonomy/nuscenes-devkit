# nuScenes detection task
In this document we present the rules, results format, classes, evaluation metrics and challenge tracks of the nuScenes detection task.

## Overview
- [Introduction](#introduction)
- [General rules](#general-rules)
- [Results format](#results-format)
- [Classes and attributes](#classes-and-attributes)
- [Evaluation metrics](#evaluation-metrics)
- [Leaderboard & challenge tracks](#leaderboard--challenge-tracks)

## Introduction
The primary task of the nuScenes dataset is 3D object detection.
The goal of 3D object detection is to place a tight 3D bounding box around every object.
Object detection is the backbone for autonomous vehicles, as well as many other applications.
Our goal is to provide a benchmark to measure performance and advance the state-of-the-art in autonomous driving.
To this end we will host the nuScenes detection challenge from March 2019.
The results will be presented at the Workshop on Autonomous Driving ([wad.ai](http://wad.ai)) at [CVPR 2019](http://cvpr2019.thecvf.com/).
![nuScenes Singapore Example](https://www.nuscenes.org/public/images/tasks.png)

## General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users apply their method on the test set and submit their results to our evaluation server, which returns the metrics listed below.
* We do not use strata (cf. easy / medium / hard in KITTI). We only filter annotations and predictions beyond 40m distance.
* Every submission has to provide information on the method and any external / map data used. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* The maximum time window of past sensor data that may be used is 0.5s.
* User needs to limit the number of submitted boxes per sample to 500 to reduce the server load. Submissions with more boxes are automatically rejected.

## Results format
We define a standardized detection results format to allow users to submit results to our evaluation server.
Users need to create single JSON file for the evaluation set, zip the file and upload it to our evaluation server.
The submission JSON includes a dictionary that maps each sample to its result boxes:
```
submission {
    "all_sample_results": <dict>      -- Maps each sample_token to a list of sample_results.
}
```
For the result box we create a new database table called `sample_result`.
The `sample_result` table is designed to mirror the `sample_annotation` table.
This allows for processing of results and annotations using the same tools.
A `sample_result` is defined as follows:
```
sample_result {
    "sample_token":       <str>       -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":        <float> [3] -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
    "size":               <float> [3] -- Estimated bounding box size in m: width, length, height.
    "rotation":           <float> [4] -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":           <float> [3] -- Estimated bounding box velocity in m/s in the global frame: vx, vy, vz. Set values to nan to ignore.
    "detection_name":     <str>       -- The predicted class for this sample_result, e.g. car, pedestrian.
    "detection_score":    <float>     -- Object prediction score between 0 and 1 for the class identified by detection_name.
    "attribute_scores":   <float> [8] -- Attribute prediction scores between 0 and 1 for the attributes: 
        cycle.with_rider, cycle.without_rider, pedestrian.moving, pedestrian.sitting_lying_down, pedestrian.standing, vehicle.moving, vehicle.parked, vehicle.stopped. 
        If any score is set to -1, the attribute error is 1.
}
```
Note that the detection classes may differ from the general nuScenes classes, as detailed below.

## Classes and attributes
The nuScenes dataset comes with annotations for 25 classes ([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples.
Hence we merge similar classes and remove classes that have less than 1000 samples in the teaser dataset.
This results in 10 classes for the detection challenge.
The full dataset will have 10x more samples for each class.
Below we show the table of detection classes and their counterpart in the general nuScenes dataset.
Double quotes (") indicate that a cell has the same class as the cell above.

|   nuScenes detection class|   nuScenes general class                  |   Annotations         |
|   ---                     |   ---                                     |   ---                 |
|   void / ignore           |   animal                                  |   6                   |
|   void / ignore           |   human.pedestrian.personal_mobility      |   24                  |
|   void / ignore           |   human.pedestrian.stroller               |   40                  |
|   void / ignore           |   human.pedestrian.wheelchair             |   5                   |
|   void / ignore           |   movable_object.debris                   |   500                 |
|   void / ignore           |   movable_object.pushable_pullable        |   583                 |
|   void / ignore           |   static_object.bicycle_rack              |   192                 |
|   void / ignore           |   vehicle.emergency.ambulance             |   19                  |
|   void / ignore           |   vehicle.emergency.police                |   88                  |
|   barrier                 |   movable_object.barrier                  |   18,449              |
|   bicycle                 |   vehicle.bicycle                         |   1,685               |
|   bus                     |   vehicle.bus.bendy                       |   98                  |
|   bus                     |   vehicle.bus.rigid                       |   1,115               |
|   car                     |   vehicle.car                             |   32,497              |
|   construction_vehicle    |   vehicle.construction                    |   1,889               |
|   motorcycle              |   vehicle.motorcycle                      |   1,975               |
|   pedestrian              |   human.pedestrian.adult                  |   20,510              |
|   pedestrian              |   human.pedestrian.child                  |   15                  |
|   pedestrian              |   human.pedestrian.construction_worker    |   2,400               |
|   pedestrian              |   human.pedestrian.police_officer         |   39                  |
|   traffic_cone            |   movable_object.trafficcone              |   7,197               |
|   trailer                 |   vehicle.trailer                         |   2,383               |
|   truck                   |   vehicle.truck                           |   8,243               |

Below we list which nuScenes classes can have which attributes.
Note that for classes with attributes exactly one attribute must be active.

|   Attributes                                          |   nuScenes detection class    |
|   ---                                                 |   ---                         |
|   void                                                |   barrier                     |
|   void                                                |   traffic_cone                |
|   cycle.{with_rider, without_rider}                   |   bicycle                     |
|   cycle.{with_rider, without_rider}                   |   motorcycle                  |
|   pedestrian.{moving, standing, sitting_lying_down}   |   pedestrian                  |
|   vehicle.{moving, parked, stopped}                   |   car                         |
|   vehicle.{moving, parked, stopped}                   |   bus                         |
|   vehicle.{moving, parked, stopped}                   |   construction_vehicle        |
|   vehicle.{moving, parked, stopped}                   |   trailer                     |
|   vehicle.{moving, parked, stopped}                   |   truck                       |

## Evaluation metrics
Below we define the metrics for the nuScenes detection task.
Our final score is a weighted sum of mean Average Precision (mAP) and several True Positive (TP) metrics.

### Average Precision metric
* **mean Average Precision (mAP)**:
We use the well-known Average Precision metric as in KITTI,
but define a match by considering the 2D center distance on the ground plane rather than intersection over union based affinities. 
Specifically, we match predictions with the ground truth objects that have the smallest center-distance up to a certain threshold.
For a given match threshold we calculate average precision (AP) by integrating recall between 0.1 and 1.
Note that we pick *0.1* as the lowest recall threshold, as precision values at recall < 0.1 tends to be noisy.  
If a recall value is not achieved, its precision is set to 0.
We finally average over match thresholds of {0.5, 1, 2, 4} meters and compute the mean across classes.

### True Positive metrics
Here we define metrics for a set of true positives (TP) that measure translation / scale / orientation / velocity and attribute errors.
All true positive metrics use a fixed matching threshold of 2m center distance and the matching and scoring happen independently per class.
The metric is averaged over the same recall thresholds as for mAP.
To bring all TP metrics into a similar range, we bound each metric to be below an arbitrarily selected metric bound and then normalize to be in *[0, 1]*.
The metric bound is *0.5* for mATE, *0.5* for mASE, *π/2* for mAOE, *1.5* for mAVE, *1.0* for mAAE.
If a recall value is not achieved for a certain range, the error is set to 1 in that range.
This mechanism enforces that submitting only the top *k* boxes does not result in a lower error.
This is particularly important as some TP metrics may decrease with increasing recall values. 
Finally we compute the mean over classes.
* **mean Average Translation Error (mATE)**: For each match we compute the translation error as the Euclidean center distance in 2D in meters.
* **mean Average Scale Error (mASE)**: For each match we compute the 3D IOU after aligning orientation and translation.
* **mean Average Orientation Error (mAOE)**: For each match we compute the orientation error as the smallest yaw angle difference between prediction and ground-truth in radians.
* **mean Average Velocity Error (mAVE)**: For each match we compute the absolute velocity error as the L2 norm of the velocity differences in 2D in m/s.
* **mean Average Attribute Error (mAAE)**: For each match we compute the attribute error as as *1 - acc*, where acc is the attribute classification accuracy of all the relevant attributes of the ground-truth class. The attribute error is ignored for classes without attributes.

### Weighted sum metric
* **Weighted sum**: We compute the weighted sum of the above metrics: mAP, mATE, mASE, mAOE, mAVE and mAAE.
For each error metric x (excl. mAP), we use *1 - x*.
We assign a weight of *5* to mAP and *1* to the 5 TP metrics.
Then we normalize by 10.

## Leaderboard & challenge tracks
Compared to other datasets and challenges, nuScenes will have a single leaderboard for the detection task.
For each submission the leaderboard will list method aspects and evaluation metrics.
Method aspects include input modalities (lidar, radar, vision), use of map data and use of external data.
To enable a fair comparison between methods, the user will be able to filter the methods by method aspects.
The user can also filter the metrics that should be taken into account for the weighted sum metric. 

We define two such filters here.
These filters correspond to the tracks in the nuScenes detection challenge.
Methods will be compared within these tracks and the winners will be decided for each track separately:

* **LIDAR detection track**: 
This track allows only lidar sensor data as input.
It is supposed to be easy to setup and support legacy lidar methods.
No external data or map data is allowed.

* **Open detection track**: 
This is where users can go wild.
We allow any combination of sensors, map and external data as long as these are reported. 

Note that for both tracks mAVE and mAAE will have 0 weight.