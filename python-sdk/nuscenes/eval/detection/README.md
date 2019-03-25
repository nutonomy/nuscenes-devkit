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
To this end we will host the nuScenes detection challenge from April 2019.
The results will be presented at the Workshop on Autonomous Driving ([wad.ai](http://wad.ai)) at [CVPR 2019](http://cvpr2019.thecvf.com/).
![nuScenes Singapore Example](https://www.nuscenes.org/public/images/tasks.png)

## General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users apply their method on the test set and submit their results to our evaluation server, which returns the metrics listed below.
* We do not use strata. Instead, we filter annotations and predictions beyond class specific distances.
* Every submission has to provide information on the method and any external / map data used. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* The maximum time window of past sensor data that may be used is 0.5s.
* User needs to limit the number of submitted boxes per sample to 500 to reduce the server load. Submissions with more boxes are automatically rejected.

## Results format
We define a standardized detection results format to allow users to submit results to our evaluation server.
Users need to create single JSON file for the evaluation set, zip the file and upload it to our evaluation server.
The submission JSON includes a dictionary that maps each sample_token to a list of `sample_result` entries.
```
submission {
    sample_token <str>: [sample_result] -- Maps each sample_token to a list of sample_results.
}
```
For the result box we create a new database table called `sample_result`.
The `sample_result` table is designed to mirror the `sample_annotation` table.
This allows for processing of results and annotations using the same tools.
A `sample_result` is defined as follows:
```
sample_result {
    "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
    "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
    "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
    "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
    "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                           See table below for valid attributes for each class, e.g. cycle.with_rider.
                                           Attributes are ignored for classes without attributes.
                                           There are a few cases (0.4%) where attributes are missing also for classes
                                           that should have them. We ignore the predicted attributes for these cases.
}
```
Note that the detection classes may differ from the general nuScenes classes, as detailed below.

## Classes, attributes, and detection ranges
The nuScenes dataset comes with annotations for 23 classes ([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples.
Hence we merge similar classes and remove rare classes.
This results in 10 classes for the detection challenge.
Below we show the table of detection classes and their counterpart in the nuScenes dataset.
For more information on the classes and their frequencies, see [this page](https://www.nuscenes.org/data-annotation).

|   nuScenes detection class|   nuScenes general class                  |
|   ---                     |   ---                                     |
|   void / ignore           |   animal                                  |
|   void / ignore           |   human.pedestrian.personal_mobility      |
|   void / ignore           |   human.pedestrian.stroller               |
|   void / ignore           |   human.pedestrian.wheelchair             |
|   void / ignore           |   movable_object.debris                   |
|   void / ignore           |   movable_object.pushable_pullable        |
|   void / ignore           |   static_object.bicycle_rack              |
|   void / ignore           |   vehicle.emergency.ambulance             |
|   void / ignore           |   vehicle.emergency.police                |
|   barrier                 |   movable_object.barrier                  |
|   bicycle                 |   vehicle.bicycle                         |
|   bus                     |   vehicle.bus.bendy                       |
|   bus                     |   vehicle.bus.rigid                       |
|   car                     |   vehicle.car                             |
|   construction_vehicle    |   vehicle.construction                    |
|   motorcycle              |   vehicle.motorcycle                      |
|   pedestrian              |   human.pedestrian.adult                  |
|   pedestrian              |   human.pedestrian.child                  |
|   pedestrian              |   human.pedestrian.construction_worker    |
|   pedestrian              |   human.pedestrian.police_officer         |
|   traffic_cone            |   movable_object.trafficcone              |
|   trailer                 |   vehicle.trailer                         |
|   truck                   |   vehicle.truck                           |

Below we list which nuScenes classes can have which attributes.
Note that some annotations are missing attributes (0.4% of all sample_annotations).

For each nuScenes detection class, the number of annotations decreases with increasing radius from the ego vehicle, 
but the number of annotations per radius varies by class. Therefore, each class has its own upper bound on evaluated
detection radius.

Below we list nuScene class specific rules for annotation and detection ranges. 

|   nuScenes detection class    |   Attributes                                          | Detection Range (meters)  |
|   ---                         |   ---                                                 |   ---                     |
|   barrier                     |   void                                                |   30                      |
|   traffic_cone                |   void                                                |   30                      |
|   bicycle                     |   cycle.{with_rider, without_rider}                   |   40                      |
|   motorcycle                  |   cycle.{with_rider, without_rider}                   |   40                      |
|   pedestrian                  |   pedestrian.{moving, standing, sitting_lying_down}   |   40                      |
|   car                         |   vehicle.{moving, parked, stopped}                   |   50                      |
|   bus                         |   vehicle.{moving, parked, stopped}                   |   50                      |
|   construction_vehicle        |   vehicle.{moving, parked, stopped}                   |   50                      |
|   trailer                     |   vehicle.{moving, parked, stopped}                   |   50                      |
|   truck                       |   vehicle.{moving, parked, stopped}                   |   50                      |

## Evaluation metrics
Below we define the metrics for the nuScenes detection task.
Our final score is a weighted sum of mean Average Precision (mAP) and several True Positive (TP) metrics.

### Preprocessing
Before running the evaluation code the following pre-processing is done on the data
* All boxes (GT and prediction) are removed if they exceed the class-specific detection range. 
* All bikes and motorcycle boxes (GT and prediction) that fall inside a bike-rack are removed. The reason is that we do not annotate bikes inside bike-racks.  
* All boxes (GT) without lidar or radar points in them are removed. The reason is that we can not guarantee that they are actually visible in the frame. We do not filter the predicted boxes based on number of points.

### Average Precision metric
* **mean Average Precision (mAP)**:
We use the well-known Average Precision metric,
but define a match by considering the 2D center distance on the ground plane rather than intersection over union based affinities. 
Specifically, we match predictions with the ground truth objects that have the smallest center-distance up to a certain threshold.
For a given match threshold we calculate average precision (AP) by integrating the recall vs precision curve for recalls and precisions > 0.1.
We thus exclude operating points with recall or precision < 0.1 from the calculation.  
We finally average over match thresholds of {0.5, 1, 2, 4} meters and compute the mean across classes.

### True Positive errors
Here we define metrics for a set of true positives (TP) that measure translation / scale / orientation / velocity and attribute errors. 
All true positive metrics use a fixed matching threshold of 2m center distance and the matching and scoring happen independently per class.
The metric is averaged over the same recall thresholds as for mAP. 
If a recall value > 0.1 is not achieved, the TP error for that class is set to 1.

Finally we compute the mean over classes.

* **mean Average Translation Error (mATE)**: For each match we compute the translation error as the Euclidean center distance in 2D in meters.
* **mean Average Scale Error (mASE)**: For each match we compute the 3D IOU after aligning orientation and translation.
* **mean Average Orientation Error (mAOE)**: For each match we compute the orientation error as the smallest yaw angle difference between prediction and ground-truth in radians. Orientation error is evaluated at 360 degree for all classes except barriers where it is only evaluated at 180 degrees. Orientation errors for cones are ignored.
* **mean Average Velocity Error (mAVE)**: For each match we compute the absolute velocity error as the L2 norm of the velocity differences in 2D in m/s. Velocity error for barriers and cones are ignored.
* **mean Average Attribute Error (mAAE)**: For each match we compute the attribute error as as *1 - acc*, where acc is the attribute classification accuracy of all the relevant attributes of the ground-truth class. Attribute error for barriers and cones are ignored.

All errors are >= 0, but note that for translation and velocity errors the errors are unbounded, and can be any positive value.

### nuScenes detection score
* **nuScenes detection score (NDS)**:
We consolidate the above metrics by computing a weighted sum: mAP, mATE, mASE, mAOE, mAVE and mAAE.
As a first step we convert the TP errors to TP scores as *x_score = max(1 - x_err, 0.0)*.
We then assign a weight of *5* to mAP and *1* to each of the 5 TP scores and calculate the normalized sum.

## Leaderboard & challenge tracks
Compared to other datasets and challenges, nuScenes will have a single leaderboard for the detection task.
For each submission the leaderboard will list method aspects and evaluation metrics.
Method aspects include input modalities (lidar, radar, vision), use of map data and use of external data.
To enable a fair comparison between methods, the user will be able to filter the methods by method aspects.
 
We define three such filters here.
These filters correspond to the tracks in the nuScenes detection challenge.
Methods will be compared within these tracks and the winners will be decided for each track separately:

* **LIDAR detection track**: 
This track allows only lidar sensor data as input.
No external data or map data is allowed. The only exception is that ImageNet may be used for pre-training (initialization).

* **VISION detection track**: 
This track allows only camera sensor data (images) as input.
No external data or map data is allowed. The only exception is that ImageNet may be used for pre-training (initialization).

* **OPEN detection track**: 
This is where users can go wild.
We allow any combination of sensors, map and external data as long as these are reported. 
