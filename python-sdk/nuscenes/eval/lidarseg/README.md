# nuScenes lidar segmentation task
![nuScenes lidar segementation logo](https://www.nuscenes.org/public/images/tasks.png) -----> @TODO

## Overview
- [Introduction](#introduction)
- [Participation](#participation)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Classes](#classes)
- [Evaluation metrics](#evaluation-metrics)

## Introduction
Here we define the lidar segmentation task on nuScenes.
The goal of this task is to predict the category of every point in a set of point clouds. There are #TODO XX categories.

## Participation
The nuScenes lidarseg segementation [evaluation server](http://evalai.cloudcv.org/web/challenges/challenge-page/356 ---> @TODO) is open 
all year round for submission.
To participate in the challenge, please create an account at [EvalAI](http://evalai.cloudcv.org/web/challenges/challenge-page/356 ---> @TODO).
Then upload your zipped result file including all of the required [meta data](#results-format).
After each challenge, the results will be exported to the nuScenes [leaderboard](https://www.nuscenes.org/lidar-segmentation) shown above.
This is the only way to benchmark your method against the test dataset. 

## Challenges
To allow users to benchmark the performance of their method against the community, we host a single [leaderboard](https://www.nuscenes.org/lidar-segmentation) all-year round.
Additionally we organize a number of challenges at leading Computer Vision conference workshops.
Users that submit their results during the challenge period are eligible for awards.
Any user that cannot attend the workshop (direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

Click [here](http://evalai.cloudcv.org/web/challenges/challenge-page/356 --> @ TODO) for the **EvalAI detection evaluation server**.

### Workshop ___@TODO___, NeurIPS 2020
The first nuScenes lidar segmentation challenge will be held at NeurIPS 2020.
Submission will open on Nov ___@TODO__, 2020 and close on Dec ___@TODO__, 2020.
Results and winners will be announced at the Workshop ___@TODO___ ([WAD](https://sites.google.com/view/wad2019 ---> @OTODO)) at [NeurIPS 2020](https://nips.cc/Conferences/2020/).
For more information see the [leaderboard](https://www.nuscenes.org/lidar-segmentation).
Note that the [evaluation server](http://evalai.cloudcv.org/web/challenges/challenge-page/356 ---> @TODO) can still be used to benchmark your results.

*Note:* Due to the COVID-19 situation, participants are **not** required to attend in person to be eligible for the prizes.

## Submission rules
### lidar segmentation-specific rules
* The maximum time window of past sensor data and ego poses that may be used at inference time is approximately 0.5s (at most 6 past camera images, 6 past radar sweeps and 10 past lidar sweeps). At training time there are no restrictions.

### General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users make predictions on the test set and submit the results to our evaluation server, which returns the metrics listed below.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* Each user or team can have at most one one account on the evaluation server.
* Each user or team can submit at most 3 results. These results must come from different models, rather than submitting results from the same model at different training epochs or with slightly different parameters.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from all nuScenes challenges. 

## Results format
We define a standardized lidar segmentation result format that serves as an input to the evaluation code.
Results are evaluated for each 2Hz keyframe, also known as a `sample`.
The lidar segmentation results for a particular evaluation set (train/val/test) are stored in a folder. 

The folder structure of the results should be as follows:
```
└── results_folder
    ├── lidarseg
    │   └── v1.0-test <- Contains the .bin files; a .bin file 
    │                    contains the labels of the points in a 
    │                    point cloud         
    └── v1.0-test
        └── submission.json  <- contains certain information about 
                                the submission
```

The contents of the `submision.json` file and `v1.0-test` folder are defined below: 
* The `submission.json` file includes meta data `meta` on the type of inputs used for this method.
  Furthermore it includes a dictionary `results` that maps each sample_token to a list of `sample_result` entries.
  Each `sample_token` from the current evaluation set must be included in `results`.
  ```
  "meta": {
      "use_camera":   <bool>          -- Whether this submission uses camera data as an input.
      "use_lidar":    <bool>          -- Whether this submission uses lidar data as an input.
      "use_radar":    <bool>          -- Whether this submission uses radar data as an input.
      "use_map":      <bool>          -- Whether this submission uses map data as an input.
      "use_external": <bool>          -- Whether this submission uses external data as an input.
  },
  "mapping": {
      class_idx <int>: class_name <str> -- Maps each class index to a class name.
  }
  ```
* The `v1.0-test` folder contains .bin files, where each .bin file contains the labels of the points for the point cloud.
  Pay special attention that each set of predictions in the folder must be a .bin file and named as <lidar_sample_data_token>_lidarseg.bin.
  A .bin file contains an array of `int` in which each value is the predicted class index of the corresponding point in the point cloud, e.g.:
  ```
  [1, 5, 4, 1, ...])
  ```

For the train and val sets, the evaluation can be performed by the user on their local machine.
For the test set, the user needs to zip the results folder and submit it to the official evaluation server.

Note that the lidar segmentation classes may differ from the general nuScenes classes, as detailed below.

## Classes
The nuScenes-lidarseg dataset comes with annotations for 32 classes ([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples.
Hence we merge similar classes and remove rare classes.
This results in # TODO XX classes for the lidar segmentation challenge.
Below we show the table of lidar segmentation classes and their counterparts in the nuScenes-lidarseg dataset.
For more information on the classes and their frequencies, see [this page](https://www.nuscenes.org/nuscenes#data-annotation).

|   lidar segmentation class    |   nuScenes-lidarseg general class         |
|   ---                         |   ---                                     |
|   void / ignore               |   animal                                  |
|   void / ignore               |   human.pedestrian.personal_mobility      |
|   void / ignore               |   human.pedestrian.stroller               |
|   void / ignore               |   human.pedestrian.wheelchair             |
|   void / ignore               |   movable_object.debris                   |
|   void / ignore               |   movable_object.pushable_pullable        |
|   void / ignore               |   static_object.bicycle_rack              |
|   void / ignore               |   vehicle.emergency.ambulance             |
|   void / ignore               |   vehicle.emergency.police                |
|   void / ignore               |   noise                                   |
|   void / ignore               |   static.other                            |
|   void / ignore               |   vehicle.ego                             |
|   barrier                     |   movable_object.barrier                  |
|   bicycle                     |   vehicle.bicycle                         |
|   bus                         |   vehicle.bus.bendy                       |
|   bus                         |   vehicle.bus.rigid                       |
|   car                         |   vehicle.car                             |
|   construction_vehicle        |   vehicle.construction                    |
|   motorcycle                  |   vehicle.motorcycle                      |
|   pedestrian                  |   human.pedestrian.adult                  |
|   pedestrian                  |   human.pedestrian.child                  |
|   pedestrian                  |   human.pedestrian.construction_worker    |
|   pedestrian                  |   human.pedestrian.police_officer         |
|   traffic_cone                |   movable_object.trafficcone              |
|   trailer                     |   vehicle.trailer                         |
|   truck                       |   vehicle.truck                           |
|   driveable_surface           |   flat.driveable_surface                  |
|   other_flat                  |   flat.other                              |
|   sidewalk                    |   flat.sidewalk                           |
|   terrain                     |   flat.terrain                            |
|   manmade                     |   static.manmade                          |
|   vegetation                  |   static.vegetation                       |


## Evaluation metrics
Below we define the metrics for the nuScenes lidar segmentation task.
Our final score is a weighted sum of mean intersection-over-union (IOU) and boundary IOU.

### Preprocessing
N.A.

### Mean IOU:
We use the well-known IOU metric, which is defined as TP / (TP + FP + FN). 
The IOU score is calculated separately for each class, and then the mean is computed across classes.

### Boundary IOU
Here we define metrics for a set of true positives (TP) that measure translation / scale / orientation / velocity and attribute errors. 
All TP metrics are calculated using a threshold of 2m center distance during matching, and they are all designed to be positive scalars.

Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%.
If 10% recall is not achieved for a particular class, all TP errors for that class are set to 1.
We define the following TP errors:
* **Average Translation Error (ATE)**: Euclidean center distance in 2D in meters.
* **Average Scale Error (ASE)**: Calculated as *1 - IOU* after aligning centers and orientation.
* **Average Orientation Error (AOE)**: Smallest yaw angle difference between prediction and ground-truth in radians. Orientation error is evaluated at 360 degree for all classes except barriers where it is only evaluated at 180 degrees. Orientation errors for cones are ignored.
* **Average Velocity Error (AVE)**: Absolute velocity error in m/s. Velocity error for barriers and cones are ignored.
* **Average Attribute Error (AAE)**: Calculated as *1 - acc*, where acc is the attribute classification accuracy. Attribute error for barriers and cones are ignored.

All errors are >= 0, but note that for translation and velocity errors the errors are unbounded, and can be any positive value.

The TP metrics are defined per class, and we then take a mean over classes to calculate mATE, mASE, mAOE, mAVE and mAAE.

### nuScenes-lidarseg score
* **nuScenes-lidarseg score (NLS)**:
We consolidate the above metrics by computing a weighted sum: mAP, mATE, mASE, mAOE, mAVE and mAAE.
As a first step we convert the TP errors to TP scores as *TP_score = max(1 - TP_error, 0.0)*.
We then assign a weight of *5* to mAP and *1* to each of the 5 TP scores and calculate the normalized sum.

### Configuration
The default evaluation metrics configurations can be found in `nuscenes/eval/detection/configs/detection_cvpr_2019.json`. ## TODO

## Leaderboard
nuScenes will maintain a single leaderboard for the lidar segmentation task.
For each submission the leaderboard will list method aspects and evaluation metrics.
Method aspects include input modalities (lidar, radar, vision), use of map data and use of external data.
To enable a fair comparison between methods, the user will be able to filter the methods by method aspects.

Methods will be compared within these tracks and the winners will be decided for each track separately.
Furthermore, there will also be an award for novel ideas, as well as the best student submission.

**Lidar track**: 
* Only lidar input allowed.
* External data or map data <u>not allowed</u>.
* May use pre-training.
  
**Open track**: 
* Any sensor input allowed.
* External data and map data allowed.  
* May use pre-training.

**Details**:
* *Sensor input:*
For the lidar track we restrict the type of sensor input that may be used.
Note that this restriction applies only at test time.
At training time any sensor input may be used.

* *Map data:*
By `map data` we mean using the *semantic* map provided in nuScenes. 

* *Meta data:*
Other meta data included in the dataset may be used without restrictions.
E.g. calibration parameters, ego poses, `location`, `timestamp`, `num_lidar_pts`, `num_radar_pts`, `translation`, `rotation` and `size`.
Note that `instance`, `sample_annotation` and `scene` description are not provided for the test set.

* *Pre-training:*
By pre-training we mean training a network for the task of image classification using only image-level labels,
as done in [[Krizhevsky NIPS 2012]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ).
The pre-training may not involve bounding boxes, masks or other localized annotations.

* *Reporting:* 
Users are required to report detailed information on their method regarding sensor input, map data, meta data and pre-training.
Users that fail to adequately report this information may be excluded from the challenge. 
