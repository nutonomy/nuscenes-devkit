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
The goal of this task is to predict the category of every point in a set of point clouds. There are 16 categories (10 foreground classes and 6 background classes).

## Participation
The nuScenes lidarseg segmentation [evaluation server](http://evalai.cloudcv.org/web/challenges/challenge-page/356 ---> @TODO) is open 
all year round for submission.
To participate in the challenge, please create an account at [EvalAI](http://evalai.cloudcv.org/web/challenges/challenge-page/356 ---> @TODO).
Then upload your zipped result folder with the required [content](#results-format).
After each challenge, the results will be exported to the nuScenes [leaderboard](https://www.nuscenes.org/lidar-segmentation).
This is the only way to benchmark your method against the test dataset. 

## Challenges
To allow users to benchmark the performance of their method against the community, we host a single [leaderboard](https://www.nuscenes.org/lidar-segmentation) all-year round.
Additionally we organize a number of challenges at leading Computer Vision conference workshops.
Users that submit their results during the challenge period are eligible for awards.
Any user that cannot attend the workshop (direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

Click [here](http://evalai.cloudcv.org/web/challenges/challenge-page/356 --> @ TODO) for the **EvalAI lidar segementation evaluation server**.

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
  Pay special attention that each set of predictions in the folder must be a .bin file and named as **<lidar_sample_data_token>_lidarseg.bin**.
  A .bin file contains an array of `int` in which each value is the predicted [class index](#classes) of the corresponding point in the point cloud, e.g.:
  ```
  [1, 5, 4, 1, ...]
  ```
  Each `lidar_sample_data_token` from the current evaluation set must be included in the `v1.0-test` folder.
  
For the train and val sets, the evaluation can be performed by the user on their local machine.
For the test set, the user needs to zip the results folder and submit it to the official evaluation server.

Note that the lidar segmentation classes may differ from the general nuScenes classes, as detailed below.

## Classes
The nuScenes-lidarseg dataset comes with annotations for 32 classes ([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples.
Hence we merge similar classes and remove rare classes.
This results in 16 classes for the lidar segmentation challenge.
Below we show the table of lidar segmentation classes and their counterparts in the nuScenes-lidarseg dataset.
For more information on the classes and their frequencies, see [this page](https://www.nuscenes.org/nuscenes#data-annotation).

|   lidar segmentation index    |   lidar segmentation class    |   nuScenes-lidarseg general class         |
|   ---                         |   ---                         |   ---                                     |
|   N.A.                        |   void / ignore               |   animal                                  |
|   N.A.                        |   void / ignore               |   human.pedestrian.personal_mobility      |
|   N.A.                        |   void / ignore               |   human.pedestrian.stroller               |
|   N.A.                        |   void / ignore               |   human.pedestrian.wheelchair             |
|   N.A.                        |   void / ignore               |   movable_object.debris                   |
|   N.A.                        |   void / ignore               |   movable_object.pushable_pullable        |
|   N.A.                        |   void / ignore               |   static_object.bicycle_rack              |
|   N.A.                        |   void / ignore               |   vehicle.emergency.ambulance             |
|   N.A.                        |   void / ignore               |   vehicle.emergency.police                |
|   N.A.                        |   void / ignore               |   noise                                   |
|   N.A.                        |   void / ignore               |   static.other                            |
|   N.A.                        |   void / ignore               |   vehicle.ego                             |
|   1                           |   barrier                     |   movable_object.barrier                  |
|   2                           |   bicycle                     |   vehicle.bicycle                         |
|   3                           |   bus                         |   vehicle.bus.bendy                       |
|   3                           |   bus                         |   vehicle.bus.rigid                       |
|   4                           |   car                         |   vehicle.car                             |
|   5                           |   construction_vehicle        |   vehicle.construction                    |
|   6                           |   motorcycle                  |   vehicle.motorcycle                      |
|   7                           |   pedestrian                  |   human.pedestrian.adult                  |
|   7                           |   pedestrian                  |   human.pedestrian.child                  |
|   7                           |   pedestrian                  |   human.pedestrian.construction_worker    |
|   7                           |   pedestrian                  |   human.pedestrian.police_officer         |
|   8                           |   traffic_cone                |   movable_object.trafficcone              |
|   9                           |   trailer                     |   vehicle.trailer                         |
|   10                          |   truck                       |   vehicle.truck                           |
|   11                          |   driveable_surface           |   flat.driveable_surface                  |
|   12                          |   other_flat                  |   flat.other                              |
|   13                          |   sidewalk                    |   flat.sidewalk                           |
|   14                          |   terrain                     |   flat.terrain                            |
|   15                          |   manmade                     |   static.manmade                          |
|   16                          |   vegetation                  |   static.vegetation                       |


## Evaluation metrics
Below we define the metrics for the nuScenes lidar segmentation task.
Our final score is a weighted sum of mean intersection-over-union (IOU) and boundary IOU.

### Preprocessing
N.A.

### Mean IOU (mIOU)
We use the well-known IOU metric, which is defined as TP / (TP + FP + FN). 
The IOU score is calculated separately for each class, and then the mean is computed across classes.

### Boundary IOU (bIOU)
To better evaluate the performance of the predicted segmentation, we introduce an additional metric to measure the accuracy for boundary delineation.
Here, we project the lidar point cloud into the range view and determine which "pixels" in the range view image are boundary "pixels".
We then define boundary IOU as TP / (TP + FP + FN) but taken only on the boundary "pixels".
This is inspired by the work done by [Milioto et al.](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf)
Note that the boundary IOU is class-agnostic, i.e. it is **not** evaluated per class.

### nuScenes-lidarseg score
* **nuScenes-lidarseg score (NLS)**:
We consolidate the above metrics by taking the average of mIOU and bIOU (i.e. equal weighting is given to both).

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
