# nuScenes tracking task 
![nuScenes Tracking logo](https://www.nuscenes.org/public/images/tracking_challenge.png)

## Overview
- [Introduction](#introduction)
- [Authors](#authors)
- [Getting started](#getting-started)
- [Participation](#participation)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Classes](#classes)
- [Evaluation metrics](#evaluation-metrics)
- [Baselines](#baselines)
- [Leaderboard](#leaderboard)
- [Yonohub](#yonohub)

## Introduction
The [nuScenes dataset](http://www.nuScenes.org) \[1\] has achieved widespread acceptance in academia and industry as a standard dataset for AV perception problems.
To advance the state-of-the-art on the problems of interest we propose benchmark challenges to measure the performance on our dataset.
At CVPR 2019 we organized the [nuScenes detection challenge](https://www.nuscenes.org/object-detection).
The nuScenes tracking challenge is a natural progression to the detection challenge, building on the best known detection algorithms and tracking these across time.
Here we describe the challenge, the rules, the classes, evaluation metrics and general infrastructure.

## Authors
The tracking task and challenge are a joint work between **Aptiv** (Holger Caesar, Caglayan Dicle, Oscar Beijbom) and **Carnegie Mellon University** (Xinshuo Weng, Kris Kitani).
They are based upon the [nuScenes dataset](http://www.nuScenes.org) \[1\] and the [3D MOT baseline and benchmark](https://github.com/xinshuoweng/AB3DMOT) defined in \[2\].

# Getting started
To participate in the tracking challenge you should first [get familiar with the nuScenes dataset and install it](https://github.com/nutonomy/nuscenes-devkit/blob/master/README.md).
In particular, the [tutorial](https://www.nuscenes.org/tutorial) explains how to use the various database tables.
The tutorial also shows how to retrieve the images, lidar pointclouds and annotations for each sample (timestamp).
To retrieve the instance/track of an object, take a look at the [instance table](https://github.com/nutonomy/nuscenes-devkit/blob/master/schema.md#instance).
Now you are ready to train your tracking algorithm on the dataset.
If you are only interested in tracking (as opposed to detection), you can use the provided detections for several state-of-the-art methods [below](#baselines).
To evaluate the tracking results, use `evaluate.py` in the [eval folder](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/tracking).
In `loaders.py` we provide some methods to organize the raw box data into tracks that may be helpful.
 
## Participation
The nuScenes tracking evaluation server is open all year round for submission.
To participate in the challenge, please create an account at [EvalAI](http://evalai.cloudcv.org/web/challenges/challenge-page/475).
Then upload your zipped result file including all of the required [meta data](#results-format).
The results will be exported to the nuScenes leaderboard shown above (coming soon).
This is the only way to benchmark your method against the test dataset.

## Challenges
To allow users to benchmark the performance of their method against the community, we host a single [leaderboard](#leaderboard) all-year round.
Additionally we organize a number of challenges at leading Computer Vision conference workshops.
Users that submit their results during the challenge period are eligible for awards.
Any user that cannot attend the workshop (direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

Click [here](http://evalai.cloudcv.org/web/challenges/challenge-page/475) for the **EvalAI tracking evaluation server**.

### AI Driving Olympics (AIDO), NIPS 2019
The first nuScenes tracking challenge will be held at NIPS 2019.
Submission will open October 1 and close December 9.
The leaderboard will remain private until the end of the challenge.
Results and winners will be announced at the [AI Driving Olympics](http://www.driving-olympics.ai/) Workshop (AIDO) at NIPS 2019.

## Submission rules
### Tracking-specific rules
* We perform 3D Multi Object Tracking (MOT) as in \[2\], rather than 2D MOT as in KITTI \[4\]. 
* Possible input modalities are camera, lidar and radar.
* We perform online tracking \[2\]. This means that the tracker may only use past and current, but not future sensor data.
* Noisy object detections are provided below (including for the test split), but do not have to be used.
* At inference time users may use all past sensor data and ego poses from the current scene, but not from a previous scene. At training time there are no restrictions.

### General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users make predictions on the test set and submit the results to our evaluation server, which returns the metrics listed below.
* We do not use strata. Instead, we filter annotations and predictions beyond class specific distances.
* Users must limit the number of submitted boxes per sample to 500.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* Each user or team can have at most one account on the evaluation server.
* Each user or team can submit at most 3 results. These results must come from different models, rather than submitting results from the same model at different training epochs or with slightly different parameters.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from all nuScenes challenges.

## Results format
We define a standardized tracking result format that serves as an input to the evaluation code.
Results are evaluated for each 2Hz keyframe, also known as `sample`.
The tracking results for a particular evaluation set (train/val/test) are stored in a single JSON file. 
For the train and val sets the evaluation can be performed by the user on their local machine.
For the test set the user needs to zip the single JSON result file and submit it to the official evaluation server (see above).
The JSON file includes meta data `meta` on the type of inputs used for this method.
Furthermore it includes a dictionary `results` that maps each sample_token to a list of `sample_result` entries.
Each `sample_token` from the current evaluation set must be included in `results`, although the list of predictions may be empty if no object is tracked.
```
submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_map":      <bool>  -- Whether this submission uses map data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
}
```
For the predictions we create a new database table called `sample_result`.
The `sample_result` table is designed to mirror the [`sample_annotation`](https://github.com/nutonomy/nuscenes-devkit/blob/master/schema.md#sample_annotation) table.
This allows for processing of results and annotations using the same tools.
A `sample_result` is a dictionary defined as follows:
```
sample_result {
    "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
    "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
    "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
    "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                       Note that the tracking_name cannot change throughout a track.
    "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                       We average over frame level scores to compute the track level score.
                                       The score is used to determine positive and negative tracks via thresholding.
}
```
Note that except for the `tracking_*` fields the result format is identical to the [detection challenge](https://www.nuscenes.org/object-detection).

## Classes
The nuScenes dataset comes with annotations for 23 classes ([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples.
Hence we merge similar classes and remove rare classes.
From these *detection challenge classes* we further remove the classes *barrier*, *trafficcone* and *construction_vehicle*, as these are typically static.
Below we show the table of the 7 tracking classes and their counterparts in the nuScenes dataset.
For more information on the classes and their frequencies, see [this page](https://www.nuscenes.org/data-annotation).

|   nuScenes general class                  |   nuScenes tracking class |
|   ---                                     |   ---                     |
|   animal                                  |   void / ignore           |
|   human.pedestrian.personal_mobility      |   void / ignore           |
|   human.pedestrian.stroller               |   void / ignore           |
|   human.pedestrian.wheelchair             |   void / ignore           |
|   movable_object.barrier                  |   void / ignore           |
|   movable_object.debris                   |   void / ignore           |
|   movable_object.pushable_pullable        |   void / ignore           |
|   movable_object.trafficcone              |   void / ignore           |
|   static_object.bicycle_rack              |   void / ignore           |
|   vehicle.emergency.ambulance             |   void / ignore           |
|   vehicle.emergency.police                |   void / ignore           |
|   vehicle.construction                    |   void / ignore           |
|   vehicle.bicycle                         |   bicycle                 |
|   vehicle.bus.bendy                       |   bus                     |
|   vehicle.bus.rigid                       |   bus                     |
|   vehicle.car                             |   car                     |
|   vehicle.motorcycle                      |   motorcycle              |
|   human.pedestrian.adult                  |   pedestrian              |
|   human.pedestrian.child                  |   pedestrian              |
|   human.pedestrian.construction_worker    |   pedestrian              |
|   human.pedestrian.police_officer         |   pedestrian              |
|   vehicle.trailer                         |   trailer                 |
|   vehicle.truck                           |   truck                   |

For each nuScenes class, the number of annotations decreases with increasing radius from the ego vehicle, 
but the number of annotations per radius varies by class. Therefore, each class has its own upper bound on evaluated
detection radius, as shown below:

|   nuScenes tracking class     |   KITTI class |   Tracking range (meters) |
|   ---                         |   ---         |   ---                     |
|   bicycle                     |   cyclist     |   40                      |
|   motorcycle                  |   cyclist     |   40                      |
|   pedestrian                  |   pedestrian / person (sitting) |   40    |
|   bus                         |   -           |   50                      |
|   car                         |   car / van   |   50                      |
|   trailer                     |   -           |   50                      |
|   truck                       |   truck       |   50                      |

In the above table we also provide the mapping from nuScenes tracking class to KITTI \[4\] class.
While KITTI defines 8 classes in total, only `car` and `pedestrian` are used for the tracking benchmark, as the other classes do not have enough samples.
Our goal is to perform tracking of all moving objects in a traffic scene.

## Evaluation metrics
Below we define the metrics for the nuScenes tracking task.
Note that all metrics below (except FPS) are computed per class and then averaged over all classes.
The challenge winner will be determined based on AMOTA.
Additionally a number of secondary metrics are computed and shown on the leaderboard.

### Preprocessing
Before running the evaluation code the following pre-processing is done on the data
* All boxes (GT and prediction) are removed if they exceed the class-specific detection range.  

### Preprocessing
Before running the evaluation code the following pre-processing is done on the data:
* All boxes (GT and prediction) are removed if they exceed the class-specific tracking range. 
* All bikes and motorcycle boxes (GT and prediction) that fall inside a bike-rack are removed. The reason is that we do not annotate bikes inside bike-racks.  
* All boxes (GT) without lidar or radar points in them are removed. The reason is that we can not guarantee that they are actually visible in the frame. We do not filter the predicted boxes based on number of points.
* To avoid excessive track fragmentation from lidar/radar point filtering, we linearly interpolate GT and predicted tracks.

### Matching criterion
For all metrics, we define a match by thresholding the 2D center distance on the ground plane rather than Intersection Over Union (IOU) based affinities.
We find that this measure is more forgiving for far-away objects than IOU which is often 0, particularly for monocular image-based approaches.
The matching threshold (center distance) is 2m.

### AMOTA and AMOTP metrics
Our main metrics are the AMOTA and AMOTP metrics developed in \[2\].
These are integrals over the MOTA/MOTP curves using `n`-point interpolation (`n = 40`).
Similar to the detection challenge, we do not include points with `recall < 0.1` (not shown in the equation), as these are typically noisy.

- **AMOTA** (average multi object tracking accuracy):
Average over the MOTA \[3\] metric (see below) at different recall thresholds.
For the traditional MOTA formulation at recall 10% there are at least 90% false negatives, which may lead to negative MOTAs.
Therefore the contribution of identity switches and false positives becomes negligible at low recall values.
In `MOTAR` we include recall-normalization term `- (1-r) * P` in the nominator, the factor `r` in the denominator and the maximum.
These guarantee that the values span the entire `[0, 1]` range and brings the three error types into a similar value range.
`P` refers to the number of ground-truth positives for the current class. 
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\dpi{400}&space;\tiny&space;\mathit{AMOTA}&space;=&space;\small&space;\frac{1}{n-1}&space;\sum_{r&space;\in&space;\{\frac{1}{n-1},&space;\frac{2}{n-1}&space;\,&space;...&space;\,&space;\,&space;1\}}&space;\mathit{MOTAR}" target="_blank">
<img width="400" src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\dpi{400}&space;\tiny&space;\mathit{AMOTA}&space;=&space;\small&space;\frac{1}{n-1}&space;\sum_{r&space;\in&space;\{\frac{1}{n-1},&space;\frac{2}{n-1}&space;\,&space;...&space;\,&space;\,&space;1\}}&space;\mathit{MOTAR}" title="\dpi{400} \tiny \mathit{AMOTA} = \small \frac{1}{n-1} \sum_{r \in \{\frac{1}{n-1}, \frac{2}{n-1} \, ... \, \, 1\}} \mathit{MOTAR}" /></a>
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\mathit{MOTAR}&space;=&space;\max&space;(0,\;&space;1&space;\,&space;-&space;\,&space;\frac{\mathit{IDS}_r&space;&plus;&space;\mathit{FP}_r&space;&plus;&space;\mathit{FN}_r&space;-&space;(1-r)&space;*&space;\mathit{P}}{r&space;*&space;\mathit{P}})" target="_blank">
<img width="450" src="https://latex.codecogs.com/gif.latex?\dpi{300}&space;\mathit{MOTAR}&space;=&space;\max&space;(0,\;&space;1&space;\,&space;-&space;\,&space;\frac{\mathit{IDS}_r&space;&plus;&space;\mathit{FP}_r&space;&plus;&space;\mathit{FN}_r&space;-&space;(1-r)&space;*&space;\mathit{P}}{r&space;*&space;\mathit{P}})" title="\mathit{MOTAR} = \max (0,\; 1 \, - \, \frac{\mathit{IDS}_r + \mathit{FP}_r + \mathit{FN}_r + (1-r) * \mathit{P}}{r * \mathit{P}})" /></a>

- **AMOTP** (average multi object tracking precision):
Average over the MOTP metric defined below.
Here `d_{i,t}` indicates the position error of track `i` at time `t` and `TP_t` indicates the number of matches at time `t`. See \[3\]. 
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{300}&space;\mathit{AMOTP}&space;=&space;\small&space;\frac{1}{n-1}&space;\sum_{r&space;\in&space;\{\frac{1}{n-1},&space;\frac{2}{n-1},&space;..,&space;1\}}&space;\frac{\sum_{i,t}&space;d_{i,t}}{\sum_t&space;\mathit{TP}_t}" target="_blank">
<img width="300" src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\mathit{AMOTP}&space;=&space;\small&space;\frac{1}{n-1}&space;\sum_{r&space;\in&space;\{\frac{1}{n-1},&space;\frac{2}{n-1},&space;..,&space;1\}}&space;\frac{\sum_{i,t}&space;d_{i,t}}{\sum_t&space;\mathit{TP}_t}" title="\mathit{AMOTP} = \small \frac{1}{n-1} \sum_{r \in \{\frac{1}{n-1}, \frac{2}{n-1}, .., 1\}} \frac{\sum_{i,t} d_{i,t}}{\sum_t \mathit{TP}_t}" />
</a>

### Secondary metrics
We use a number of standard MOT metrics including CLEAR MOT \[3\] and ML/MT as listed on [motchallenge.net](https://motchallenge.net).
Contrary to the above AMOTA and AMOTP metrics, these metrics use a confidence threshold to determine positive and negative tracks.
The confidence threshold is selected for every class independently by picking the threshold that achieves the highest MOTA.
The track level scores are determined by averaging the frame level scores.
Tracks with a score below the confidence threshold are discarded.
* **MOTA** (multi object tracking accuracy) \[3\]: This measure combines three error sources: false positives, missed targets and identity switches.
* **MOTP** (multi object tracking precision) \[3\]: The misalignment between the annotated and the predicted bounding boxes.
* **FAF**: The average number of false alarms per frame.
* **MT** (ratio of mostly tracked trajectories): The ratio of ground-truth trajectories that are covered by a track hypothesis for at least 80% of their respective life span.
* **ML** (ratio of mostly lost trajectories): The ratio of ground-truth trajectories that are covered by a track hypothesis for at most 20% of their respective life span.
* **FP** (number of false positives): The total number of false positives.
* **FN** (number of false negatives): The total number of false negatives (missed targets).
* **IDS** (number of identity switches): The total number of identity switches.
* **Frag** (number of track fragmentations): The total number of times a trajectory is fragmented (i.e. interrupted during tracking).

Users are asked to provide the runtime of their method:
* **FPS** (tracker speed in frames per second): Processing speed in frames per second excluding the detector on the benchmark. Users report both the detector and tracking FPS separately as well as cumulative. This metric is self-reported and therefore not directly comparable.

Furthermore we propose a number of additional metrics:
* **TID** (average track initialization duration in seconds): Some trackers require a fixed window of past sensor readings. Trackers may also perform poorly without a good initialization. The purpose of this metric is to measure for each track the initialization duration until the first object was successfully detected. If an object is not tracked, we assign the entire track duration as initialization duration. Then we compute the average over all tracks.     
* **LGD** (average longest gap duration in seconds): *Frag* measures the number of fragmentations. For the application of Autonomous Driving it is crucial to know how long an object has been missed. We compute this duration for each track. If an object is not tracked, we assign the entire track duration as initialization duration.

### Configuration
The default evaluation metrics configurations can be found in `nuscenes/eval/tracking/configs/tracking_nips_2019.json`.

### Baselines
To allow the user focus on the tracking problem, we release object detections from state-of-the-art methods as listed on the [detection leaderboard](https://www.nuscenes.org/object-detection).
We thank Alex Lang (Aptiv), Benjin Zhu (Megvii) and Andrea Simonelli (Mapillary) for providing these.
The use of these detections is entirely optional.
The detections on the train, val and test splits can be downloaded from the table below.
Our tracking baseline is taken from *"A Baseline for 3D Multi-Object Tracking"* \[2\] and uses each of the provided detections.
The results for object detection and tracking can be seen below.
These numbers are measured on the val split and therefore not identical to the test set numbers on the leaderboard.
Note that we no longer use the weighted version of AMOTA (*Updated 10 December 2019*). 

|   Method             | NDS  | mAP  | AMOTA | AMOTP | Modality | Detections download                                              | Tracking download                                               |
|   ---                | ---  | ---  | ---   | ---   | ---      | ---                                                              | ---                                                             |
|   Megvii \[6\]       | 62.8 | 51.9 | 17.9  | 1.50  | Lidar    | [link](https://www.nuscenes.org/data/detection-megvii.zip)       | [link](https://www.nuscenes.org/data/tracking-megvii.zip)       |
|   PointPillars \[5\] | 44.8 | 29.5 |  3.5  | 1.69  | Lidar    | [link](https://www.nuscenes.org/data/detection-pointpillars.zip) | [link](https://www.nuscenes.org/data/tracking-pointpillars.zip) |
|   Mapillary \[7\]    | 36.9 | 29.8 |  4.5  | 1.79  | Camera   | [link](https://www.nuscenes.org/data/detection-mapillary.zip)    | [link](https://www.nuscenes.org/data/tracking-mapillary.zip)    |

#### Overfitting
Some object detection methods overfit to the training data.
E.g. for the PointPillars method we see a drop in mAP of 6.2% from train to val split (35.7% vs. 29.5%).
This may affect (learning-based) tracking algorithms, when the training split has more accurate detections than the validation split.
To remedy this problem we have split the existing `train` set into `train_detect` and `train_track` (350 scenes each).
Both splits have the same distribution of Singapore, Boston, night and rain data.
You can use these splits to train your own detection and tracking algorithms.
The use of these splits is entirely optional.
The object detection baselines provided in the table above are trained on the *entire* training set, as our tracking baseline \[2\] is not learning-based and therefore not prone to overfitting.

## Leaderboard
nuScenes will maintain a single leaderboard for the tracking task.
For each submission the leaderboard will list method aspects and evaluation metrics.
Method aspects include input modalities (lidar, radar, vision), use of map data and use of external data.
To enable a fair comparison between methods, the user will be able to filter the methods by method aspects.
 
We define three such filters here which correspond to the tracks in the nuScenes tracking challenge.
Methods will be compared within these tracks and the winners will be decided for each track separately.
Note that the tracks are identical to the [nuScenes detection challenge](https://www.nuscenes.org/object-detection) tracks.

**Lidar track**: 
* Only lidar input allowed.
* External data or map data <u>not allowed</u>.
* May use pre-training.
  
**Vision track**: 
* Only camera input allowed.
* External data or map data <u>not allowed</u>.
* May use pre-training.
 
**Open track**:
* Any sensor input allowed (radar, lidar, camera, ego pose).
* External data and map data allowed.  
* May use pre-training.

**Details**:
* *Sensor input:*
For the lidar and vision tracks we restrict the type of sensor input that may be used.
Note that this restriction applies only at test time.
At training time any sensor input may be used.
In particular this also means that at training time you are allowed to filter the GT boxes using `num_lidar_pts` and `num_radar_pts`, regardless of the track.
However, during testing the predicted boxes may *not* be filtered based on input from other sensor modalities.

* *Map data:*
By `map data` we mean using the *semantic* map provided in nuScenes. 

* *Meta data:*
Other meta data included in the dataset may be used without restrictions.
E.g. calibration parameters, ego poses, `location`, `timestamp`, `num_lidar_pts`, `num_radar_pts`, `translation`, `rotation` and `size`.
Note that `instance`, `sample_annotation` and `scene` description are not provided for the test set.

* *Pre-training:*
By pre-training we mean training a network for the task of image classification using only image-level labels,
as done in [[Krizhevsky NIPS 2012]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ).
The pre-training may not involve bounding box, mask or other localized annotations.

* *Reporting:* 
Users are required to report detailed information on their method regarding sensor input, map data, meta data and pre-training.
Users that fail to adequately report this information may be excluded from the challenge.

## Yonohub 
[Yonohub](https://yonohub.com/) is a web-based system for building, sharing, and evaluating complex systems, such as autonomous vehicles, using drag-and-drop tools.
It supports general blocks for nuScenes, as well as the detection and tracking baselines and evaluation code.
For more information read the [medium article](https://medium.com/@ahmedmagdyattia1996/using-yonohub-to-participate-in-the-nuscenes-tracking-challenge-338a3e338db9) and the [tutorial](https://docs.yonohub.com/docs/yonohub/nuscenes-package/).
Yonohub also provides [free credits](https://yonohub.com/nuscenes-package-and-sponsorship/) of up to $1000 for students to get started with Yonohub on nuScenes.
Note that these are available even after the end of the official challenge.   

## References
- \[1\] *"nuScenes: A multimodal dataset for autonomous driving"*, H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan and O. Beijbom, In arXiv 2019.
- \[2\] *"A Baseline for 3D Multi-Object Tracking"*, X. Weng and K. Kitani, In arXiv 2019.
- \[3\] *"Multiple object tracking performance metrics and evaluation in a smart room environment"*, K. Bernardin, A. Elbs, R. Stiefelhagen, In Sixth IEEE International Workshop on Visual Surveillance, in conjunction with ECCV, 2006.
- \[4\] *"Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"*, A. Geiger, P. Lenz, R. Urtasun, In CVPR 2012.
- \[5\] *"PointPillars: Fast Encoders for Object Detection from Point Clouds"*, A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang and O. Beijbom, In CVPR 2019.
- \[6\] *"Class-balanced Grouping and Sampling for Point Cloud 3D Object Detection"*, B. Zhu, Z. Jiang, X. Zhou, Z. Li, G. Yu, In arXiv 2019.
- \[7\] *"Disentangling Monocular 3D Object Detection"*, A. Simonelli, S. R. Bulo, L. Porzi, M. Lopez-Antequera, P. Kontschieder, In arXiv 2019.
- \[8\] *"PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud"*, S. Shi, X. Wang, H. Li, In CVPR 2019.
