# nuScenes lidar panoptic segmentation and tracking task
![nuScenes lidar panoptic logo](https://www.nuscenes.org/public/images/panoptic_challenge.jpg)

## Overview
- [Introduction](#introduction)
- [Citation](#citation)
- [Participation](#participation)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Classes](#classes)
- [Evaluation metrics](#evaluation-metrics)
- [Leaderboard](#leaderboard)

## Introduction
We define the lidar panoptic segmentation and panoptic tracking tasks on Panoptic nuScenes. This challenge is a
collaboration between Motional and the Robot Learning Lab of Prof. Valada at the University of Freiburg. For panoptic
segmentation, the goal is to predict the semantic categories of every point, and additional instance IDs for things.
While panoptic segmentation focuses on static frames, panoptic tracking additionally enforces temporal coherence and
pixel-level associations over time. For both tasks, there are 16 categories (10 thing and 6 stuff classes). Refer to
the [Panoptic nuScenes paper](https://arxiv.org/pdf/2109.03805.pdf) for more details.

## Citation
When using the dataset in your research, please cite [Panoptic nuScenes](https://arxiv.org/abs/2109.03805):
```
@article{fong2021panoptic,
  title={Panoptic nuScenes: A Large-Scale Benchmark for LiDAR Panoptic Segmentation and Tracking},
  author={Fong, Whye Kit and Mohan, Rohit and Hurtado, Juana Valeria and Zhou, Lubing and Caesar, Holger and
          Beijbom, Oscar and Valada, Abhinav},
  journal={arXiv preprint arXiv:2109.03805},
  year={2021}
}
```

## Participation
The Panoptic nuScenes challenge [evaluation server](https://eval.ai/web/challenges/challenge-page/1243/overview) is
open all year round for submission. Participants can choose to attend both panoptic segmentation and panoptic tracking
tasks, or only the panoptic segmentation task. To participate in the challenge, please create an account at
[EvalAI](https://eval.ai). Then upload your zipped result folder with the required [content](#results-format). After
each challenge, the results will be exported to the [Panoptic nuScenes leaderboard](https://www.nuscenes.org/panoptic).
This is the only way to benchmark your method against the test dataset. We require that all participants send the
following information to nuScenes@motional.com after submitting their results on EvalAI:
- Team name
- Method name
- Authors
- Affiliations
- Method description (5+ sentences)
- Project URL
- Paper URL
- FPS in Hz (and the hardware used to measure it)

## Challenges
To allow users to benchmark the performance of their method against the community, we host a single
[Panoptic nuScenes leaderboard](https://www.nuscenes.org/panoptic) with filters for different task tracks all year
round. Additionally we organize a number of challenges at leading Computer Vision conference workshops. Users that
submit their results during the challenge period are eligible for awards. Any user that cannot attend the workshop
(direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

### 7th AI Driving Olympics, NeurIPS 2021
The first Panoptic nuScenes challenge will be held at [NeurIPS 2021](https://nips.cc/Conferences/2021/).
Submissions will be accepted from 1 September 2021. **The submission deadline is 1 December 2021, 12:00pm, noon, UTC.**
Results and winners will be announced at the [7th AI Driving Olympics](https://driving-olympics.ai/) at NeurIPS 2021.
For more information see the [leaderboard](https://www.nuscenes.org/panoptic). Note that the
[evaluation server](https://eval.ai/web/challenges/challenge-page/1243/overview) can still be used to benchmark your
results after the challenge.


## Submission rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users make predictions on the test set and submit the results to our evaluation server, which returns the metrics
listed below.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* Each user or team can have at most one account *per year* on the evaluation server. Users that create multiple
accounts to circumvent the rules will be excluded from the competition.
* Each user or team can submit at most three results *per year*. These results must come from different models, rather
than submitting results from the same model at different training epochs or with slightly different parameters.
* Faulty submissions that return an error on Eval AI do not count towards the submission limit.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from all nuScenes
challenges.

## Ground truth format
A ground truth label file named `{token}_panoptic.npz` is provided for each sample in the Panoptic nuScenes dataset.
A `.npz` file contains the panoptic label array (uint16 format) of the corresponding points in a pointcloud. The
panoptic label of each point is: **([general class index](#classes) * 1000 + instance index)**. Note here general class
index (32 classes in total) rather than the challenge class index (16 classes in total) is used. For example, a ground
truth instance from car class (general class index = 17), and with assigned car instance index 1, will have a ground truth
panoptic label of 1000 * 17 + 1 = 17001 in the .npz file. Since these ground truth panoptic labels are generated from
annotated bounding boxes, points that are included in more than 1 bounding box will be ignored, and assigned with
panoptic label 0: class index 0 and instance index 0. For points from [stuff](#classes), their panoptic labels will be
[general class index] * 1000. To align with thing classes, you may think the stuff classes as sharing an instance index
of 0 by all points. To load a ground truth file, you can use:
```
from nuscenes.utils.data_io import load_bin_file
label_file_path = /data/sets/nuscenes/panoptic/v1.0-mini/{token}_panoptic.npz
panoptic_label_arr = load_bin_file(label_file_path, 'panoptic')
```

## Results format
We define a standardized panoptic segmentation and panoptic tracking result format that serves as an input to the
evaluation code. Results are evaluated for each 2Hz keyframe, also known as a `sample`. The results for a particular
evaluation set (train/val/test) are stored in a folder.

The folder structure of the results should be as follows:
```
└── results_folder
    ├── panoptic
    │   └── {test, train, val} <- Contains the .npz files; a .npz file contains the labels of the points in a
    │                             pointcloud, note they must have the same indices as in the original pointcloud.
    └── {test, train, val}
        └── submission.json  <- contains certain information about the submission.
```

The contents of the `submission.json` file and `test` folder are defined below:
* The `submission.json` file includes meta data `meta` on the task to attend and type of inputs used for this method.
  The `task` field format is `[segmentation|tracking]-[lidar|open]`, where `segmentation` and `tracking` correspond to
  panoptic segmentation and panoptic tracking tasks. The other fields are used to decide whether the
  submission belongs to lidar track or open track.
  ```
  "meta": {
      "task":         <str>           -- What task to attend. Must be one of ["segmentation", "tracking"].
      "use_camera":   <bool>          -- Whether this submission uses camera data as an input.
      "use_lidar":    <bool>          -- Whether this submission uses lidar data as an input.
      "use_radar":    <bool>          -- Whether this submission uses radar data as an input.
      "use_map":      <bool>          -- Whether this submission uses map data as an input.
      "use_external": <bool>          -- Whether this submission uses external data as an input.
  },
  ```
* The `test` folder contains .npz files, where each .npz file contains the labels of the points for the point cloud.
  Pay special attention that each set of predictions in the folder must be a .npz file and named as
  **<lidar_sample_data_token>_panoptic.npz**. A .npz file contains an array of `uint16` values in which each value is
  the predicted panoptic label **(1000 * [challenge class index](#classes) + per-category instance index)** of the
  corresponding point in the point cloud. Different from ground truth, here **challenge class index is used rather than
  general class index**. **Per-category instance index is reset to start from 1, i.e., [1, 2, 3, ..., N] (N <= 999)**
  for all instances of each class within each scene. For example, a predicted instance from car class (challenge class
  index = 4), and with assigned car instance index 1, will have a panoptic label of 1000 * 4 + 1 = 4001 in the .npz
  file. For stuff classes, we set the instance index in the above equation to 0. Below is an example of a predicted
  panoptic array.
  ```
  [1001, 5001, 4001, 5002, 0, ...]
  ```
  Below is an example of how to save the predictions for a single point cloud:
  ```
  npz_file_out = lidar_sample_data_token + '_panoptic.npz"
  np.savez_compressed(npz_file_out, data=predictions.astype(np.uint16))
  ```
  Each `lidar_sample_data_token` from the current evaluation set must be included in the `test` folder.
  
For the train and val sets, the evaluation can be performed by the user on their local machine.
For the test set, the user needs to zip the results folder and submit it to the official evaluation server.

Note that the lidar panoptic challenge classes differ from the general nuScenes classes, as detailed below.

## Classes
The Panoptic nuScenes dataset comes with annotations for 32 classes
([details](https://www.nuscenes.org/data-annotation)).
Some of these only have a handful of samples. Just as for the nuScenes-lidarseg dataset, we merge similar classes and
remove rare classes. This results in 10 thing classes and 6 stuff classes for the lidar panoptic challenge.
Below we show the table of panoptic challenge classes and their counterparts in the Panoptic nuScenes dataset.
For more information on the classes and their frequencies, see
[this page](https://www.nuscenes.org/nuscenes#data-annotation).

| general class index |   challenge class index   |  challenge class name (thing/stuff)  |   general class name                     |
|   ---               |   ---                     |   ---                                |   ---                                    |
|   1                 |   0                       |   void / ignore                      |   animal                                 |
|   5                 |   0                       |   void / ignore                      |   human.pedestrian.personal_mobility     |
|   7                 |   0                       |   void / ignore                      |   human.pedestrian.stroller              |
|   8                 |   0                       |   void / ignore                      |   human.pedestrian.wheelchair            |
|   10                |   0                       |   void / ignore                      |   movable_object.debris                  |
|   11                |   0                       |   void / ignore                      |   movable_object.pushable_pullable       |
|   13                |   0                       |   void / ignore                      |   static_object.bicycle_rack             |
|   19                |   0                       |   void / ignore                      |   vehicle.emergency.ambulance            |
|   20                |   0                       |   void / ignore                      |   vehicle.emergency.police               |
|   0                 |   0                       |   void / ignore                      |   noise                                  |
|   29                |   0                       |   void / ignore                      |   static.other                           |
|   31                |   0                       |   void / ignore                      |   vehicle.ego                            |
|   9                 |   1                       |   barrier               (thing)      |   movable_object.barrier                 |
|   14                |   2                       |   bicycle               (thing)      |   vehicle.bicycle                        |
|   15                |   3                       |   bus                   (thing)      |   vehicle.bus.bendy                      |
|   16                |   3                       |   bus                   (thing)      |   vehicle.bus.rigid                      |
|   17                |   4                       |   car                   (thing)      |   vehicle.car                            |
|   18                |   5                       |   construction_vehicle  (thing)      |   vehicle.construction                   |
|   21                |   6                       |   motorcycle            (thing)      |   vehicle.motorcycle                     |
|   2                 |   7                       |   pedestrian            (thing)      |   human.pedestrian.adult                 |
|   3                 |   7                       |   pedestrian            (thing)      |   human.pedestrian.child                 |
|   4                 |   7                       |   pedestrian            (thing)      |   human.pedestrian.construction_worker   |
|   6                 |   7                       |   pedestrian            (thing)      |   human.pedestrian.police_officer        |
|   12                |   8                       |   traffic_cone          (thing)      |   movable_object.trafficcone             |
|   22                |   9                       |   trailer               (thing)      |   vehicle.trailer                        |
|   23                |   10                      |   truck                 (thing)      |   vehicle.truck                          |
|   24                |   11                      |   driveable_surface     (stuff)      |   flat.driveable_surface                 |
|   25                |   12                      |   other_flat            (stuff)      |   flat.other                             |
|   26                |   13                      |   sidewalk              (stuff)      |   flat.sidewalk                          |
|   27                |   14                      |   terrain               (stuff)      |   flat.terrain                           |
|   28                |   15                      |   manmade               (stuff)      |   static.manmade                         |
|   30                |   16                      |   vegetation            (stuff)      |   static.vegetation                      |


## Evaluation metrics
Below we introduce the key metrics for panoptic segmentation and panoptic tracking tasks. We use Panoptic Quality (PQ)
as the primary ranking metric for panoptic segmentation tasks, and Panoptic Tracking (PAT) metric as the primary
ranking metric for panoptic tracking tasks.

### Panoptic Segmentation

#### Panoptic Quality (PQ)
We use the standard PQ [Kirillov et al.](https://arxiv.org/pdf/1801.00868.pdf), which is defined as
∑{**1(p,g)** IoU(p,g)} / (|TP|+ 0.5|FP|+ 0.5|FN|). The set of true positives (TP), false positives (FP), and false
negatives (FN), are computed by matching prediction p to ground-truth g based on the IoU scores. Here **1**(p, g) is an
indicator function of being 1 if (p, g) is TP, otherwise 0. Note in lidar panoptic segmentation, the prediction and
ground truth are 1D panoptic arrays of panoptic labels for lidar points instead of panoptic label images in standard PQ.

#### Segmentation Quality (SQ) and Recognition Quality (RQ)
PQ can be decomposed in terms of Segmentation Quality (SQ) and Recognition Quality (RQ) as SQ x RQ. We additionally use
SQ and RQ, which are defined as ∑{**1**(p,g) IoU(p,g)} / TP and TP / (|TP|+ 0.5|FP|+ 0.5|FN|), respectively.

#### Modified Panoptic Quality (PQ†)
We also use PQ† [Porzi et al.](https://arxiv.org/pdf/1905.01220.pdf), which maintains the PQ metric for thing classes,
but modifies the metric for stuff classes.  PQ† only uses the IoU for stuff classes without differentiating between
different segments.

### Panoptic Tracking

### Panoptic Tracking (PAT)
We define the Panoptic Tracking (PAT) metric, which is based on two separable components that are explicitly related to
the task and allow straightforward interpretation. PAT is computed as the harmonic mean of the Panoptic Quality (PQ)
and Tracking Quality (TQ): (2 x PQ x TQ) / (PQ + TQ), with range [0, 1]. To better represent the tracking quality,
TQ is comprised of association score and track fragmentation components. Interested readers can refer to the
[Panoptic nuScenes paper](https://arxiv.org/pdf/2109.03805.pdf) for more details of PAT metric.

### LiDAR Segmentation and Tracking Quality (LSTQ)
We also use the LSTQ metric [Mehmet et al](https://arxiv.org/pdf/2102.12472.pdf). The LSTQ metric is computed as a
geometric mean of the classification score and association score.

### Panoptic Tracking Quality (PTQ)

We also use the PTQ [Hurtado et al.](https://arxiv.org/pdf/2004.08189.pdf) that extends PQ with the IoU of matched
segments with track ID discrepancy, penalizing the incorrect track predictions. PTQ is defined as
(∑{**1**(p,g) IoU(p,g)} - |IDS|) / (|TP|+ 0.5|FP|+ 0.5|FN|), where IDS stands for ID switches, and it is computed as
the number of true positives (TP) that differ between tracking prediction and ground truth.

From the same paper as PTQ, sPTQ (soft PTQ) penalizes the track ID discrepancy by subtracting the IoU scores at frames
with ID switches instead of the total number of ID switches. The sPTQ metric is defined as:
(∑{**1**(p,g) IoU(p,g)} - ∑(s)∈ IDS {s}) / (|TP|+ 0.5|FP|+ 0.5|FN|).

Each score is calculated separately for each class, and then the mean is computed across classes. Note that points of
class index 0 is ignored in the calculation.

## Leaderboard
nuScenes will maintain a single panoptic leaderboard with filters to split 4 specific tracks: Segmentation-lidar,
Segmentation-open, Tracking-lidar and Tracking-open. Submissions of the first two panoptic segmentation tracks will be
evaluated with segmentation metrics, while the two tracking submissions will be evaluated with both panoptic tracking
metrics as well as frame based panoptic segmentation metrics. For each submission the leaderboard will
list method aspects and evaluation metrics. Method aspects include input modalities (lidar, radar, vision), use of map
data and use of external data. To enable a fair comparison between methods, the user will be able to filter the methods
by method aspects.

Both panoptic segmentation and panoptic tracking tasks will have a `Lidar track` and `Open Track` respectively.
Methods will be compared within these tracks and the winners will be decided for each track separately.

**Segmentation-lidar track**:
* Only lidar input allowed.
* Only current frame and preceding frames within 0.5s are allowed, maximally 10 frames (inclusive of current frame).
* Only lidar panoptic annotations from Panoptic nuScenes are allowed.
* External data or map data <u>not allowed</u>.
* May use pre-training.

**Segmentation-open track**:
* Any sensor input allowed.
* All past data allowed, no future data allowed.
* All nuScenes, Panoptic nuScenes, nuScenes-lidarseg and nuImages annotations are allowed.
* External data and map data allowed.
* May use pre-training.

**Tracking-lidar track**:
* Only lidar input allowed.
* All past data allowed, no future data allowed.
* Only lidar panoptic annotations from Panoptic nuScenes are allowed.
* External data or map data <u>not allowed</u>.
* May use pre-training.

**Tracking-open track**:
* Any sensor input allowed.
* All past data allowed, no future data allowed.
* All nuScenes, Panoptic nuScenes, nuScenes-lidarseg and nuImages annotations are allowed.
* External data and map data allowed.
* May use pre-training.

**Details**:
* *Sensor input:*
For the lidar track we restrict the type of sensor input that may be used. Note that this restriction applies only at
test time. At training time any sensor input may be used.

* *Map data:*
By `map data` we mean using the *semantic* map provided in nuScenes. 

* *Meta data:*
Other meta data included in the dataset may be used without restrictions. E.g. bounding box annotations provided in
nuScenes, calibration parameters, ego poses, `location`, `timestamp`, `num_lidar_pts`, `num_radar_pts`, `translation`,
`rotation` and `size`. Note that .npz files, `instance`, `sample_annotation` and `scene` description are not provided
for the test set.

* *Pre-training:*
By pre-training we mean training a network for the task of image classification using only image-level labels, as done
in [[Krizhevsky NIPS 2012]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networ).
The pre-training may not involve bounding boxes, masks or other localized annotations.

* *Reporting:* 
Users are required to report detailed information on their method regarding sensor input, map data, meta data and pre-training.
Users that fail to adequately report this information may be excluded from the challenge. 
