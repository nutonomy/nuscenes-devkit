# nuScenes predicton task
In this document we present the rules, submission process, and evaluation metrics of the nuScenes prediction task.
Click [here](http://evalai.cloudcv.org/web/challenges/challenge-page/356) for the **EvalAI detection evaluation server**.

## Overview
- [Introduction](#introduction)
- [Participation](#participation)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Evaluation metrics](#evaluation-metrics)

## Introduction
The goal of the nuScenes prediction task is to predict the future trajectories of objects in the nuScenes dataset.
A trajectory is a sequence of x-y locations. For this challenge, the predictions are 6-seconds long and sampled at
2 hertz.

## Challenges
To allow users to benchmark the performance of their method against the community, we will host a single leaderboard all-year round.
Additionally, we organize a number of challenges at leading Computer Vision and Machine Learning conference workshops.
Users that submit their results during the challenge period are eligible for awards.
Any user that cannot attend the workshop (direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

### Workshop on Benchmarking Progress in Autonomous Driving, ICRA 2020
The first nuScenes prediction challenge will be held at [ICRA 2020](https://www.icra2020.org/).
The submission period will open in early march and continue until May 28th, 2020.
Results and winners will be announced at the [Workshop on Benchmarking Progress in Autonomous Driving](http://montrealrobotics.ca/driving-benchmarks/).
Note that the evaluation server can still be used to benchmark your results after the challenge period.

## Submission rules
### Prediction-specific rules
* The user can submit up to 25 proposed future trajectories, called `modes`, for each agent along with a probability the agent follows that proposal. Our metrics (explained below) will measure how well this proposed set of trajectories matches the ground truth.
* Unlike previous challenges, the user will not submit their submissions to the eval server. Instead, we will run the user's model
on the test set for them. To make this possible, the user should modify the `do_inference_for_submission` function in the [inference script](https://github.com/nutonomy/nuscenes-devkit/blob/nuscenes-predict-challenge/python-sdk/nuscenes/eval/predict/do_inference.py)
so that their model is used.
* Up to two seconds of past history can be used to predict the future trajectory for each agent.
* Every submission to the challenge must be accompanied by a technical report describing the method in sufficient detail to allow for independent verification.

### General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Top leaderboard entries and their papers will be manually reviewed.
* Each user or team can have at most one one account on the evaluation server.
* Each user or team can submit at most 3 results. These results must come from different models, rather than submitting results from the same model at different training epochs or with slightly different parameters.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from all nuScenes challenges.

## Results format
The `do_inference_for_submission` function must produce a list of [`Predictions`](https://github.com/nutonomy/nuscenes-devkit/blob/nuscenes-predict-challenge/python-sdk/nuscenes/eval/predict/data_classes.py). A `Prediction` has the following components:

```
instance: Instance token for agent.
sample: Sample token for agent.
prediction: Numpy array of shape [num_modes, n_timesteps, state_dim]
probabilities: Numpy array of shape [num_modes]
```

Each agent in NuScenes is indexed by an instance token and a sample token. As mentioned previously, `num_modes` can be up to 25. Since we are making 6 second predictions at 2 Hz, `n_timesteps` is 12. We are concerned only with x-y coordinates, so `state_dim` is 2. Note that the prediction must be reported in **the agent's local coordinate frame**.
For the train and val sets the evaluation can be performed by the user on their local machine by making the modifications to `do_inference_fo_submission` and running the [eval_script](https://github.com/nutonomy/nuscenes-devkit/blob/nuscenes-predict-challenge/python-sdk/nuscenes/eval/predict/eval_pipeline.py) with the `--split_name` arg set to `train` or `val`.

## Evaluation metrics
Below we define the metrics for the nuScenes prediction task.

### Minimum Average Displacement Error over k (minADE_k)
The average of pointwise l2 distances between the predicted trajectory and ground truth over the `k` most likely predictions.

### Minimum Final Displacement Error over k (minFDE_k)
The final displacement error (FDE) is the l2 distance between the final points of the prediction and ground truth. We take the minimum FDE over the k most likely predictions and average over all agents.

### Hit Rate At 2 meters over k (HitRate_2_k)
If the maximum pointwise l2 distance between the prediction and ground truth is less than 2 meters, we define the prediction as a hit.
For each agent, we take the k most likely predictions and evaluate if any are hits. The HitRate_2k is the proportion of hits over all agents.

### Configuration
The default evaluation metrics configurations can be found in this [file](https://github.com/nutonomy/nuscenes-devkit/blob/nuscenes-predict-challenge/python-sdk/nuscenes/eval/predict/configs/predict_2020_icra.json).
