# nuScenes prediction task
![nuScenes Prediction logo](https://www.nuscenes.org/public/images/prediction.png)

## Overview
- [Introduction](#introduction)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Evaluation metrics](#evaluation-metrics)

## Introduction
The goal of the nuScenes prediction task is to predict the future trajectories of objects in the nuScenes dataset.
A trajectory is a sequence of x-y locations. For this challenge, the predictions are 6-seconds long and sampled at
2 hertz.

## Challenges
To allow users to benchmark the performance of their method against the community, we will host a single leaderboard all year round.
Additionally, we intend to organize a number of challenges at leading Computer Vision and Machine Learning conference workshops.
Users that submit their results during the challenge period are eligible for awards. These awards may be different for each challenge.

### Workshop on Benchmarking Progress in Autonomous Driving, ICRA 2020
The first nuScenes prediction challenge will be held at [ICRA 2020](https://www.icra2020.org/).
The submission period will open April 1 and continue until May 28th, 2020.
Results and winners will be announced at the [Workshop on Benchmarking Progress in Autonomous Driving](http://montrealrobotics.ca/driving-benchmarks/).
Note that the evaluation server can still be used to benchmark your results after the challenge period.

*Update:* Due to the COVID-19 situation, participants are **not** required to attend in person
to be eligible for the prizes.

## Submission rules
### Prediction-specific rules
* The user can submit up to 25 proposed future trajectories, called `modes`, for each agent along with a probability the agent follows that proposal. Our metrics (explained below) will measure how well this proposed set of trajectories matches the ground truth.
* Up to two seconds of past history can be used to predict the future trajectory for each agent.
* Unlike previous challenges, the leaderboard will be ranked according to performance on the nuScenes val set. This is because we cannot release the annotations on the test set, so users would not be able to run their models on the test set and then submit their predictions to the server. To prevent overfitting on the val set, the top 5 submissions on the leaderboard will be asked to send us their code and we will run their model on the test set. The winners will be chosen based on their performance on the test set, not the val set.
* Every submission to the challenge must be accompanied by a technical report describing the method in sufficient detail to allow for independent verification.

### General rules
* We release annotations for the train and val set, but not for the test set. We have created a hold out set for validation
from the training set called the `train_val` set.
* We release sensor data for train, val and test set.
* Top leaderboard entries and their papers will be manually reviewed to ensure no cheating was done.
* Each user or team can have at most one one account on the evaluation server.
* Each user or team can submit at most 3 results. These results must come from different models, rather than submitting results from the same model at different training epochs or with slightly weights or hyperparameter values.
* Any attempt to make more submissions than allowed will result in a permanent ban of the team or company from all nuScenes challenges.

## Results format
Users must submit a json file with a list of [`Predictions`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/data_classes.py) for each agent. A `Prediction` has the following components:

```
instance: Instance token for agent.
sample: Sample token for agent.
prediction: Numpy array of shape [num_modes, n_timesteps, state_dim]
probabilities: Numpy array of shape [num_modes]
```

Each agent in nuScenes is indexed by an instance token and a sample token. As mentioned previously, `num_modes` can be up to 25. Since we are making 6 second predictions at 2 Hz, `n_timesteps` is 12. We are concerned only with x-y coordinates, so `state_dim` is 2. Note that the prediction must be reported in **the global coordinate frame**.
Consult the [`baseline_model_inference`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/baseline_model_inference.py) script for an example on how to make a submission for two physics-based baseline models.

## Evaluation metrics
Below we define the metrics for the nuScenes prediction task.

### Minimum Average Displacement Error over k (minADE_k)
The average of pointwise L2 distances between the predicted trajectory and ground truth over the `k` most likely predictions.

### Minimum Final Displacement Error over k (minFDE_k)
The final displacement error (FDE) is the L2 distance between the final points of the prediction and ground truth. We take the minimum FDE over the k most likely predictions and average over all agents.

### Miss Rate At 2 meters over k (MissRate_2_k)
If the maximum pointwise L2 distance between the prediction and ground truth is greater than 2 meters, we define the prediction as a miss.
For each agent, we take the k most likely predictions and evaluate if any are misses. The MissRate_2_k is the proportion of misses over all agents.

### Configuration
The metrics configuration file for the ICRA 2020 challenge can be found in this [file](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/configs/predict_2020_icra.json).
