from nuscenes.predict import PredictHelper
from nuscenes.predict.models import PhysicsOracle, ConstantVelocityHeading
from nuscenes.utils.splits import get_prediction_challenge_split
from nuscenes import NuScenes
import json
import os

def main(version: str, split_name: str, output_dir: str) -> None:

    nusc = NuScenes(version=version)
    helper = PredictHelper(nusc)
    dataset = get_prediction_challenge_split(split_name)
    oracle = PhysicsOracle(helper)
    cv_heading = ConstantVelocityHeading(helper)

    cv_preds = []
    oracle_preds = []
    for token in dataset:
        cv_preds.append(cv_heading(token).serialize())
        oracle_preds.append(oracle(token).serialize())

    json.dump(cv_preds, open(os.path.join(output_dir, "cv_preds.json"), "w"))
    json.dump(oracle_preds, open(os.path.join(output_dir, "oracle_preds.json"), "w"))


if __name__ == "__main__":

    main("v1.0-mini", "mini_val", "/home/freddyboulton/Desktop/")


