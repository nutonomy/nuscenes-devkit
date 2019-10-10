# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: float = 50,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    pass  # TODO
