from typing import List, Dict

import numpy as np


def get_scenarios() -> List[Dict[str, dict]]:
    """ """

    scenarios = []

    # Scenario 1.
    # Parallel motion 1 meter distance.
    pos_gt = np.array([[(1, -3), (1, -2), (1, -1), (1, -0)],
                      [(0, -3), (0, -2), (0, -1), (0, -0)], ]).astype(float)
    pos_pred = pos_gt
    sigma = 0.1
    pos_pred += sigma * np.random.randn(*pos_pred.shape)

    input_data = {'pos_gt':  pos_gt,
                  'pos_pred': pos_pred}
    output_data = {'ids': 0.0}

    scenarios.append({'input': input_data, 'output': output_data})

    # Scenario 2.
    # Parallel motion bring closer predictions.
    pos_gt = np.array([[(1, -3), (1, -2), (1, -1), (1, -0)],
                      [(0, -3), (0, -2), (0, -1), (0, -0)], ]).astype(float)
    pos_pred = pos_gt

    pos_pred[0, :, 0] -= 0.3
    pos_pred[1, :, 0] += 0.3
    sigma = 0.1
    pos_pred += sigma * np.random.randn(*pos_pred.shape)

    input_data = {'pos_gt':  pos_gt,
                  'pos_pred': pos_pred}
    output_data = {'ids': 0.0}

    scenarios.append({'input': input_data, 'output': output_data})

    # Scenario 3.
    # Parallel motion bring closer both ground truth and predictions.
    pos_gt = np.array([[(1, -3), (1, -2), (1, -1), (1, -0)],
                      [(0, -3), (0, -2), (0, -1), (0, -0)], ]).astype(float)
    pos_pred = pos_gt

    pos_gt[0, :, 0] -= 0.3
    pos_gt[1, :, 0] += 0.3
    pos_pred[0, :, 0] -= 0.3
    pos_pred[1, :, 0] += 0.3
    sigma = 0.1
    pos_pred += sigma * np.random.randn(*pos_pred.shape)

    input_data = {'pos_gt':  pos_gt,
                  'pos_pred': pos_pred}
    output_data = {'ids': 0.0}

    scenarios.append({'input': input_data, 'output': output_data})

    # Scenario 4.
    # Crossing motion.
    pos_gt = np.array([[(2, -3), (1, -2), (0, -1), (-1, -0)],
                      [(-2, -3), (-1, -2), (0, -1), (1, -0)], ]).astype(float)
    pos_pred = pos_gt
    sigma = 0.1
    pos_pred += sigma * np.random.randn(*pos_pred.shape)

    input_data = {'pos_gt':  pos_gt,
                  'pos_pred': pos_pred}
    output_data = {'ids': 0.0}

    scenarios.append({'input': input_data, 'output': output_data})

    # Scenario 5.
    # Identity switch due to a single misdetection (3rd timestamp).
    pos_pred = np.array([
        [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
        [(-2, 0), (-1, 0), (3, 0), (1, 0), (2, 0)],
    ]).astype(float)
    pos_gt = np.array([
        [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)],
    ]).astype(float)
    sigma = 0.1
    pos_pred += sigma * np.random.randn(*pos_pred.shape)

    input_data = {'pos_gt':  pos_gt,
                  'pos_pred': pos_pred}
    output_data = {'ids': 2}

    scenarios.append({'input': input_data, 'output': output_data})

    return scenarios
