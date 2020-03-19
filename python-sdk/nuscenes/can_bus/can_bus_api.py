# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import argparse
import json
import os
import re
import warnings
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as scipy_dist


class NuScenesCanBus:
    """
    This class encapsulates the files of the nuScenes CAN bus expansion set.
    It can be used to access the baseline navigation route as well as the various CAN bus messages.
    """

    def __init__(self,
                 dataroot: str = '/data/sets/nuscenes',
                 max_misalignment: float = 5.0):
        """
        Initialize the nuScenes CAN bus API.
        :param dataroot: The nuScenes directory where the "can" folder is located.
        :param max_misalignment: Maximum distance in m that any pose is allowed to be away from the route.
        """
        # Check that folder exists.
        self.can_dir = os.path.join(dataroot, 'can_bus')
        if not os.path.isdir(self.can_dir):
            raise Exception('Error: CAN bus directory not found: %s. Please download it from '
                            'https://www.nuscenes.org/download' % self.can_dir)

        # Define blacklist for scenes where route and ego pose are not aligned.
        if max_misalignment == 5.0:
            # Default settings are hard-coded for performance reasons.
            self.route_blacklist = [
                71, 73, 74, 75, 76, 85, 100, 101, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119,
                261, 262, 263, 264, 276, 302, 303, 304, 305, 306, 334, 388, 389, 390, 436, 499, 500, 501, 502, 504,
                505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 517, 518, 547, 548, 549, 550, 551, 556, 557,
                558, 559, 560, 561, 562, 563, 564, 565, 730, 731, 733, 734, 735, 736, 737, 738, 778, 780, 781, 782,
                783, 784, 904, 905, 1073, 1074
            ]
        else:
            misaligned = self.list_misaligned_routes()
            self.route_blacklist = [int(s[-4:]) for s in misaligned]

        # Define blacklist for scenes without CAN bus data.
        self.can_blacklist = [
            161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314
        ]

        # Define all messages.
        self.can_messages = [
            'ms_imu', 'pose', 'steeranglefeedback', 'vehicle_monitor', 'zoesensors', 'zoe_veh_info'
        ]
        self.derived_messages = [
            'meta', 'route'
        ]
        self.all_messages = self.can_messages + self.derived_messages

    def print_all_message_stats(self,
                                scene_name: str) -> None:
        """
        Prints the meta stats for each CAN message type of a particular scene.
        :param scene_name: The name of the scene, e.g. scene-0001.
        """
        all_messages = {}
        for message_name in self.can_messages:
            messages = self.get_messages(scene_name, 'meta')
            all_messages[message_name] = messages
        print(json.dumps(all_messages, indent=2))

    def print_message_stats(self,
                            scene_name: str,
                            message_name: str) -> None:
        """
        Prints the meta stats for a particular scene and message name.
        :param scene_name: The name of the scene, e.g. scene-0001.
        :param message_name: The name of the CAN bus message type, e.g. ms_imu.
        """
        assert message_name != 'meta', 'Error: Cannot print stats for meta '
        messages = self.get_messages(scene_name, 'meta')
        print(json.dumps(messages[message_name], indent=2))

    def plot_baseline_route(self,
                            scene_name: str,
                            out_path: str = None) -> None:
        """
        Plot the baseline route and the closest ego poses for a scene.
        Note that the plot is not closed and should be closed by the caller.
        :param scene_name: The name of the scene, e.g. scene-0001.
        :param out_path: Output path to dump the plot to. Ignored if None.
        """
        # Get data.
        route, pose = self.get_pose_and_route(scene_name)

        # Visualize.
        plt.figure()
        plt.plot(route[:, 0], route[:, 1])
        plt.plot(pose[:, 0], pose[:, 1])
        plt.plot(pose[0, 0], pose[0, 1], 'rx', MarkerSize=10)
        plt.legend(('Route', 'Pose', 'Start'))
        plt.xlabel('Map coordinate x in m')
        plt.ylabel('Map coordinate y in m')
        if out_path is not None:
            plt.savefig(out_path)
        plt.show()

    def plot_message_data(self,
                          scene_name: str,
                          message_name: str,
                          key_name: str,
                          dimension: int = 0,
                          out_path: str = None,
                          plot_format: str = 'b-') -> None:
        """
        Plot the data for a particular message.
        :param scene_name: The name of the scene, e.g. scene-0001.
        :param message_name: The name of the CAN bus message type, e.g. ms_imu.
        :param key_name: The name of the key in the message, e.g. linear_accel.
        :param dimension: Which dimension to render (default is 0). If -1, we render the norm of the values.
        :param out_path: Output path to dump the plot to. Ignored if None.
        :param plot_format: A string that describes a matplotlib format, by default 'b-' for a blue dashed line.
        """
        # Get data.
        messages = self.get_messages(scene_name, message_name)
        data = np.array([m[key_name] for m in messages])
        utimes = np.array([m['utime'] for m in messages])

        # Convert utimes to seconds and subtract the minimum.
        utimes = (utimes - min(utimes)) / 1e6

        # Take selected column.
        if dimension == -1:
            data = np.linalg.norm(data, axis=1)
        elif dimension == 0:
            pass
        elif data.ndim > 1 and data.shape[1] >= dimension + 1:
            data = data[:, dimension]
        else:
            raise Exception('Error: Invalid dimension %d for key "%s"!' % (dimension, key_name))

        # Render.
        plt.figure()
        plt.plot(utimes, data, plot_format, markersize=1)
        plt.title(scene_name)
        plt.xlabel('Scene time in s')
        plt.ylabel('%s - %s' % (message_name, key_name))
        if out_path is not None:
            plt.savefig(out_path)
        plt.show()

    def list_misaligned_routes(self,
                               max_misalignment: float = 5.0) -> List[str]:
        """
        Print all scenes where ego poses and baseline route are misaligned.
        We use the Hausdorff distance to decide on the misalignment.
        :param max_misalignment: Maximum distance in m that any pose is allowed to be away from the route.
        :return: A list of all the names of misaligned scenes.
        """
        # Get all scenes.
        all_files = os.listdir(self.can_dir)
        scene_list = list(np.unique([f[:10] for f in all_files]))  # Get the scene name from e.g. scene-0123_meta.json.

        # Init.
        misaligned = []

        for scene_name in scene_list:
            # Get data.
            route, pose = self.get_pose_and_route(scene_name, print_warnings=False)

            # Filter by Hausdorff distance.
            dists = scipy_dist.cdist(pose, route)
            max_dist = np.max(np.min(dists, axis=1))
            if max_dist > max_misalignment:
                misaligned.append(scene_name)

        return misaligned

    def get_pose_and_route(self,
                           scene_name: str,
                           print_warnings: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the route and pose for a scene as numpy arrays.
        :param scene_name: The name of the scene, e.g. scene-0001.
        :param print_warnings: Whether to print out warnings if the requested data is not available or not reliable.
        :return: A tuple of route and pose arrays (each point is 2d).
        """
        # Load baseline route and poses.
        route = self.get_messages(scene_name, 'route', print_warnings=print_warnings)
        pose = self.get_messages(scene_name, 'pose', print_warnings=print_warnings)

        # Convert to numpy format.
        route = np.asarray(route)
        pose = np.asarray([p['pos'][:2] for p in pose])

        return route, pose

    def get_messages(self,
                     scene_name: str,
                     message_name: str,
                     print_warnings: bool = True) -> Union[List[Dict], Dict]:
        """
        Retrieve the messages for a particular scene and message type.
        :param scene_name: The name of the scene, e.g. scene-0001.
        :param message_name: The name of the CAN bus message type, e.g. ms_imu.
        :param print_warnings: Whether to print out warnings if the requested data is not available or not reliable.
        :return: The raw contents of the message type, either a dict (for `meta`) or a list of messages.
        """
        # Check inputs. Scene names must be in the format scene-0123.
        assert re.match('^scene-\\d\\d\\d\\d$', scene_name)
        assert message_name in self.all_messages, 'Error: Invalid CAN bus message name: %s' % message_name

        # Check for data issues.
        scene_id = int(scene_name[-4:])
        if scene_id in self.can_blacklist:
            # Check for logs that have no CAN bus data.
            raise Exception('Error: %s does not have any CAN bus data!' % scene_name)
        elif print_warnings:
            # Print warnings for scenes that are known to have bad data.
            if message_name == 'route':
                if scene_id in self.route_blacklist:
                    warnings.warn('Warning: %s is not well aligned with the baseline route!' % scene_name)
            elif message_name == 'vehicle_monitor':
                if scene_id in [419]:
                    warnings.warn('Warning: %s does not have any vehicle_monitor messages!')

        # Load messages.
        message_path = os.path.join(self.can_dir, '%s_%s.json' % (scene_name, message_name))
        with open(message_path, 'r') as f:
            messages = json.load(f)
        assert type(messages) in [list, dict]

        # Rename all dict keys to lower-case.
        if isinstance(messages, dict):
            messages = {k.lower(): v for (k, v) in messages.items()}

        return messages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot stats for the CAN bus API.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes')
    parser.add_argument('--scene_name', type=str, default='scene-0028')
    parser.add_argument('--message_name', type=str, default='steeranglefeedback')
    parser.add_argument('--key_name', type=str, default='value')
    args = parser.parse_args()

    nusc_can = NuScenesCanBus(dataroot=args.dataroot)
    if args.message_name == 'route+pose':
        nusc_can.plot_baseline_route(args.scene_name)
    else:
        nusc_can.plot_message_data(args.scene_name, args.message_name, args.key_name)
