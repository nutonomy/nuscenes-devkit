# nuScenes dev-kit.
# Code written by Freddy Boulton 2020.
import abc
from typing import List

import numpy as np


class StaticLayerRepresentation(abc.ABC):
    """ Represents static map information as a numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class AgentRepresentation(abc.ABC):
    """ Represents information of agents in scene as numpy array. """

    @abc.abstractmethod
    def make_representation(self, instance_token: str, sample_token: str) -> np.ndarray:
        raise NotImplementedError()


class Combinator(abc.ABC):
    """ Combines the StaticLayer and Agent representations into a single one. """

    @abc.abstractmethod
    def combine(self, data: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()


class InputRepresentation:
    """
    Specifies how to represent the input for a prediction model.
    Need to provide a StaticLayerRepresentation - how the map is represented,
    an AgentRepresentation - how agents in the scene are represented,
    and a Combinator, how the StaticLayerRepresentation and AgentRepresentation should be combined.
    """

    def __init__(self, static_layer: StaticLayerRepresentation, agent: AgentRepresentation,
                 combinator: Combinator):

        self.static_layer_rasterizer = static_layer
        self.agent_rasterizer = agent
        self.combinator = combinator

    def make_input_representation(self, instance_token: str, sample_token: str) -> np.ndarray:

        static_layers = self.static_layer_rasterizer.make_representation(instance_token, sample_token)
        agents = self.agent_rasterizer.make_representation(instance_token, sample_token)

        return self.combinator.combine([static_layers, agents])

