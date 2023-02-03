from abc import ABC, abstractmethod
from numpy import ndarray
from .simworldabs import SimWorldAbs

"""
Abstract class for visualizers, defining functions vital for interacting with the environment
"""


class VisualizerAbs(ABC):
    @abstractmethod
    def __init__(self, game_config):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def visualize_move(self, env: SimWorldAbs, move) -> None:
        pass
