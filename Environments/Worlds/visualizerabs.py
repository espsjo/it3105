from abc import ABC, abstractmethod
from numpy import ndarray
from .simworldabs import SimWorldAbs


class VisualizerAbs(ABC):
    """
    Abstract class for visualizers, defining functions vital for interacting with the environment
    """

    @abstractmethod
    def __init__(self, game_config):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def visualize_move(self, env: SimWorldAbs, move) -> None:
        pass
