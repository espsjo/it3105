from abc import ABC, abstractmethod
from numpy import ndarray

"""
Abstract class for the SimWorlds, defining functions vital for interacting with the environment
"""


class SimWorldAbs(ABC):
    @abstractmethod
    def reset_states(self, player_start, visualize) -> None:
        pass

    @abstractmethod
    def play_move(self, move) -> bool:
        pass

    @abstractmethod
    def get_legal_moves(self) -> ndarray:
        pass

    @abstractmethod
    def is_won(self) -> bool:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def get_state(self, flatten, include_turn) -> ndarray or int:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass

    @abstractmethod
    def set_state(self, state: ndarray or int) -> None:
        pass

    @abstractmethod
    def set_player(self, player: int) -> None:
        pass

    @abstractmethod
    def get_board_hist(self) -> ndarray:
        pass
