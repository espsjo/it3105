from abc import ABC, abstractmethod
from numpy import ndarray


class SimWorldAbs(ABC):
    """
    Abstract class for the SimWorlds, defining functions vital for interacting with the environment
    """

    @abstractmethod
    def reset_states(self, player_start) -> None:
        pass

    @abstractmethod
    def play_move(self, move) -> bool:
        pass

    @abstractmethod
    def get_legal_moves(self, state=None) -> ndarray:
        pass

    @abstractmethod
    def is_won(self) -> bool:
        pass

    @abstractmethod
    def get_reward(self, player) -> float:
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

    @abstractmethod
    def get_actions(self) -> ndarray:
        pass

    @abstractmethod
    def get_winner(self) -> int:
        pass
