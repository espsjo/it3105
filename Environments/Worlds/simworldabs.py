from abc import ABC, abstractmethod
from numpy import ndarray


class SimWorldAbs(ABC):
    @abstractmethod
    def reset_states(self) -> None:
        pass

    @abstractmethod
    def play_move(self) -> bool:
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
    def get_state(self) -> ndarray or int:
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        pass
