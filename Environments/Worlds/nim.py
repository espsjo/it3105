import numpy as np
import time
from .simworldabs import SimWorldAbs
from numpy import ndarray


class NIM(SimWorldAbs):
    """
    Class for representing logic and state handling in the NIM game
    Parameters:
        nim_config: (dict) Config for the NIM game
    """

    def __init__(self, nim_config):
        self.stones = nim_config["STONES"]
        self.min_stones = nim_config["MIN_STONES"]
        self.max_stones = nim_config["MAX_STONES"]
        self.delay = nim_config["DELAY"]
        self.won_msg = nim_config["WON_MSG"]
        self.reset_states(player_start=1)

    def reset_states(self, player_start=1) -> None:
        """
        Resets the states
        Parameters:
            player_start: (int) Specify player to start (optional)
        Returns:
            None
        """
        # Update the state to represent number of stones
        self.state = self.stones

        # Initialise player turn
        self.player = player_start

        # Variable for winner of game (0/1/2)
        self.winner = 0

        # History of states
        self.board_hist = []

        self.legal_moves = self.get_legal_moves(self.state)

    def is_winning(self) -> bool:
        """
        Checks if a player has won
        Parameters:
            None
        Returns:
            bool
        """
        return self.state == 0

    def check_valid_move(self, move) -> bool:
        """
        Checks if a move is legal
        Parameters:
            move: (int) Move to check
        Returns:
            bool
        """
        min_move = self.min_stones
        max_move = min(self.max_stones, self.state)
        return min_move <= move <= max_move

    def play_move(self, move) -> bool:
        """
        Amend the state based on the played move, given that it is legal
        Parameters:
            move: (int) Move to be played
        Returns:
            bool
        """
        if self.winner != 0:
            return False
        if not self.check_valid_move(move):
            return False

        self.board_hist.append(self.state)

        new_state = self.state

        # Update state
        new_state -= move

        self.state = new_state

        # Update legal moves
        self.legal_moves = self.get_legal_moves(self.state)

        # Check for win
        if self.is_winning():
            self.winner = self.player
            self.board_hist.append(self.state)
            if self.won_msg:
                print(f"\nThe winner is player {self.winner}")
        # Update player turn
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

        return True

    def get_legal_moves(self, state=None) -> np.ndarray:
        """
        Finds every legal move, given the current state
        Parameters:
            state: (int) The current state representation
        Returns:
            np.ndarray
        """
        if state is None:
            return self.legal_moves
        min_move = self.min_stones
        if isinstance(state, (list, ndarray)):
            state = state[-1]
        max_move = min(self.max_stones, state)
        return np.array([num for num in range(min_move, max_move + 1)])

    def get_reward(self, player) -> int:
        """
        Function for simulating a reward
        Parameters:
            player: (int) Player to check reward for
        Returns:
            int
        """
        if self.winner == 0:
            return 0
        if self.winner != player:
            return -1
        else:
            return 1

    def get_current_player(self) -> int:
        """
        Function for getting the current player
        Parameters:
            None
        Returns:
            int
        """
        return self.player

    def get_state(self, flatten=False, include_turn=False) -> np.ndarray:
        """
        Function for getting the current state
        Does not have option to include turn, as it is unnecessary for the given game
        Parameters:
            flatten: (bool) To flatten state or not (Irrelevant here, but required from abstract)
            include_turn: (bool) To include the player turn
        Returns:
            np.ndarray
        """
        if include_turn:
            return np.array([self.player, self.state])
        return np.array([self.state])

    def is_won(self) -> bool:
        """
        Function for checking if the game is over
        Parameters:
            None
        Returns:
            bool
        """
        return bool(self.winner)

    def set_state(self, state: ndarray or int) -> None:
        """
        Function for setting state
        Parameters:
            state: (np.ndarray / int) State to be set
        Returns:
            None
        """
        if isinstance(state, ndarray):
            self.state = state[-1]
        self.state = state

    def set_player(self, player: int) -> None:
        """
        Function for setting player
        Parameters:
            player: (int) Player to be set
        Returns:
            None
        """
        if player == 1 or player == 2:
            self.player = player

    def get_board_hist(self) -> np.ndarray:
        """
        Function for returning the board history
        Parameters:
            None
        Returns:
            np.ndarray
        """
        return np.array(self.board_hist)

    def get_actions(self) -> ndarray:
        """
        Function for retuning how many actions all possible actions, legal or not
        Parameters:
            None
        Returns:
            np.ndarray
        """
        return np.array([i for i in range(self.min_stones, self.max_stones + 1)])

    def get_winner(self) -> int:
        """
        Function for returning winner
        Parameters:
            None
        Returns:
            int
        """
        return self.winner
