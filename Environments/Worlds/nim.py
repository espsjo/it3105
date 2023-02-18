import numpy as np
import time
from .simworldabs import SimWorldAbs
from numpy import ndarray


class NIM(SimWorldAbs):
    """
    Initialise variables
    Returns void
    """

    def __init__(self, nim_config):
        self.stones = nim_config["STONES"]
        self.min_stones = nim_config["MIN_STONES"]
        self.max_stones = nim_config["MAX_STONES"]
        self.delay = nim_config["DELAY"]
        self.won_msg = nim_config["WON_MSG"]
        self.reset_states(player_start=1)

    """
    Resets the states
    Returns void
    """

    def reset_states(self, player_start=1):

        # Update the state to represent number of stones
        self.state = self.stones

        # Initialise player turn
        self.player = player_start

        # Variable for winner of game (0/1/2)
        self.winner = 0

        # History of states
        self.board_hist = []

        self.legal_moves = self.get_legal_moves(self.state)

    """
    Checks if a player has won
    Returns bool
    """

    def is_winning(self):
        return self.state == 0

    """
    Checks if a move is legal
    Returns bool
    """

    def check_valid_move(self, move):
        min_move = self.min_stones
        max_move = min(self.max_stones, self.state)
        return min_move <= move <= max_move

    """
    Amend the state based on the played move, given that it is legal
    Returns bool
    """

    def play_move(self, move):
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

    """
    Finds every legal move, given the current state
    Returns array
    """

    def get_legal_moves(self, state=None):
        if state is None:
            return self.legal_moves
        min_move = self.min_stones
        if isinstance(state, (list, ndarray)):
            state = state[-1]
        max_move = min(self.max_stones, state)
        return np.array([num for num in range(min_move, max_move + 1)])

    """  
    Function for simulating a reward
    Returns int
    """

    def get_reward(self, player):
        if self.winner == 0:
            return 0
        if self.winner != player:
            return -1
        else:
            return 1

    """ 
    Function for getting the current player
    Returns int
    """

    def get_current_player(self):
        return self.player

    """ 
    Function for getting the current state 
    Does not have option to include turn, as it is unnecessary for the given game
    Returns array
    """

    def get_state(self, flatten=False, include_turn=False):
        if include_turn:
            return [self.player, self.state]
        return [self.state]

    """
    Function for checking if the game is over
    Returns bool
    """

    def is_won(self):
        return bool(self.winner)

    """
    Function for setting state
    Returns void
    """

    def set_state(self, state: ndarray or int):
        if isinstance(state, ndarray):
            self.state = state[0]
        self.state = state

    """
    Function for setting player
    Returns void
    """

    def set_player(self, player: int) -> None:
        if player == 1 or player == 2:
            self.player = player

    """
    Function for returning the board history
    Returns array
    """

    def get_board_hist(self):
        return np.array(self.board_hist)

    """
    Function for retuning how many actions all possible actions, legal or not
    Returns array
    """

    def get_actions(self) -> ndarray:
        return np.array([i for i in range(self.min_stones, self.max_stones + 1)])

    """ 
    Function for returning winner
    Returns int
    """

    def get_winner(self) -> int:
        return self.winner
