import numpy as np
import time
from .simworldabs import SimWorldAbs


class NIM(SimWorldAbs):
    """
    Initialise variables
    Returns void
    """

    def __init__(self, nim_config, visualize: bool):
        self.stones = nim_config["STONES"]
        self.min_stones = nim_config["MIN_STONES"]
        self.max_stones = nim_config["MAX_STONES"]
        self.delay = nim_config["DELAY"]
        self.verbose = visualize
        self.won_msg = nim_config["WON_MSG" or self.verbose]
        self.reset_states(player_start=1, visualize=self.verbose)

    """
    Resets the states
    Returns void
    """

    def reset_states(self, player_start, visualize: bool = None):

        # Update the verbose variable after reset
        self.verbose = visualize if visualize != None else self.verbose

        # Update the state to represent number of stones
        self.state = self.stones

        # Initialise player turn
        self.player = player_start

        # Variable for winner of game (0/1/2)
        self.winner = 0

        # History of states
        self.board_hist = []

        self.legal_moves = self.get_legal_moves()

        if self.verbose:
            print(
                f"""
## NEW GAME OF NIM (Stones = {self.stones}, min: {self.min_stones}, max: {self.max_stones}) ##"""
            )

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

        if self.verbose:
            print(
                f"""
There are {self.state} remaining stone(s).
Player {self.player} has chosen to take {move} stone(s)
Now, there are {new_state} stone(s) left."""
            )

        self.state = new_state

        # Update legal moves
        self.legal_moves = self.get_legal_moves()

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

        time.sleep(self.delay)

    """
    Finds every legal move, given the current state
    Returns array
    """

    def get_legal_moves(self):
        min_move = self.min_stones
        max_move = min(self.max_stones, self.state)
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

    def get_state(self, flatten, include_turn):
        return self.state

    """
    Function for checking if the game is over
    Returns bool
    """

    def is_won(self):
        return bool(self.winner)
