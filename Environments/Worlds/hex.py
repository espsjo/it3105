import numpy as np
from copy import deepcopy
from .hex_gui import HexGUI
from .simworldabs import SimWorldAbs
from numpy import ndarray

"""
Class for the Hex game, based on the SimWorldAbs abstract class. More extensive than need be, but that is in many cases to 
account for the ability to process moves on both forms: (int,int) and int. When all functionality is implemented, this code 
could be reduced to only cover necessary cases.
"""


class Hex(SimWorldAbs):
    """
    Initialise variables
    Parameters:
        hex_config: (dict) Config file for hex game
    """

    def __init__(self, hex_config):
        self.board_size = hex_config["BOARD_SIZE"]
        self.won_msg = hex_config["WON_MSG"]
        self.win_cond = set([i for i in range(self.board_size)])
        self.reset_states(player_start=1)

    def reset_states(self, player_start=1) -> None:

        """
        Reset state, and variables containing state information
        Parameters:
            player_start: (int) Specify which player to start (optional)
        Returns:
            None
        """
        # Represented with 1 and 2
        self.player = player_start

        # 0: Open cell, 1: Player 1, 2: Player 2
        self.state = np.zeros((self.board_size, self.board_size))

        # Legal moves are represented as indexes in the flattened state
        self.legal_moves = self.get_legal_moves(self.state)

        # Variable for which player eventually wins. If != 0, the game should stop
        self.winner = 0

        # Store the winning island
        self.winning_island = None

        # Dictionary of disjoint sets belonging to each player
        self.nodesets = {1: [], 2: []}

        # History of board
        self.board_hist = []

    def is_winning(self, player):
        """
        Function for validating if a player has won the current game or not
        Parameters:
            player: (int) What player to check
        Returns:
            bool, set
        """

        disjoints = self.nodesets[player]
        # Player 1 wants to get a connected island where the y-coord (row) counts from 0 to board_size-1
        coord = 1
        # Player 2 wants to get a connected island where the x-coord (col) counts from 0 to board_size-1
        if player == 2:
            coord = 0

        for disj in disjoints:
            if len(disj) < self.board_size:
                continue
            x_y = set(
                [self.convert_node(flat_n, isflattened=True)[coord] for flat_n in disj]
            )
            if x_y == self.win_cond:
                return True, disj

        return False, None

    def get_winner(self) -> int:
        """
        Function for returning winner
        Parameters:
            None
        Returns:
            int
        """
        return self.winner

    def check_valid_move(self, move) -> bool:
        """
        Function for validating a move
        Parameters:
            move: (int / tuple) Check if a move is valid
        Returns:
            bool
        """

        if not isinstance(move, tuple):
            return move in self.legal_moves
        else:
            return self.convert_node(move, isflattened=False) in self.legal_moves

    def play_move(self, move) -> bool:
        """
        Function for updating the state with a move, given that the move is legal.
        Parameters:
            move: (int / tuple) Move to be played
        Returns:
            bool
        """
        if self.winner != 0:
            return False
        if not self.check_valid_move(move):
            return False

        self.board_hist.append(deepcopy(self.state))

        new_state = self.state
        flattened = not isinstance(move, tuple)

        if flattened:
            flat_move = move
            move = self.convert_node(move, isflattened=True)
        else:
            flat_move = self.convert_node(move, isflattened=False)
            move = move

        # Update state
        new_state[move[0]][move[1]] = self.player
        self.state = new_state
        # Update legal moves
        self.legal_moves = self.get_legal_moves(self.state)

        # Update disjoint sets
        self.update_disjoint_sets(flat_move, self.player)

        # Check for win
        winning, self.winning_island = self.is_winning(self.player)
        if winning:
            self.winner = self.player
            # Add the final state
            self.board_hist.append(deepcopy(self.state))
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
        Function for returning legal moves, specifically for redistributing the probabilities in ANET to not include illegal moves
        Parameters:
            state: (np.ndarray / list) State representation
        Returns:
            np.ndarray
        """
        if state is None:
            return self.legal_moves
        state = np.array(self.flatten_state(state))
        return np.where(state == 0)[0]

    def is_won(self) -> bool:
        """
        Function for checking if the game is over
        Parameters:
            None
        Returns:
            bool
        """
        return bool(self.winner)

    def flatten_state(self, state, include_turn: bool = False) -> np.ndarray:
        """
        Function for flattening the state, in variation with or without the current player turn
        Parameters:
            state: (list / np.ndarray) State representation
            include_turn: (bool) To include the turn in the flattened state (optional)
        Returns:
            np.ndarray
        """
        flat_state = state.flatten()
        if include_turn:
            flat_state = np.insert(flat_state, 0, self.player, axis=0)
        return flat_state

    def get_reward(self, player) -> int:
        """
        Function for simulating a reward
        Parameters:
            player: (int) What player to simulate for
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

    def get_state(
        self, flatten: bool = False, include_turn: bool = False
    ) -> np.ndarray:
        """
        Function for getting the current state, either flattened or matrix, with the option to include turn in both
        Parameters:
            flatten: (bool) To flatten the state or not
            include_turn: (bool) To include the turn or not
        Returns:
            np.ndarray
        """
        state = self.state
        if flatten:
            state = self.flatten_state(state, include_turn=include_turn)
        elif include_turn:
            state = np.insert(state, 0, self.player)
        return state

    def get_neighbours(self, node, returnFlattened: bool) -> list:
        """
        Function for getting the neighbours of a given "node"
        Parameters:
            node: (tuple / int) A cell in the state
            returnFlattened: (bool) To return flattened neighbours
        Returns:
            list
        """

        if not isinstance(node, tuple):
            node = self.convert_node(node, isflattened=True)

        r = node[0]
        c = node[1]
        pot_neighbours = [
            (r - 1, c),
            (r - 1, c + 1),
            (r, c - 1),
            (r, c + 1),
            (r + 1, c - 1),
            (r + 1, c),
        ]
        neighbours = []
        for n in pot_neighbours:
            x = n[0]
            y = n[1]

            if 0 <= x <= self.board_size - 1:
                if 0 <= y <= self.board_size - 1:
                    if returnFlattened:
                        neighbours.append(self.convert_node(n, isflattened=False))
                    else:
                        neighbours.append(n)
        return neighbours

    def update_disjoint_sets(self, move, player) -> None:
        """
        Function for updating the disjoint sets of player nodes. "Islands of nodes belonging to a player"
        Parameters:
            move: (tuple, int) The played move
            player: (int) The player to update sets for
        Returns:
            None
        """

        if isinstance(move, tuple):
            move = self.convert_node(move, isflattened=False)

        # Get neighbours of the move
        neighs = set(self.get_neighbours(move, returnFlattened=True))
        disjoints = self.nodesets[player]

        # Make a greater set of all the sets which neighbours the move, as well as the move. If no neighbours, only make a set of the move
        still_disjoint = []
        inserted = [move]
        for i in disjoints:
            if not neighs.isdisjoint((i)):
                inserted += i
            else:
                still_disjoint.append(i)

        # The new disjoint sets are all the former which did not neighbour the move, as well as one combined set of the move and all neighbouring sets
        still_disjoint.append(set(inserted))
        self.nodesets[player] = still_disjoint

    def convert_node(self, node, isflattened: bool):
        """
        Function for converting node between flattened and unflattened
        Parameters:
            node: (tuple, int) Cell to change representation of
            isflattened: (bool) Specify what state cell is in now
        Returns:
            tuple / int
        """

        if isflattened:  # Node is an index in the flattened state
            dim1 = node // self.board_size
            dim2 = node % self.board_size
            return (dim1, dim2)
        else:  # Node is a tuple
            dim1 = self.board_size * node[0]
            dim2 = node[1]
            return dim1 + dim2

    def set_state(self, state: ndarray or int) -> None:
        """
        Function for setting state
        Parameters:
            state: (np.ndarray) State to set
        Returns:
            None
        """

        if state.ndim == 1:
            self.state = self.unflatten_state(state)
        else:
            self.state = state

        self.legal_moves = self.get_legal_moves(state)

    def set_player(self, player: int) -> None:
        """
        Function for setting player
        Parameters:
            player: (int) Player to set
        Returns:
            None
        """
        if player == 1 or player == 2:
            self.player = player

    def unflatten_state(self, state) -> np.ndarray:
        """
        Function for unflattening state
        Parameters:
            state: (np.ndarray / list) State to change
        Returns:
            np.ndarray
        """
        s = []
        r = 0
        for i in range(self.board_size):
            s.append([state[n] for n in range(r, self.board_size * (i + 1))])
            r += self.board_size
        return np.array(s)

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
        return np.array([i for i in range(self.board_size**2)])
