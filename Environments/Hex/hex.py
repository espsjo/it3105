import numpy as np


class Hex:
    """  
    Initialise variables
    Returns void
    """

    def __init__(self, board_size):
        self.board_size = board_size
        self.reset_states(player_start=0)

    """  
    Reset state, and variables containing state information
    Returns void
    """

    def reset_states(self, player_start):

        # Represented with 1 and 2
        self.player = player_start

        # 0: Open cell, 1: Player 1, 2: Player 2
        self.state = np.zeros((self.board_size, self.board_size))

        # Legal moves are represented as indexes in the flattened state
        self.legal_moves = np.ones(self.board_size**2)

        # Variable for which player eventually wins. If != 0, the game should stop
        self.winner = 0

    """ 
    Function for validating if a player has won the current game or not 
    Returns boolean 
    """

    def is_winning(self):
        pass

    """ 
    Function for validating a move
    Returns boolean
    """

    def check_valid_move(self, move):
        pass

    """  
    Function for updating the state with a move, given that the move is legal. 
    !Does not update state! -> Have to update player, state, valid moves, flattened state
    Returns array
    """

    def play_move(self, move):
        new_state = self.state

        # Update state if valid
        if self.check_valid_move(move):
            new_state[move[0]][move[1]] = self.player

        return new_state

    """  
    Function for returning legal moves, specifically for redistributing the probabilities in ANET to not include illegal moves
    Returns array
    """

    def get_legal_moves(self):
        return self.legal_moves

    """  
    Function for flattening the state, in variation with or without the current player turn
    Returns array
    """

    def flatten_state(self, include_turn: bool):
        flat_state = self.state.flatten()
        if include_turn:
            flat_state = np.insert(flat_state, 0, self.player, axis=0)
        return flat_state

    """  
    Function for simulating a reward
    Returns int
    """

    def get_reward(self):
        # if "logic for checking winner":
        #     return 1
        # else:
        #     return 0
        pass

    """ 
    Function for getting the current state 
    Returns array
    """

    def get_state(self):
        return self.state

    """
    Function for getting the neighbours of a given "node"
    Returns array
    """

    def get_neighbours(self, node):
        r = node[0]
        c = node[1]
        neighbours = []

        pass
    """  
    Function for converting node between flattened and unflattened
    Returns tuple or int
    """

    def convert_node(self, node, flattened: bool):
        if flattened:  # Node is an index in the flattened state
            dim1 = node//self.board_size
            dim2 = node % self.board_size
            return (dim1, dim2)
        else:  # Node is a tuple
            dim1 = self.board_size*node[0]
            dim2 = node[1]
            return (dim1 + dim2)


if __name__ == "main":
    hex_game = Hex()
