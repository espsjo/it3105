import numpy as np
from copy import deepcopy


class Hex:
    """  
    Initialise variables
    Returns void
    """

    def __init__(self, board_size):
        self.board_size = board_size
        self.reset_states(player_start=1)

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
        self.legal_moves = np.array([i for i in range(self.board_size**2)])

        # Variable for which player eventually wins. If != 0, the game should stop
        self.winner = 0

        # Dictionary of disjoint sets belonging to each player
        self.nodesets = {1: [], 2: []}

        # History of board
        self.board_hist = []

        # Winning condition used in is_winning
        self.win_cond = set([i for i in range(self.board_size)])

    """ 
    Function for validating if a player has won the current game or not 
    Returns bool
    """

    def is_winning(self, player):

        disjoints = self.nodesets[player]
        # Player 1 wants to get a connected island where the y-coord (row) counts from 0 to board_size-1
        coord = 1
        # Player 2 wants to get a connected island where the x-coord (col) counts from 0 to board_size-1
        if player == 2:
            coord = 0

        for disj in disjoints:
            if len(disj) < self.board_size:
                continue
            x_y = set([self.convert_node(flat_n, isflattened=True)[coord]
                      for flat_n in disj])
            if x_y == self.win_cond:
                return True

        return False

    """ 
    Function for validating a move
    Returns boolean
    """

    def check_valid_move(self, move):

        if isinstance(move, int):
            return (move in self.legal_moves)
        else:
            return (self.convert_node(move, isflattened=False) in self.legal_moves)

    """  
    Function for updating the state with a move, given that the move is legal. 
    Returns bool
    """

    def play_move(self, move):
        if self.winner != 0:
            return False
        if not self.check_valid_move(move):
            return False

        self.board_hist.append(deepcopy(self.state))

        new_state = self.state
        flattened = isinstance(move, int)

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
        self.legal_moves = np.setdiff1d(
            self.legal_moves, np.array([flat_move]))

        # Update disjoint sets
        self.update_disjoint_sets(flat_move, self.player)

        # Check for win
        if self.is_winning(self.player):
            self.winner = self.player
            # Add the final state
            self.board_hist.append(deepcopy(self.state))
            #print("WINNER WINNER CHICKEN DINNER")

        # Update player turn
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1

        return True

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
    Returns array
    """

    def get_state(self):
        return self.state

    """
    Function for getting the neighbours of a given "node"
    Returns list
    """

    def get_neighbours(self, node, returnFlattened: bool):
        if isinstance(node, int):
            node = self.convert_node(node, isflattened=True)

        r = node[0]
        c = node[1]
        pot_neighbours = [(r-1, c), (r-1, c+1), (r, c-1),
                          (r, c+1), (r+1, c-1), (r+1, c)]
        neighbours = []
        for n in pot_neighbours:
            x = n[0]
            y = n[1]

            if 0 <= x <= self.board_size-1:
                if 0 <= y <= self.board_size-1:
                    if returnFlattened:
                        neighbours.append(
                            self.convert_node(n, isflattened=False))
                    else:
                        neighbours.append(n)
        return neighbours

    """ 
    Function for updating the disjoint sets of player nodes. "Islands of nodes belonging to a player"
    Returns void
    """

    def update_disjoint_sets(self, move, player):
        if not isinstance(move, int):
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

    """  
    Function for converting node between flattened and unflattened
    Returns tuple or int
    """

    def convert_node(self, node, isflattened: bool):
        if isflattened:  # Node is an index in the flattened state
            dim1 = node//self.board_size
            dim2 = node % self.board_size
            return (dim1, dim2)
        else:  # Node is a tuple
            dim1 = self.board_size*node[0]
            dim2 = node[1]
            return (dim1 + dim2)


""" 
Function for choosing random moves to simulate
Returns bool
"""


def random_sim(max_moves, max_itr):
    n = 0
    while True:
        h = Hex(board_size=5)
        moves = 0
        while not h.winner:
            moves += 1
            move = int(np.random.choice(h.get_legal_moves()))
            h.play_move(move)
        if moves < max_moves or n > max_itr:
            print(
                f"The winner was {h.winner} with the move {h.convert_node(move,isflattened=True)}")
            print(h.state)
            break
        n += 1


if __name__ == "__main__":
    random_sim(max_moves=15, max_itr=100)
