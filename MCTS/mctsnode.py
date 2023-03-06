from __future__ import annotations
from typing import Dict


class MCTSNode:
    """
    Class for representing a node in the Monte Carlo Tree.
    Contains information about how many times the node has been visited
    Also contains information about how many times each action has been taken, running reward
    Does not contain information about parents/children, that is handled in the main mcts class
    Parameters:
        state: (np.ndarry) The state this node represents
        player: (int) The player which turn it is at this node
    """

    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.N_s = 0
        self.E = 0
        self.N_sa: Dict[int, int] = {}  # action (int): num (int)
        self.children: Dict[int, MCTSNode] = {}  # action (int): child (MCTSNode)
        self.parent: MCTSNode = None

    def add_child(self, move, child: MCTSNode) -> None:
        """
        Function for adding child to this node
        Parameters:
            move: (int) The move which leads to child
            child: (MCTSNode) The child node
        Returns:
            None
        """
        self.children[move] = child
        child.parent = self

    def is_leaf(self) -> bool:
        """
        Returning if the node is a leaf or not
        Parameters:
            None
        Returns:
            bool
        """
        return not self.children

    def update_E(self, e) -> None:
        """
        Updates the E value
        Parameters:
            e: (float) E value to be added
        Returns:
            None
        """
        self.E += e

    def increment_N_s(self) -> None:
        """
        Increments the N_s value by 1
        Parameters:
            None
        Returns:
            None
        """
        self.N_s += 1

    def increment_N_sa(self, move) -> None:
        """
        Increments the N_sa value by 1
        Parameters:
            move: (int) Which move is to be updated
        Returns:
            None
        """
        self.N_sa[move] = self.N_sa.get(move, 0) + 1
