from __future__ import annotations
from typing import Dict

"""
Class for representing a node in the Monte Carlo Tree.
Contains information about how many times the node has been visited
Also contains information about how many times each action has been taken, running reward
Does not contain information about parents/children, that is handled in the main mcts class
"""


class MCTSNode:
    """
    Inits the variables to be tracked
    Returns void
    """

    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.N_s = 0
        self.E = 0
        self.N_sa: Dict[int, int] = {}  # action (int): num (int)
        self.children: Dict[int, MCTSNode] = {}  # action (int): child (MCTSNode)
        self.parent: MCTSNode = None

    """
    Function for adding child to this node
    Returns void
    """

    def add_child(self, move, child: MCTSNode):
        self.children[move] = child
        child.parent = self

    """
    Returning if the node is a leaf or not
    Returns bool
    """

    def is_leaf(self):
        return not self.children

    """
    Returns the E value
    Returns int
    """

    def update_E(self, e):
        self.E += e

    """
    Returns the N_s value
    Returns int
    """

    def increment_N_s(self):
        self.N_s += 1

    """
    Returns the N_sa value
    Returns int
    """

    def increment_N_sa(self, move):
        self.N_sa[move] = self.N_sa.get(move, 0) + 1
