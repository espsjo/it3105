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

    def __init__(self, state):
        self.state = state
        self.N_s = 1
        self.N_sa = {}
        self.E_t = 0
        self.children = []

    """
    Add a child node to this node
    Returns void
    """

    def add_child(self, child):
        self.children.append(child)

    """
    Get the child nodes of this node
    Returns list
    """

    def get_children(self):
        return self.children

    """
    Gets the state which the node represents
    Returns array
    """

    def get_state(self):
        return self.state

    """
    Gets the number of times visited
    Returns int
    """

    def get_N_s(self):
        return self.N_s

    """
    Gets the number of times an action has been taken from the given node
    Returns int
    """

    def get_N_sa(self, move):
        return self.N_sa.get(move, 0)

    """
    Gets the running total eval
    Returns int
    """

    def get_E_t(self):
        return self.E_t

    """
    Increments the running eval
    Returns void
    """

    def update_E_t(self, e_t):
        self.E_t += e_t

    """
    Updates the times visited
    Returns void
    """

    def update_N_s(self):
        self.N_s += 1

    """
    Updates the times visited
    Returns void
    """

    def update_N_sa(self, move):
        self.N_sa[move] = self.get_N_sa(move) + 1
