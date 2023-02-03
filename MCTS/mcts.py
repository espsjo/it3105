from Environments.Worlds.simworldabs import SimWorldAbs
from .mctsnode import MCTSNode
import numpy as np
import time
from typing import Dict
from copy import deepcopy

"""
Class for carrying out iterations in the Monte Carlo Tree Search 
"""


class MCTS:
    """
    Function for initializing MCTS
    Returns void
    """

    def __init__(self, MCTS_config, simworld: SimWorldAbs, player: int = 1):
        # The environment
        self.env = simworld

        # The player perspective (remains constant)
        self.player = player

        # Constants
        self.EPSILON = MCTS_config["EPSILON"]
        self.UCT_C = MCTS_config["UCT_C"]

        # Initialise root and player info
        self.turn = self.env.get_current_player()
        self.root_state = self.env.get_state(flatten=True, include_turn=False)
        self.root = MCTSNode(self.root_state, self.turn)

    """
    Updates the environment with the chosen move, inits all the variables to correct
    Returns void
    """

    def mcts_move(self, move: int):
        self.env.play_move(move)
        self.root_state = self.env.get_state(flatten=True, include_turn=False)
        self.turn = self.env.get_current_player()
        # Dont keep children - start over
        self.root = MCTSNode(self.root_state, self.turn)

    """
    Runs one iteration of MCTS
    Returns void
    """

    def itr(self):
        current, world = self.search()
        current, world = self.expand(current, world)
        world = self.rollout(world)
        self.backprop(current, world)

    """
    Runs the tree-search
    Returns the current node and world state
    """

    def search(self):
        # Search
        world = deepcopy(self.env)
        current = self.root
        while not current.is_leaf():
            max_bool = self.player == world.get_current_player()
            action = self.argf(self.compute_uct(current, max_bool), max_bool)
            world.play_move(action)
            current.increment_N_sa(action)
            current = current.children[action]
        return current, world

    """
    Expands the node with its children
    Returns one of the children and the given world state
    """

    def expand(self, current, world):
        # Expand
        if not world.is_won():
            legal = world.get_legal_moves(world.get_state())
            for act in legal:
                temp_world = deepcopy(world)
                temp_world.play_move(act)
                current.add_child(
                    act,
                    MCTSNode(
                        temp_world.get_state(flatten=True),
                        temp_world.get_current_player(),
                    ),
                )
        # Choose one of the children
        if current.children:
            new_act = np.random.choice(list(current.children.keys()))
            current.increment_N_sa(new_act)
            current = current.children[new_act]
            world.play_move(new_act)
        return current, world

    """
    Rollout on the child given
    Returns the world state
    """

    def rollout(self, world):
        # Rollout
        while not world.is_won():
            legal = world.get_legal_moves(world.get_state())
            action = int(np.random.choice(legal))  ## INSERT SOME POLICY
            world.play_move(action)
        return world

    """
    Backprops the reward to all the nodes
    Returns void
    """

    def backprop(self, current, world):
        while current is not None:
            current.update_E(world.get_reward(self.player))
            current.increment_N_s()
            current = current.parent

    """
    Helper function for computing UCT
    Returns dict (move: uct)
    """

    def compute_uct(self, parent: MCTSNode, max_bool: bool):
        act_uct: Dict[int, float] = {}
        sign = 1
        if not max_bool:
            sign = -1
        N_s = parent.N_s
        for act, node in parent.children.items():
            N_sa = parent.N_sa.get(act, 0)
            E = node.E
            Q_sa = 0
            if N_sa != 0:
                Q_sa = E / N_sa
            u_sa = self.UCT_C * np.sqrt((np.log(N_s)) / (1 + N_sa))
            act_uct[act] = Q_sa + u_sa * sign
        return act_uct

    """
    Returns the normal distribution, and also the moves the indexes map to
    Returns list,list
    """

    def norm_distr(self):
        d = []
        total_vis = self.root.N_s
        for a in self.env.get_actions():
            if a in self.root.children:
                child_vis = self.root.children[a].N_s
                d.append(float(child_vis / total_vis))
            else:
                d.append(0)

        return d, self.env.get_actions()

    """
    Helper function for finding argmax/argmin of a dictionary
    Returns float
    """

    def argf(self, d, max_bool: bool):
        f = max if max_bool else min
        return f(d, key=d.get)
