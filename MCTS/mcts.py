from Environments.Worlds.simworldabs import SimWorldAbs
from .mctsnode import MCTSNode
import numpy as np
from typing import Dict
from copy import deepcopy
from ANET.anet import ANET
from ANET.litemodel import LiteModel


class MCTS:
    """
    Class for carrying out iterations in the Monte Carlo Tree Search
    Parameters:
        MCTS_config: (dict) MCTS config settings
        simworld: (SimWorldAbs) The environment
        player: (int) The player perspective (optional)
    """

    def __init__(self, MCTS_config, simworld: SimWorldAbs, player: int = 1):
        # The environment
        self.env = simworld

        # The player perspective (remains constant)
        self.player = player

        # Constants
        self.UCT_C = MCTS_config["UCT_C"]
        self.KEEP_SUBTREE = MCTS_config["KEEP_SUBTREE"]

        # Initialise root and player info
        self.turn = self.env.get_current_player()
        self.root_state = self.env.get_state(flatten=True, include_turn=False)
        self.root = MCTSNode(self.root_state, self.turn)

    def mcts_move(self, move: int) -> None:
        """
        Updates the environment with the chosen move, inits all the variables to correct
        Parameters:
            move: (int) The move to play
        Returns:
            None
        """
        self.env.play_move(move)
        if not self.KEEP_SUBTREE or (move not in self.root.children.keys()):
            self.root_state = self.env.get_state(flatten=True, include_turn=False)
            self.turn = self.env.get_current_player()
            self.root = MCTSNode(self.root_state, self.turn)
        else:
            self.root = self.root.children[move]
            self.root.parent = None

    def itr(self, actor=None, litemodel=None):
        """
        Runs one iteration of MCTS
        Parameters:
            actor: (ANET) The actor net which will carry out moves in tree search
            litemodel: (LiteModel) Optional litemodel for faster runtime
        Returns:
            None
        """
        current, world = self.search()
        current, world = self.expand(current, world)
        world = self.rollout(world, actor, litemodel)
        self.backprop(current, world)

    def search(self):
        """
        Runs the tree-search
        Parameters:
            None
        Returns:
            MCTSNode, SimWorldAbs
        """
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

    def expand(self, current, world):
        """
        Expands the node with its children
        Parameters:
            current: (MCTSNode) The current node
            world: (SimWorldAbs) The current environment state
        Returns:
            MCTSNode, SimWorldAbs
        """
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

    def rollout(
        self, world: SimWorldAbs, actor: ANET, litemodel: LiteModel
    ) -> SimWorldAbs:
        """
        Rollout on the child given
        Parameters:
            world: (SimWorldAbs) The current environment state
            actor: (ANET) The actor net which will carry out moves in tree search
            litemodel: (LiteModel) Optional litemodel for faster runtime
        Returns:
            SimWorldAbs
        """
        # Rollout
        while not world.is_won():
            legal_moves = world.get_legal_moves(world.get_state())
            if actor != None:
                state = world.get_state(flatten=True, include_turn=True)
                action = actor.get_action(
                    state, legal_moves, choose_greedy=False, litemodel=litemodel
                )
            else:
                action = int(np.random.choice(legal_moves))

            world.play_move(action)
        return world

    def backprop(self, current, world) -> None:
        """
        Backprops the reward to all the nodes
        Parameters:
            current: (MCTSNode) The current node
            world: (SimWorldAbs) The current environment state
        Returns:
            None
        """
        while current is not None:
            current.update_E(world.get_reward(self.player))
            current.increment_N_s()
            current = current.parent

    def compute_uct(self, parent: MCTSNode, max_bool: bool) -> dict:
        """
        Helper function for computing UCT
        Returns dict (move: uct)
        Parameters:
            parent: (MCTSNode) The parent node
            max_bool: (bool) Specify if to max or min
        Returns:
            dict
        """
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

    def norm_distr(self):
        """
        Returns the normal distribution, and also the moves the indexes map to
        Parameters:
            None
        Returns:
            list, list
        """
        d = []
        total_vis = self.root.N_s
        for a in self.env.get_actions():
            if a in self.root.children:
                child_vis = self.root.children[a].N_s
                d.append(float(child_vis / total_vis))
            else:
                d.append(0)

        return d, self.env.get_actions()

    def argf(self, d, max_bool: bool) -> float:
        """
        Helper function for finding argmax/argmin of a dictionary
        Parameters:
            d: (dict) Dictionary to argmax/argmin
            max_bool: (bool) Specify if to max or min
        Returns:
            float
        """
        f = max if max_bool else min
        return f(d, key=d.get)
