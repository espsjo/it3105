from Environments.simworld import SimWorld
from .mctsnode import MCTSNode
import time

"""
Class for carrying out the Monte Carlo Tree Search
"""


class MCTS:
    """
    Function for initializing MCTS
    Returns void
    """

    def __init__(self, MCTS_config, simworld: SimWorld):
        self.env = (
            simworld.get_world()
        )  # Hex or NIM, with the functions in simworldabs available
        self.EPSILON = MCTS_config["EPSILON"]
        self.UCT_C = MCTS_config["UCT_C"]
        self.MAX_TIME = MCTS_config["MAX_TIME"]
        self.MAX_SIMS = MCTS_config["MAX_SIMS"]

        self.root_state = self.env.get_state(flatten=True, include_turn=True)
        self.root = MCTSNode(self.root_state)
        self.player = self.env.get_current_player()

    def search(self):
        t = self.MAX_TIME
        while time.time() - t < self.MAX_TIME:
            self.simulate()
        return self.sel_move()

    def simulate(self):
        pass
