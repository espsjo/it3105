from ActorClient import ActorClient
from ANET.anet import ANET
from Config.config import config, game_configs, ANET_config, OHT_config
from Environments.simworld import SimWorld
import numpy as np
from Environments.visualizer import Visualizer


class OHT(ActorClient):
    """
    Initializes the class to prepare for OHT
    Overrides key parameters in config-files, so that it matches those needed for OHT
    Parameters:
        auth: (str) The secret certificate used to join
        qualify: (bool) If the attempt should count towards qualification
    """

    def __init__(self, auth, qualify):
        super().__init__(auth=auth, qualify=qualify)

        config["GAME"] = OHT_config["GAME"]
        game_configs[config["GAME"]]["BOARD_SIZE"] = OHT_config["BOARD_SIZE"]
        ANET_config["LOAD_PATH"] = OHT_config["LOAD_PATH"]
        ANET_config["EPSILON"] = OHT_config["EPSILON"]

        self.BOARD_SIZE = OHT_config["BOARD_SIZE"]
        self.env = SimWorld(config, game_configs, simulator=False).get_world()
        self.actor = ANET(
            ANET_config=ANET_config,
            Environment=self.env,
            model_name=OHT_config["MODEL_NAME"],
        )

        UI_ON = OHT_config["UI_ON"]
        self.GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None

    def handle_game_start(self, start_player):
        """
        Resets the environement, specifying which player starts the game.
        Parameters:
            start_player: (int) 1/2 depending on which player should start the game
        Returns:
            None
        """
        self.env.reset_states(player_start=start_player)
        if self.GUI:
            self.GUI.visualize_move(self.env)

    def handle_get_action(self, state):
        """
        Takes in a game state, and lets the actor choose a move
        Parameters:
            state: (list) The game state
        Returns:
            tuple
        """
        self.env.set_player(state[0])
        state = np.array(state[1:])
        state = (
            np.transpose(state.reshape((self.BOARD_SIZE, self.BOARD_SIZE)))
        ).flatten()
        self.env.set_state(state)

        if self.GUI:
            self.GUI.visualize_move(self.env)

        move = self.actor.get_action(
            self.env.get_state(flatten=True, include_turn=True),
            self.env.get_legal_moves(),
            choose_greedy=True,
        )

        foo = self.env.play_move(move)
        if self.GUI:
            self.GUI.visualize_move(self.env)
        if not foo:
            print("ERROR")

        move = np.flip(self.convert_node(move))
        return int(move[0]), int(move[1])

    ## HELPER FUNCTION ##
    def convert_node(self, node) -> tuple:
        """
        Function for converting node between flattened and unflattened
        Parameters:
            node: (int) Cell to change representation of
            isflattened: (bool) Specify what state cell is in now
        Returns:
            tuple
        """
        dim1 = node // self.BOARD_SIZE
        dim2 = node % self.BOARD_SIZE

        return int(dim1), int(dim2)


"""
Main init
"""
if __name__ == "__main__":
    oht = OHT(auth=OHT_config["AUTH"], qualify=OHT_config["QUALIFY"])
    oht.run(mode=OHT_config["MODE"])
