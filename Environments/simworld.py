from .Worlds.hex import Hex
from .Worlds.nim import NIM
from .Worlds.simworldabs import SimWorldAbs


class SimWorld:
    """
    Class for initialising environments into a common object
    Parameters:
        config: (dict) General config
        game_configs: (dict) All game config files
        simulator: (bool) Specify is this simworld is a simulator or not
    """

    def __init__(self, config, game_configs, simulator: bool):

        GAME = config["GAME"]
        GAME_CONFIG = game_configs[GAME]
        if not simulator == False:
            GAME_CONFIG["WON_MSG"] = not simulator

        if GAME == "hex":
            self.World = Hex(GAME_CONFIG)

        elif GAME == "nim":
            self.World = NIM(GAME_CONFIG)

    def get_world(self) -> SimWorldAbs:
        """
        Returns the environment object
        Parameters:
            None
        Returns:
            SimWorldAbs
        """
        return self.World
