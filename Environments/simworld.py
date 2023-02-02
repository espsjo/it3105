from .Worlds.hex import Hex
from .Worlds.nim import NIM
from .Worlds.simworldabs import SimWorldAbs

"""
Class for initialising environments into a common object
"""


class SimWorld:
    def __init__(self, config, game_configs, deep: bool = False):

        VISUALIZE = config["UI_ON"]
        GAME = config["GAME"]
        GAME_CONFIG = game_configs[GAME]

        if GAME == "hex":
            self.World = Hex(GAME_CONFIG, visualize=(VISUALIZE and not deep))

        elif GAME == "nim":
            self.World = NIM(GAME_CONFIG, visualize=(VISUALIZE and not deep))

    def get_world(self) -> SimWorldAbs:
        return self.World
