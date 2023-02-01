from .Worlds.hex import Hex
from .Worlds.nim import NIM
from .Worlds.simworldabs import SimWorldAbs

"""
Class for initialising environments into a common object
"""


class SimWorld:
    def __init__(self, config, game_configs):

        VISUALIZE = config["UI_ON"]
        GAME = config["GAME"]
        GAME_CONFIG = game_configs[GAME]

        if GAME == "hex":
            self.World = Hex(GAME_CONFIG, visualize=VISUALIZE)

        elif GAME == "nim":
            self.World = NIM(GAME_CONFIG, visualize=VISUALIZE)

    def get_world(self):
        return self.World
