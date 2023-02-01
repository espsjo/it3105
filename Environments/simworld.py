from .Worlds.hex import Hex
from .Worlds.nim import NIM
from .Worlds.simworldabs import SimWorldAbs

"""
Class for initialising environments into a common object
"""


class SimWorld:
    def __init__(self, GAME, GAME_CONFIG, visualize: bool):

        if GAME == "hex":
            self.World = Hex(GAME_CONFIG, visualize)

        elif GAME == "nim":
            self.World = NIM(GAME_CONFIG, visualize)

    def get_world(self):
        return self.World
