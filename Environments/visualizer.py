from .Worlds.nim_gui import NIMGUI
from .Worlds.hex_gui import HexGUI
from .Worlds.visualizerabs import VisualizerAbs


class Visualizer:
    def __init__(self, config, game_configs):
        GAME = config["GAME"]
        game_config = game_configs[GAME]

        if GAME == "hex":
            self.GUI = HexGUI(game_config)

        if GAME == "nim":
            self.GUI = NIMGUI(game_config)

    def get_GUI(self) -> VisualizerAbs:
        return self.GUI
