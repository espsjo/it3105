from .Worlds.nim_gui import NIMGUI
from .Worlds.hex_gui import HexGUI
from .Worlds.visualizerabs import VisualizerAbs


class Visualizer:
    """
    Returns a common visualizer object based on abstract
    Parameters:
        config: (dict) General config
        game_configs: (dict) All game config files
    """

    def __init__(self, config, game_configs):
        GAME = config["GAME"]
        game_config = game_configs[GAME]

        if GAME == "hex":
            self.GUI = HexGUI(game_config)

        if GAME == "nim":
            self.GUI = NIMGUI(game_config)

    def get_GUI(self) -> VisualizerAbs:
        """
        Returns the visualizer object
        Parameters:
            None
        Returns:
            VisualizerAbs
        """
        return self.GUI
