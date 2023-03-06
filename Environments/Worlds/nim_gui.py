from .visualizerabs import VisualizerAbs
from .simworldabs import SimWorldAbs
import os
import time


class NIMGUI(VisualizerAbs):
    """
    Class for visualizing nim game
    Parameters:
        nim_config: (dict) Config for nim game
    """

    def __init__(self, nim_config):
        self.nim_config = nim_config
        self.DELAY = self.nim_config["DELAY"]
        self.stones = nim_config["STONES"]
        self.min_stones = nim_config["MIN_STONES"]
        self.max_stones = nim_config["MAX_STONES"]
        self.state = self.stones

    def reset(self) -> None:
        """
        Resets the console
        Parameters:
            None
        Returns:
            None
        """
        clear = lambda: os.system("cls")
        clear()

    def visualize_move(self, env: SimWorldAbs, move=None) -> None:
        """
        Visualizes a move by printing info to console
        Parameters:
            env: (SimWorldAbs) The environment
            move: (int) The move that was played
        Returns:
            None
        """

        self.env = env
        if move == None:
            print(
                f"""
## NEW GAME OF NIM (Stones = {self.stones}, min: {self.min_stones}, max: {self.max_stones}) ##"""
            )
        else:
            self.player = self.env.get_current_player()
            former = self.state
            self.state = self.env.get_state(include_turn=False)[0]
            print(
                f"""
There are {former} remaining stone(s).
Player {self.player} has chosen to take {move} stone(s)
Now, there are {self.state} stone(s) left."""
            )

        if self.env.is_won():
            print(f"\nThe winner is player {self.env.get_winner()}")
            if self.DELAY > 0.05:
                time.sleep(self.DELAY * 10)
                self.reset()

        if self.DELAY > 0.05:
            time.sleep(self.DELAY)
