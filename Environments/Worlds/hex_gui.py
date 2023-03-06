from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
from .visualizerabs import VisualizerAbs
from .simworldabs import SimWorldAbs


class HexGUI(VisualizerAbs):
    """
    Initialises the class
    Parameters:
        hex_config: (dict) Config settings for hex
    """

    def __init__(self, hex_config):
        self.hex_config = hex_config
        self.ANIMATION_SPEED = self.hex_config["ANIMATION_SPEED"]
        self.DISPLAY_INDEX = self.hex_config["DISPLAY_INDEX"]
        self.setup()

    def setup(self) -> None:
        """
        Sets up the GUI and som key variables
        Parameters:
            None
        Returns:
            None
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(1)
        self.ax.set_aspect("equal")
        self.colors = {0: "white", 1: "red", 2: "blue"}

    def reset(self) -> None:
        """
        Resets the figure for a new game
        Parameters:
            None
        Returns:
            None
        """
        plt.cla()
        plt.pause(0.001)

    def visualize_move(self, Hex: SimWorldAbs, move=None) -> None:
        """
        Function for visualsing the board after each move. Call to plt.show() depends on a player actually winning the game,
        but this has been mathematically proven to always be the case.
        Also highlights the "island" that proved winning for the player.
        A little "hacky" code, especially for the part where a player wins, but does the job for basic visualizing.
        Parameters:
            Hex: (SimWorldAbs) The environment to help visualize the board
            move: (int) The last played move
        Returns:
            None
        """

        self.Hex = Hex
        self.move = move
        if move != None:
            self.move = self.Hex.convert_node(move, isflattened=True)
        self.board = self.Hex.get_state()
        self.won = self.Hex.is_won()
        self.winner = self.Hex.get_winner()
        if self.won:
            winning_island = self.Hex.winning_island
            plt.title(
                f"The winner was {self.colors[self.winner]} (player {self.winner}) "
            )

            l = []
            for i, j in enumerate(self.board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == self.move:
                    colors = {0: "white", 1: "darkred", 2: "darkblue"}

                # Horizontal cartesian coords
                x = c[0] * np.sqrt(3) + 0.87 * c[1]

                # Vertical cartersian coords
                y = -c[1] * 3 / 2

                if not i in winning_island:
                    hex = RegularPolygon(
                        (x, y),
                        numVertices=6,
                        radius=1,
                        orientation=np.radians(0),
                        facecolor=colors[j],
                        alpha=1,
                        edgecolor="k",
                    )
                    self.ax.add_patch(hex)
                else:
                    l.append(
                        RegularPolygon(
                            (x, y),
                            numVertices=6,
                            radius=1,
                            orientation=np.radians(0),
                            facecolor=colors[j],
                            alpha=1,
                            edgecolor="darkorange",
                            linewidth=4,
                        )
                    )
            for i in l:
                self.ax.add_patch(i)
            self.ax.axis("off")
            plt.axis("scaled")
            plt.pause(2)
            self.reset()
        else:
            for i, j in enumerate(self.board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == self.move:
                    colors = {0: "white", 1: "darkred", 2: "darkblue"}
                # Horizontal cartesian coords
                x = c[0] * np.sqrt(3) + 0.87 * c[1]

                # Vertical cartersian coords
                y = -c[1] * 3 / 2

                hex = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=1,
                    orientation=np.radians(0),
                    facecolor=colors[j],
                    alpha=1,
                    edgecolor="k",
                )
                if self.DISPLAY_INDEX:
                    plt.text(x - 0.2, y - 0.1, str(i))

                self.ax.add_patch(hex)
            self.ax.axis("off")
            plt.axis("scaled")
            plt.pause(self.ANIMATION_SPEED)
