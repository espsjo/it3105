from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np


class HexGUI:
    """
    Initialises the class
    Returns void
    """

    def __init__(self, Hex, ANIMATION_SPEED: float):
        self.Hex = Hex
        self.ANIMATION_SPEED = ANIMATION_SPEED
        self.setup()

    """ 
    Sets up the GUI and som key variables
    Returns void
    """

    def setup(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(1)
        self.ax.set_aspect("equal")
        self.colors = {0: "white", 1: "red", 2: "blue"}

    """
    Resets the figure for a new game
    Returns void
    """

    def reset(self):
        plt.cla()

    """ 
    Function for visualsing the board after each move. Call to plt.show() depends on a player actually winning the game,
    but this has been mathematically proven to always be the case.
    Also highlights the "island" that proved winning for the player.
    A little "hacky" code, especially for the part where a player wins, but does the job for basic visualizing.
    Returns void
    """

    def visualize_move(self, board, won, move):
        self.board = board
        if won:
            winning_island = self.Hex.winning_island
            plt.title(f"The winner was {self.colors[won]} (player {won}) ")

            l = []
            for i, j in enumerate(board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == move:
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
            plt.pause(3)
        else:
            for i, j in enumerate(self.board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == move:
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
                plt.text(x - 0.2, y - 0.1, str(i))

                self.ax.add_patch(hex)
            self.ax.axis("off")
            plt.axis("scaled")
            plt.pause(self.ANIMATION_SPEED)
