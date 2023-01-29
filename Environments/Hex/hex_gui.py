from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np


class HexGUI:
    def __init__(self, Hex, ANIMATION_SPEED: float):
        self.Hex = Hex
        self.ANIMATION_SPEED = ANIMATION_SPEED
        self.setup()

    def setup(self):
        _, self.ax = plt.subplots(1)
        self.ax.set_aspect('equal')
        self.colors = {0: 'white', 1: 'red', 2: 'blue'}
        self.board = self.Hex.state
        self.visualize_move(self.board, False, False)

    def visualize_move(self, board, won, move):
        if won:
            winning_island = self.Hex.winning_island
            plt.title(
                f"The winner was {self.colors[won]} (player {won}) ")

            l = []
            for i, j in enumerate(board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == move:
                    colors = {0: 'white', 1: 'darkred', 2: 'darkblue'}

                # Horizontal cartesian coords
                x = c[0]*np.sqrt(3)+0.87*c[1]

                # Vertical cartersian coords
                y = -c[1]*3/2

                if not i in winning_island:
                    hex = RegularPolygon((x, y), numVertices=6, radius=1,
                                         orientation=np.radians(0),
                                         facecolor=colors[j], alpha=1, edgecolor='k')
                else:
                    l.append(RegularPolygon((x, y), numVertices=6, radius=1, orientation=np.radians(
                        0), facecolor=colors[j], alpha=1, edgecolor='darkorange', linewidth=4))
                self.ax.add_patch(hex)
            for i in l:
                self.ax.add_patch(i)
            self.ax.axis('off')
            plt.axis('scaled')
            plt.pause(0.001)
        else:
            for i, j in enumerate(self.board.flatten()):
                colors = self.colors
                c = self.Hex.convert_node(i, isflattened=True)
                if c == move:
                    colors = {0: 'white', 1: 'darkred', 2: 'darkblue'}
                # Horizontal cartesian coords
                x = c[0]*np.sqrt(3)+0.87*c[1]

                # Vertical cartersian coords
                y = -c[1]*3/2

                hex = RegularPolygon((x, y), numVertices=6, radius=1,
                                     orientation=np.radians(0),
                                     facecolor=colors[j], alpha=1, edgecolor='k')

                self.ax.add_patch(hex)
            self.ax.axis('off')
            plt.axis('scaled')
            plt.pause(self.ANIMATION_SPEED)
        if won:
            plt.show()
