from matplotlib import pyplot as plt
from numpy import cos, sin, pi
from .hex import Hex


class HexGUI:
    def __init__(self, Hex: Hex, ANIMATION_SPEED: float):
        self.Hex = Hex
        self.hist = self.Hex.board_hist
        self.winner = self.Hex.winner
        self.winning_island = self.Hex.winning_island
        self.ANIMATION_SPEED = ANIMATION_SPEED
        self.visualize()

    def visualize(self):
        plt.axes()
        ax = plt.gca()
        t1 = cos(pi/4)
        t2 = sin(pi/4)
        d = {0: 'white', 1: 'red', 2: 'blue'}
        for board in self.hist:
            for i, j in enumerate(board.flatten()):

                cell = self.Hex.convert_node(i, isflattened=True)

                circle = plt.Circle(
                    (cell[0]*t1-cell[1]*t2, cell[0]*t2 + cell[1]*t1),
                    0.3,
                    linewidth=3,
                    ec='black',
                    fc=d[j],

                )
                ax.add_patch(circle)
                ax.axis('off')
                plt.axis('scaled')

            plt.pause(self.ANIMATION_SPEED)
        plt.title(f"The winner was {d[self.winner]} (player {self.winner}) ")

        for i, j in enumerate(self.hist[-1].flatten()):

            cell = self.Hex.convert_node(i, isflattened=True)

            if not i in self.winning_island:

                circle = plt.Circle(
                    (cell[0]*t1-cell[1]*t2, cell[0]*t2 + cell[1]*t1),
                    0.3,
                    linewidth=3,
                    ec='black',
                    fc=d[j],

                )
            else:
                circle = plt.Circle(
                    (cell[0]*t1-cell[1]*t2, cell[0]*t2 + cell[1]*t1),
                    0.3,
                    linewidth=3,
                    ec='darkorange',
                    fc=d[j],

                )
            ax.add_patch(circle)
            ax.axis('off')
            plt.axis('scaled')

        plt.pause(0.001)
        plt.show()
