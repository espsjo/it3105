from matplotlib import pyplot as plt
from numpy import cos, sin, pi


class HexGUI:
    def __init__(self, Hex):
        self.Hex = Hex

    def visualize(self, history, winner):
        self.hist = history
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

            plt.pause(0.25)
        plt.title(f"The winner was {d[winner]} (player {winner}) ")
        plt.show()
