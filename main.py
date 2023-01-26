from Environments.Hex.hex import Hex
from Environments.Hex.hex_gui import HexGUI
from constants import Constants
import numpy as np


def main():
    h = Hex(Constants.BOARD_SIZE)
    hv = HexGUI(h)
    while not h.winner:
        move = int(np.random.choice(h.get_legal_moves()))
        h.play_move(move)
    hv.visualize(h.board_hist, h.winner)


if __name__ == '__main__':
    main()
