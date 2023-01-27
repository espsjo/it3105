from Environments.Hex.hex import Hex
from Environments.Hex.hex_gui import HexGUI
from constants import Constants
import numpy as np


def main():
    h = Hex(Constants.BOARD_SIZE)
    while not h.winner:
        move = int(np.random.choice(h.get_legal_moves()))
        h.play_move(move)
    hv = HexGUI(h, ANIMATION_SPEED=Constants.ANIMATION_SPEED)


if __name__ == '__main__':
    main()
