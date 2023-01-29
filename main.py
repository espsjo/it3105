from Environments.Hex.hex import Hex
from constants import hex_config, config
import numpy as np


def main():
    h = Hex(hex_config, visualize=True)
    while not h.winner:
        move = int(np.random.choice(h.get_legal_moves()))
        h.play_move(move)


if __name__ == '__main__':
    main()
