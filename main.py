from Environments.Hex.hex import Hex
from Environments.NIM.nim import NIM
from config import hex_config, nim_config, config
import numpy as np


def main():
    GAME = config["GAME"]
    UI_ON = config["UI_ON"]
    if GAME == "hex":
        h = Hex(hex_config, visualize=UI_ON)
        while not h.is_won():
            move = int(np.random.choice(h.get_legal_moves()))
            h.play_move(move)
    if GAME == "nim":
        n = NIM(nim_config, verbose=UI_ON)
        while not n.is_won():
            move = int(np.random.choice(n.get_legal_moves()))
            n.play_move(move)


if __name__ == "__main__":
    main()
