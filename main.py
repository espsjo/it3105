from Environments.Worlds.hex import Hex
from Environments.Worlds.nim import NIM
from Environments.simworld import SimWorld
from config import config, game_configs
import numpy as np


def main():
    GAME = config["GAME"]
    UI_ON = config["UI_ON"]
    GAME_CONFIG = game_configs[GAME]
    World = SimWorld(GAME, GAME_CONFIG, visualize=False).get_world()

    if GAME == "hex":
        h = Hex(GAME_CONFIG, visualize=UI_ON)
        for i in range(2):
            while not h.is_won():
                move = int(np.random.choice(h.get_legal_moves()))
                h.play_move(move)
            h.reset_states(visualize=UI_ON, player_start=1)
    if GAME == "nim":
        n = NIM(GAME_CONFIG, verbose=UI_ON)
        while not n.is_won():
            if n.get_current_player() == 2:
                move = int(input(f"\nMove: "))
            else:
                move = int(np.random.choice(n.get_legal_moves()))
            n.play_move(move)


if __name__ == "__main__":
    main()
