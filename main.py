from Environments.Worlds.hex import Hex
from Environments.Worlds.nim import NIM
from Environments.simworld import SimWorld
from config import config, game_configs
import numpy as np


def main():
    GAME = config["GAME"]
    UI_ON = config["UI_ON"]
    GAME_CONFIG = game_configs[GAME]

    World = SimWorld(GAME, GAME_CONFIG, visualize=UI_ON).get_world()
    for i in range(2):
        while not World.is_won():
            move = int(np.random.choice(World.get_legal_moves()))
            World.play_move(move)
        World.reset_states(visualize=UI_ON, player_start=1)


if __name__ == "__main__":
    main()
