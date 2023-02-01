from Environments.Worlds.hex import Hex
from Environments.Worlds.nim import NIM
from Environments.simworld import SimWorld
from config import config, game_configs
import numpy as np

"""
Ability to play against computer - only by registering moves in console
Returns void 
"""


def play(n_games):
    GAME = config["GAME"]
    UI_ON = config["UI_ON"]
    GAME_CONFIG = game_configs[GAME]

    World = SimWorld(GAME, GAME_CONFIG, visualize=UI_ON).get_world()
    for i in range(n_games):
        while not World.is_won():
            if World.get_current_player() == 2:
                move = int(input(f"\nMove: "))
            else:
                move = int(np.random.choice(World.get_legal_moves()))
            World.play_move(move)
        World.reset_states(visualize=UI_ON, player_start=1)


if __name__ == "__main__":
    play(1)
