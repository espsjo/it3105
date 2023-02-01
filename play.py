from Environments.simworld import SimWorld
from config import config, game_configs
import numpy as np

"""
Ability to play against computer/human - only by registering moves in console (index form)
Returns void 
"""


def play(n_games, human_ai: bool):
    World = SimWorld(config, game_configs).get_world()
    for i in range(n_games):
        World.reset_states(player_start=1)
        while not World.is_won():
            p = World.get_current_player()
            if World.get_current_player() == 2:
                move = int(input(f"\nPlayer {p} move: "))
            else:
                if human_ai:
                    move = int(np.random.choice(World.get_legal_moves()))
                else:
                    move = int(input(f"\nPlayer {p} move: "))
            World.play_move(move)


if __name__ == "__main__":
    play(1, human_ai=False)
