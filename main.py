from Environments.simworld import SimWorld
from config import config, game_configs
import numpy as np


def main(n_games):
    World = SimWorld(config, game_configs).get_world()
    for i in range(n_games):
        World.reset_states(player_start=1)
        while not World.is_won():
            move = int(np.random.choice(World.get_legal_moves()))
            World.play_move(move)


if __name__ == "__main__":
    main(1)
