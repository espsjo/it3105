from Environments.simworld import SimWorld
from MCTS.mcts import MCTS
from config import config, game_configs, MCTS_config
import time


def main(n_games):
    World = SimWorld(config, game_configs).get_world()
    for x in range(n_games):
        mcts_env = SimWorld(config, game_configs, deep=True).get_world()
        m = MCTS(MCTS_config, mcts_env, player=1)
        mcts_env.reset_states(player_start=1)
        World.reset_states(player_start=1)
        while not World.is_won():
            i = 0
            t = time.time()
            while (i < 1000) and (time.time() - t < 3):
                m.itr()
                i += 1
            norm, moves = m.norm_distr()
            move = moves[norm.index(max(norm))]
            World.play_move(move)
            m.mcts_move(move, change_player=True)
            print(norm)
            print(moves)
            print(move)


if __name__ == "__main__":
    main(1)
