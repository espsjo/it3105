from Environments.simworld import SimWorld
from MCTS.mcts import MCTS
from config import config, game_configs, MCTS_config
import time
import numpy as np

"""
Ability to play against computer/human - only by registering moves in console (index form)
Returns void 
"""


def play(n_games, human_ai: bool):
    # World = SimWorld(config, game_configs).get_world()
    # for i in range(n_games):
    #     World.reset_states(player_start=1)
    #     while not World.is_won():
    #         p = World.get_current_player()
    #         if World.get_current_player() == 2:
    #             move = int(input(f"\nPlayer {p} move: "))
    #         else:
    #             if human_ai:
    #                 move = int(
    #                     np.random.choice(World.get_legal_moves(World.get_state()))
    #                 )
    #             else:
    #                 move = int(input(f"\nPlayer {p} move: "))
    #         World.play_move(move)

    World = SimWorld(config, game_configs).get_world()
    for x in range(n_games):
        mcts_env = SimWorld(config, game_configs, deep=True).get_world()
        m = MCTS(MCTS_config, mcts_env, player=1)
        mcts_env.reset_states(player_start=1)
        World.reset_states(player_start=1)
        while not World.is_won():
            if World.get_current_player() == 1:
                i = 0
                t = time.time()
                while (i < 1000) and (time.time() - t < 3):
                    m.itr()
                    i += 1
                norm, moves = m.norm_distr()
                move = moves[norm.index(max(norm))]
            else:
                move = int(input("Move: "))
            World.play_move(move)
            m.mcts_move(move, change_player=True)


if __name__ == "__main__":
    play(1, human_ai=False)
