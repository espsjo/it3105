from Environments.simworld import SimWorld
from Environments.visualizer import Visualizer
from MCTS.mcts import MCTS
from config import config, game_configs, MCTS_config
import time
import numpy as np

"""
Ability to play against computer/human - only by registering moves in console (index form)
Returns void 
"""


def play(n_games):
    World = SimWorld(config, game_configs, simulator=False).get_world()
    UI_ON = config["UI_ON"]
    GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None

    for x in range(n_games):
        World.reset_states(player_start=1)
        mcts_env = SimWorld(config, game_configs, simulator=True).get_world()
        m = MCTS(MCTS_config, mcts_env, player=1)
        mcts_env.reset_states(player_start=1)
        if GUI:
            GUI.visualize_move(World)
        while not World.is_won():
            if World.get_current_player() == 1:
                i = 0
                t = time.time()
                while (i < 1000) and (time.time() - t < 2):
                    m.itr()
                    i += 1
                norm, moves = m.norm_distr()
                move = moves[norm.index(max(norm))]
            else:
                move = int(input(f"\nMove: "))
            boo = World.play_move(move)
            m.mcts_move(move)
            if GUI and boo:
                GUI.visualize_move(World, move)


if __name__ == "__main__":
    play(2)
