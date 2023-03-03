from Environments.simworld import SimWorld
from MCTS.mcts import MCTS
from Config.config import config, game_configs, MCTS_config
import time
from Environments.visualizer import Visualizer


def main(n_games):
    World = SimWorld(config, game_configs, simulator=False).get_world()
    mcts_env = SimWorld(config, game_configs, simulator=True).get_world()
    UI_ON = config["UI_ON"]
    GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None
    player_start = 1
    for x in range(n_games):
        World.reset_states(player_start=player_start)
        mcts_env.reset_states(player_start=player_start)
        m = MCTS(MCTS_config, mcts_env, player=player_start)
        if GUI:
            GUI.visualize_move(World)
        while not World.is_won():
            i = 0
            t = time.time()
            while (i < 1000) and (time.time() - t < 3):
                m.itr()
                i += 1
            norm, moves = m.norm_distr()
            move = moves[norm.index(max(norm))]
            World.play_move(move)
            m.mcts_move(move)
            if GUI:
                GUI.visualize_move(World, move)


if __name__ == "__main__":
    main(1)
