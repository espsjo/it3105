from Environments.simworld import SimWorld
from Environments.visualizer import Visualizer
from Config.config import config, game_configs, ANET_config, MCTS_config
from ANET.anet import ANET
from MCTS.mcts import MCTS
import time

"""
Ability to play a net against either a human or pure MCTS - only by registering moves in console (index form)
Returns void 
"""


def play(n_games, name, human=True, net_player1=True) -> None:
    """
    Carries out the game
    Paramters:
        n_games: (int) Games to be played
        name: (str) Model name to load
        human: (bool) If net should play against human or MCTS
        net_player1: (bool) If the net should be player 1
    """
    World = SimWorld(config, game_configs, simulator=False).get_world()
    UI_ON = config["UI_ON"]
    GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None
    mcts_env = SimWorld(config, game_configs, simulator=True).get_world()

    actor = ANET(ANET_config=ANET_config, Environment=World, model_name=name)

    for _ in range(n_games):
        World.reset_states(player_start=1)
        if not human:
            mcts_env.reset_states(player_start=1)
            m = MCTS(MCTS_config, mcts_env, player=1)

        if GUI:
            GUI.visualize_move(World)
        while not World.is_won():
            if World.get_current_player() == net_player1:
                # x = actor.action_distrib(
                #     World.get_state(flatten=True, include_turn=True),
                #     World.get_legal_moves(),
                # )
                # x = [(f"{n}: {str(round(i, 2))}") for n, i in enumerate(x)]
                # print(x)

                move = actor.get_action(
                    World.get_state(flatten=True, include_turn=True),
                    World.get_legal_moves(),
                    choose_greedy=True,
                )
            else:
                if human:
                    move = int(input(f"\nMove: "))
                else:
                    i = 0
                    t = time.time()
                    while (i < 1000) and (time.time() - t < 3):
                        m.itr()
                        i += 1
                    norm, moves = m.norm_distr()
                    move = moves[norm.index(max(norm))]
            boo = World.play_move(move)
            if not human:
                m.mcts_move(move)

            # sleep(0.1)
            if GUI and boo:
                GUI.visualize_move(World, move)


if __name__ == "__main__":
    play(n_games=5, name="4x4/OVERNIGHT_hex_4x4_8", human=False, net_player1=False)
