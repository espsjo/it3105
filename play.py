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


def play(n_games, name, human=True, net_player=1, player_start=1) -> None:
    """
    Carries out the game
    Paramters:
        n_games: (int) Games to be played
        name: (str) Model name to load
        human: (bool) If net should play against human or MCTS
        net_player: (int) What player the net should be
        player_start: (int) What player should start
    """
    World = SimWorld(config, game_configs, simulator=False).get_world()
    UI_ON = config["UI_ON"]
    GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None
    mcts_env = SimWorld(config, game_configs, simulator=True).get_world()

    actor = ANET(ANET_config=ANET_config, Environment=World, model_name=name)

    for _ in range(n_games):
        World.reset_states(player_start)
        if not human:
            mcts_env.reset_states(player_start)
            m = MCTS(MCTS_config, mcts_env, player=player_start)

        if GUI:
            GUI.visualize_move(World)
        while not World.is_won():
            if World.get_current_player() == net_player:
                x = actor.action_distrib(
                    World.get_state(flatten=True, include_turn=True),
                    World.get_legal_moves(),
                )
                x = [(f"{n}: {str(round(i, 4))}") for n, i in enumerate(x)]
                print(x)

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
                    while i < 2000 and (time.time() - t < 10):
                        # m.itr(actor=actor, greedy=True)
                        m.itr()
                        i += 1
                    print(i)
                    norm, moves = m.norm_distr()
                    move = moves[norm.index(max(norm))]
            boo = World.play_move(move)
            if not human:
                m.mcts_move(move)

            if GUI and boo:
                GUI.visualize_move(World, move)


if __name__ == "__main__":
    play(
        n_games=5,
        name="7x7/LETSGO/LETSGO_hex_7x7_8",  # "4x4/LETSGO/LETSGO_hex_4x4_4",  # "NIM_20_1_3/LETSGO_nim_20_1_3_5",
        human=True,
        net_player=1,
        player_start=1,
    )
