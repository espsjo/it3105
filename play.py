from Environments.simworld import SimWorld
from Environments.visualizer import Visualizer
from Config.config import config, game_configs, ANET_config
from ANET.anet import ANET

"""
Ability to play against computer/human - only by registering moves in console (index form)
Returns void 
"""


def play(n_games):
    World = SimWorld(config, game_configs, simulator=False).get_world()
    UI_ON = config["UI_ON"]
    GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None

    actor = ANET(
        ANET_config=ANET_config, Environment=World, model_name="OVERNIGHT_hex_7x7_8"
    )

    for _ in range(n_games):
        World.reset_states(player_start=1)
        if GUI:
            GUI.visualize_move(World)
        while not World.is_won():
            if World.get_current_player() == 1:
                x = actor.action_distrib(
                    World.get_state(flatten=True, include_turn=True),
                    World.get_legal_moves(),
                )
                x = [(f"{n}: {str(round(i, 2))}") for n, i in enumerate(x)]
                print(x)

                move = actor.get_action(
                    World.get_state(flatten=True, include_turn=True),
                    World.get_legal_moves(),
                    choose_greedy=True,
                )
            else:
                # move = np.random.choice(World.get_legal_moves())
                move = int(input(f"\nMove: "))
            boo = World.play_move(move)
            # sleep(0.1)
            if GUI and boo:
                GUI.visualize_move(World, move)


if __name__ == "__main__":
    play(1)
