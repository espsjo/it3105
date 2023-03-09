from ANET.anet import ANET
from Environments.simworld import SimWorld
from Environments.visualizer import Visualizer
from Config.config import config, game_configs, ANET_config
import numpy as np


class TOPP:
    def __init__(self) -> None:
        self.env = SimWorld(config, game_configs, simulator=False).get_world()
        UI_ON = config["UI_ON"]
        self.GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None

    def play_game(self, act1: ANET, act2: ANET):
        self.env.reset_states(player_start=1)
        if self.GUI:
            self.GUI.visualize_move(self.env)

        while not self.env.is_won():

            player: ANET = act1 if self.env.get_current_player() == 1 else act2

            move = player.get_action(
                self.env.get_state(flatten=True, include_turn=True),
                self.env.get_legal_moves(),
                choose_greedy=True,
            )

            foo = self.env.play_move(move)
            if self.GUI and foo:
                self.GUI.visualize_move(self.env, move)

        return self.env.get_winner()

    def round_ribbon(self, model_names: list, games=6):
        actors = {}
        played_won = {}
        count = len(model_names)
        for i, name in enumerate(model_names):
            actors[i] = ANET(ANET_config, self.env, model_name=name)
            played_won[name] = [0, 0]  # Played, won

        for a1 in range(count - 1):
            for a2 in range(a1 + 1, count):
                for game in range(games):
                    a1first = game <= (games // 2)
                    p1 = actors[a1] if a1first else actors[a2]
                    p2 = actors[a2] if a1first else actors[a1]

                    win = self.play_game(p1, p2)

                    if a1first:
                        winner = a1 if win == 1 else a2
                    else:
                        winner = a2 if win == 1 else a1

                    played_won[model_names[a1]][0] = played_won[model_names[a1]][0] + 1
                    played_won[model_names[a2]][0] = played_won[model_names[a2]][0] + 1

                    played_won[model_names[winner]][1] = (
                        played_won[model_names[winner]][1] + 1
                    )

        return played_won, actors


if __name__ == "__main__":
    topp = TOPP()
    models = ["LONG_hex_4x4_4 copy", "LONG_hex_4x4_4"]
    stats, _ = topp.round_ribbon(models, 3)
    print(stats)
