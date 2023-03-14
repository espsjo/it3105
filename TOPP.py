from ANET.anet import ANET
from Environments.simworld import SimWorld
from Environments.visualizer import Visualizer
from Config.config import config, game_configs, ANET_config, TOPP_config
import numpy as np
import matplotlib.pyplot as plt
import os
import re


class TOPP:
    """
    Class for carrying out a TOPP turnament, with the ability to pit two policies against eachother
    Paramters:
        None
    """

    def __init__(self) -> None:
        config["UI_ON"] = TOPP_config["TOPP_UI"]
        self.env = SimWorld(config, game_configs, simulator=False).get_world()
        UI_ON = config["UI_ON"]
        self.GUI = Visualizer(config, game_configs).get_GUI() if UI_ON else None
        self.PATH = TOPP_config["LOAD_PATH"]

    def play_game(
        self, act1: ANET, act2: ANET, probabilistic: bool, greedy_bias: float
    ) -> int:
        """
        Function which plays one game between two actors. Returns the winning player (1 / 2)
        Paramters:
            act1: (ANET) Player 1
            act2: (ANET) Player 2
            probabilistic: (bool) If choosing move is always best or by probability
            greedy_bias: (float) A bias towards the best move (1 -> Always choose best)
        Returns:
            int
        """
        self.env.reset_states(player_start=1)
        if self.GUI:
            self.GUI.visualize_move(self.env)

        while not self.env.is_won():

            player: ANET = act1 if self.env.get_current_player() == 1 else act2
            if probabilistic:
                norm_distr = player.action_distrib(
                    self.env.get_state(flatten=True, include_turn=True),
                    self.env.get_legal_moves(),
                )
                # Finds the 3 max indexes
                ind = np.argpartition(norm_distr, -5)[-5:]
                # Sorts them
                ind = ind[np.argsort(np.array(norm_distr)[ind])]
                ind = np.flip(ind)

                r = np.random.random()
                # Sets to the best move
                corr_moves = self.env.get_actions()
                move = corr_moves[ind[0]]

                # Randomly might select max2, max3 to be the new move
                for i in range(5):
                    r -= norm_distr[ind[i]] + greedy_bias
                    if r <= 0:
                        move = corr_moves[ind[i]]
                        break
            else:
                move = player.get_action(
                    self.env.get_state(flatten=True, include_turn=True),
                    self.env.get_legal_moves(),
                    choose_greedy=True,
                )

            foo = self.env.play_move(move)
            if self.GUI and foo:
                self.GUI.visualize_move(self.env, move)

        return self.env.get_winner()

    def round_ribbon(
        self,
        model_names: list,
        games=2,
        probabilistic: bool = False,
        greedy_bias=0.2,
    ) -> dict:
        """
        Function for playing all models against eachother x number of games. Returns a dict showing games played and won per model name.
        Paramters:
            model_names: (list) Names of the models to play against eachother
            games: (int) Number of games to play
            probabilistic: (bool) If choosing move is always best or by probability
            greedy_bias: (float) A bias towards the best move (1 -> Always choose best)
        Returns:
            dict
        """
        actors = {}
        played_won = {}
        count = len(model_names)
        for i, name in enumerate(model_names):
            ANET_config["LOAD_PATH"] = self.PATH
            actors[i] = ANET(ANET_config, self.env, model_name=name)
            played_won[name] = [0, 0]  # Played, won

        for a1 in range(count - 1):
            for a2 in range(a1 + 1, count):
                for game in range(games):
                    a1first = game + 1 <= (games // 2)
                    p1 = actors[a1] if a1first else actors[a2]
                    p2 = actors[a2] if a1first else actors[a1]

                    win = self.play_game(p1, p2, probabilistic, greedy_bias)

                    if a1first:
                        winner = a1 if win == 1 else a2
                    else:
                        winner = a2 if win == 1 else a1
                    if self.GUI:
                        print(
                            f"{model_names[a1]} ({1 if a1first else 2}) vs {model_names[a2]} ({2 if a1first else 1}) -> {model_names[winner]} won"
                        )

                    played_won[model_names[a1]][0] = played_won[model_names[a1]][0] + 1
                    played_won[model_names[a2]][0] = played_won[model_names[a2]][0] + 1

                    played_won[model_names[winner]][1] = (
                        played_won[model_names[winner]][1] + 1
                    )

        return played_won


"""
Main function for running the round-ribbon tournament. The function reads all names out of the Models/TOPP and loads the models
(Might update to only catch exception, but for now only store models in that folder)
Displays a graphic visualization of the results or only prints to console, depending on settings
"""
if __name__ == "__main__":
    topp = TOPP()

    # Round-ribbon
    stats = topp.round_ribbon(
        model_names=[
            os.path.splitext(f)[0] for f in os.listdir(TOPP_config["LOAD_PATH"])
        ],
        games=TOPP_config["GAMES"],
        probabilistic=TOPP_config["PROBABILISTIC"],
        greedy_bias=TOPP_config["GREEDY_BIAS"],
    )

    # Plotting
    if TOPP_config["PLOT_STATS"]:
        names = list(stats.keys())
        num = np.arange(len(names))
        stat = [i[1] for i in stats.values()]
        games = np.average([i[0] for i in stats.values()])
        plt.figure(figsize=(13, 8))
        plt.bar(num, stat, alpha=0.5)
        plt.axhline(y=games, color="g", linestyle="-")
        new_names = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in names]
        plt.xticks(num, new_names, rotation="vertical")
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.15)
        plt.ylabel("Games won")
        plt.title("TOPP Results")
        plt.grid()
        plt.show()
    else:
        print(stats)
