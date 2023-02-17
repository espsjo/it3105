from config import config, game_configs, MCTS_config, RLLearner_config, ANET_config
import numpy as np
from time import time
from Environments.simworld import SimWorld
from MCTS.mcts import MCTS
from ANET.anet import ANET
import random

from Environments.visualizer import Visualizer


class RLLearner:
    def __init__(self):
        self.MCTS_config = MCTS_config
        self.MAX_TIME = MCTS_config["MAX_TIME"]
        self.MAX_SIMS = MCTS_config["MAX_SIMS"]
        self.EPISODES = RLLearner_config["EPISODES"]
        self.BUFFER_SIZE = RLLearner_config["BUFFER_SIZE"]
        self.MINIBATCH_SIZE = RLLearner_config["MINIBATCH_SIZE"]
        self.SAVE = RLLearner_config["SAVE"]
        self.SAVE_INTERVAL = RLLearner_config["SAVE_INTERVAL"]
        self.SAVE_PATH = RLLearner_config["SAVE_PATH"]

        self.simworld = SimWorld(config, game_configs, simulator=False).get_world()
        self.mcts_world = SimWorld(config, game_configs, simulator=True).get_world()

        self.actor = ANET(self.simworld)

        self.rbuf_index = 0
        # self.GUI = Visualizer(config, game_configs).get_GUI()

    def train(self):

        rbuf = []

        if self.SAVE:
            ###
            # LOGIC FOR SAVING NET
            ###
            pass

        for eps in range(self.EPISODES):

            # Varying player start and reseting envs
            player = np.random.choice([1, 2])

            self.simworld.reset_states(player_start=player)
            self.mcts_world.reset_states(player_start=player)
            monte_carlo = MCTS(self.MCTS_config, self.mcts_world, player=player)
            # Run one episode
            # self.GUI.visualize_move(self.simworld)
            while not self.simworld.is_won():
                t_start = time()
                # Iterating MCTS
                i = 0
                while (i < self.MAX_SIMS) and (time() - t_start < self.MAX_TIME):
                    monte_carlo.itr(self.actor)  ### PASS INN ACTOR TO GUIDE ROLLOUTS
                    i += 1
                # Retriving normal distributions as well as indexes of corresponding moves
                norm_distr, corr_moves = monte_carlo.norm_distr()
                state = self.simworld.get_state(flatten=True, include_turn=True)

                # Creating a case
                rbuf_case = (state, np.array(norm_distr))

                # Keeping a fixed size rbuf
                if self.rbuf_index < self.BUFFER_SIZE:
                    rbuf.append(rbuf_case)
                else:
                    i = self.rbuf_index % self.BUFFER_SIZE
                    rbuf[i] = rbuf_case
                self.rbuf_index += 1

                # Choosing and playing move
                move = corr_moves[norm_distr.index(max(norm_distr))]
                self.simworld.play_move(move)
                monte_carlo.mcts_move(move)
                # self.GUI.visualize_move(self.simworld, move)

            # Sampling a minibatch

            mbatch = random.sample(rbuf, min(len(rbuf), self.MINIBATCH_SIZE))
            self.actor.train(mbatch)
            self.actor.epsilon_decay()

            s, t = random.choice(rbuf)
            self.actor.evaluate(
                state=s,
                target_distr=t,
                legal_moves=self.simworld.get_legal_moves(s[1:]),
            )

            if (eps + 1) % self.SAVE_INTERVAL == 0 and self.SAVE:
                ###
                # LOGIC FOR SAVING NET
                ###
                pass
