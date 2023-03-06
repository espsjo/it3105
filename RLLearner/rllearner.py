import numpy as np
from time import time
from Environments.simworld import SimWorld
from MCTS.mcts import MCTS
from ANET.anet import ANET
import random
from Environments.visualizer import Visualizer
from matplotlib import pyplot as plt
from ANET.litemodel import LiteModel


class RLLearner:
    """
    Class for creating (or loading) a neural net, and then training it according to parameters in the config files
    Parameters:
        config: (dict) General config
        MCTS_config: (dict) Describing how to perform MCTS
        RLLearner_config: (dict) Describing how to perform training
        ANET_config: (dict) Describing how to initialize as well as train the neural network
    """

    def __init__(
        self, config, game_configs, MCTS_config, RLLearner_config, ANET_config
    ):
        self.MCTS_config = MCTS_config
        self.MAX_TIME = MCTS_config["MAX_TIME"]
        self.MIN_SIMS = MCTS_config["MIN_SIMS"]
        self.EPISODES = RLLearner_config["EPISODES"]
        self.BUFFER_SIZE = RLLearner_config["BUFFER_SIZE"]
        self.MINIBATCH_SIZE = RLLearner_config["MINIBATCH_SIZE"]
        self.SAVE = RLLearner_config["SAVE"]
        self.SAVE_INTERVAL = RLLearner_config["SAVE_INTERVAL"]
        self.SAVE_PATH = RLLearner_config["SAVE_PATH"]
        self.SAVE_NAME = RLLearner_config["SAVE_NAME"]
        self.TRAIN_UI = RLLearner_config["TRAIN_UI"]

        self.simworld = SimWorld(config, game_configs, simulator=False).get_world()
        self.mcts_world = SimWorld(config, game_configs, simulator=True).get_world()

        self.actor = ANET(
            ANET_config=ANET_config, Environment=self.simworld, model_name=None
        )
        self.litemodel = None

        self.save_ind = 0

        self.rbuf_index = 0

        self.GUI = Visualizer(config, game_configs).get_GUI()

    def train(self) -> None:
        """
        Training the neural net by running episodes where moves are chosen from MCTS, which again is using the Actor/Random to guide rollout moves
        Samples from a replay buffer and trains the net on cases consisting of (x: state, y: distribution from MCTS)
        Saves the net at given intervals
        Parameters:
            None
        Returns:
            None
        """
        rbuf = []

        if self.SAVE:
            self.actor.save_model(self.SAVE_NAME + str(self.save_ind), self.SAVE_PATH)
            self.save_ind += 1
            pass

        for eps in range(self.EPISODES):
            print(f"\nEPISODE: {eps+1}; EPSILON: {self.actor.epsilon}")
            # Varying player start and reseting envs
            player = np.random.choice([1, 2])

            self.simworld.reset_states(player_start=player)
            self.mcts_world.reset_states(player_start=player)
            monte_carlo = MCTS(self.MCTS_config, self.mcts_world, player=player)

            if eps % 5 == 0:
                self.litemodel = LiteModel.from_keras_model(self.actor.model)

            # Run one episode
            if self.TRAIN_UI:
                self.GUI.visualize_move(self.simworld)
            while not self.simworld.is_won():
                t_start = time()

                # Iterating MCTS
                i = 0
                while (i < self.MIN_SIMS) or (time() - t_start < self.MAX_TIME):
                    monte_carlo.itr(
                        self.actor, self.litemodel
                    )  ### PASS INN ACTOR TO GUIDE ROLLOUTS
                    i += 1
                print(f"{i}; {time() - t_start}")
                # Retriving normal distributions as well as indexes of corresponding moves
                norm_distr, corr_moves = monte_carlo.norm_distr()
                state = self.simworld.get_state(flatten=True, include_turn=True)
                # Creating a case
                rbuf_case = (state, np.array(norm_distr))

                # Keeping a fixed size rbuf
                if self.rbuf_index < self.BUFFER_SIZE:
                    rbuf.append(rbuf_case)
                else:
                    ind = self.rbuf_index % self.BUFFER_SIZE
                    rbuf[ind] = rbuf_case
                self.rbuf_index += 1

                # Choosing and playing move
                move = corr_moves[norm_distr.index(max(norm_distr))]
                moved = self.simworld.play_move(move)
                monte_carlo.mcts_move(move)
                if self.TRAIN_UI and moved:
                    self.GUI.visualize_move(self.simworld, move)
            # Sampling a minibatch
            mbatch = random.sample(rbuf, min(len(rbuf), self.MINIBATCH_SIZE))
            self.actor.train(mbatch)
            self.actor.epsilon_decay()

            if (eps + 1) % self.SAVE_INTERVAL == 0 and self.SAVE:
                self.actor.save_model(
                    self.SAVE_NAME + str(self.save_ind), self.SAVE_PATH
                )

                h = self.actor.get_history()
                x = []
                y = []
                y_val = []
                for i, hist in enumerate(h):
                    x.append(i)
                    y.append(hist.history["loss"][-1])
                    y_val.append(hist.history["val_loss"][-1])

                # bug here closes visualizing of training...
                plt.plot(x, y, label="Loss")
                plt.plot(x, y_val, label="Val_Loss")
                plt.legend()
                plt.ylim(0, max(y) + 1)
                plt.savefig(
                    f"{self.SAVE_PATH}/{self.SAVE_NAME}{str(self.save_ind)}_loss_hist.pdf"
                )
                plt.close()

                self.save_ind += 1
