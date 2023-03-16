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
        self,
        config,
        game_configs,
        MCTS_config,
        RLLearner_config,
        ANET_config,
    ):
        self.ANET_CONFIG = ANET_config
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
        self.GREEDY_BIAS = RLLearner_config["GREEDY_BIAS"]

        self.simworld = SimWorld(config, game_configs, simulator=False).get_world()
        self.mcts_world = SimWorld(config, game_configs, simulator=True).get_world()

        self.actor = self.actor = ANET(
            ANET_config=ANET_config, Environment=self.simworld, model_name=None
        )

        self.litemodel = None

        self.save_ind = 0

        self.rbuf_index = 0

        self.GUI = Visualizer(config, game_configs).get_GUI()

        self.RBUF_MIN_SIZE = 0

    def load_model(
        self,
        model_name: str,
        epsilon: float = None,
        learning_rate: float = None,
        epnr: int = 0,
        RBUF_MIN_SIZE: int = 0,
    ):
        """
        Loads the given model back, and enables you to resume training.
        Will not start training the model before a given RBUF-size is reached
        Unless specified, epnr = 0 and epsilon will be loaded from config files.
        Other parameters will be loaded from config.
        Parameters:
            model_name: (str) The name of the model (loaded from path in ANET_config)
            epsilon: (float) Specify epsilon
            learning_rate: (float) Specify base learning rate (decay from ANET_config)
            epnr: (int) Specify episode_nr starting from (Used in LR decay)
            RBUF_MIN_SIZE: (int) Specify a minimum size in RBUF before resuming model training (scales down to BUFFER_SIZE if larger)
        Returns:
            None
        """

        self.RBUF_MIN_SIZE = (
            RBUF_MIN_SIZE if RBUF_MIN_SIZE < self.BUFFER_SIZE else self.BUFFER_SIZE
        )
        self.ANET_CONFIG["EPSILON"] = (
            self.ANET_CONFIG["EPSILON"] if epsilon is None else epsilon
        )
        self.actor.LEARNING_RATE = (
            self.ANET_CONFIG["LEARNING_RATE"]
            if learning_rate is None
            else learning_rate
        )
        self.actor = ANET(
            ANET_config=self.ANET_CONFIG,
            Environment=self.simworld,
            model_name=model_name,
        )
        self.actor.epnr = epnr
        self.save_ind = epnr % self.SAVE_INTERVAL

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

            # Varying player start and reseting envs (GOOD RESULTS WITHOUT, BUT SHOULD PROBABLY BE DONE)

            # Resetting environment
            # player = np.random.choice([1, 2])
            player = 1
            temp_rbuf = []
            self.simworld.reset_states(player_start=player)
            self.mcts_world.reset_states(player_start=player)
            monte_carlo = MCTS(self.MCTS_config, self.mcts_world, player=player)

            # Updating the litemodel used for prediction
            if eps % 5 == 0:
                self.litemodel = LiteModel.from_keras_model(self.actor.model)

            # Init GUI
            if self.TRAIN_UI:
                self.GUI.visualize_move(self.simworld)

            # Run one episode
            while not self.simworld.is_won():
                t_start = time()

                # Iterating MCTS
                i = 0
                while (i < self.MIN_SIMS) or (time() - t_start < self.MAX_TIME):
                    monte_carlo.itr(
                        self.actor, self.litemodel
                    )  ### PASS INN ACTOR TO GUIDE ROLLOUTS
                    i += 1
                # Printing to console the number of games rolled out as well as time spent
                print(f"{i}; {time() - t_start}")

                # Retriving normal distributions as well as indexes of corresponding moves
                norm_distr, corr_moves = monte_carlo.norm_distr()
                state = self.simworld.get_state(flatten=True, include_turn=True)

                # Adding case to RBUF temp
                rbuf_case = [state, np.array(norm_distr), 0]
                temp_rbuf.append(rbuf_case)

                # Finds the 3 max indexes
                ind = np.argpartition(norm_distr, -3)[-3:]
                # Sorts them
                ind = ind[np.argsort(np.array(norm_distr)[ind])]
                ind = np.flip(ind)

                r = np.random.random()
                # Sets to the best move
                move = corr_moves[ind[0]]

                # Randomly might select max2, max3 to be the new move
                for i in range(3):
                    r -= norm_distr[ind[i]] + self.GREEDY_BIAS
                    if r <= 0:
                        move = corr_moves[ind[i]]
                        break

                # Playing the move in the "real" game, as well as in the simulator
                moved = self.simworld.play_move(move)
                monte_carlo.mcts_move(move)

                # Visualize
                if self.TRAIN_UI and moved:
                    self.GUI.visualize_move(self.simworld, move)

            # Get reward, add to training case, finally add them to the RBUF
            r = self.simworld.get_reward(player=player)
            for i in range(len(temp_rbuf)):
                temp_rbuf[i][2] = r
                # Keeping a fixed size rbuf
                if self.rbuf_index < self.BUFFER_SIZE:
                    rbuf.append(temp_rbuf[i])
                else:
                    ind = self.rbuf_index % self.BUFFER_SIZE
                    rbuf[ind] = temp_rbuf[i]
                self.rbuf_index += 1

            # Checking for RBUF size in case of loaded model (default to 0)
            if len(rbuf) >= self.RBUF_MIN_SIZE:
                # Sampling a minibatch of cases: (x: state, y: target_distr, y_crit: target_eval)
                mbatch = random.sample(rbuf, min(len(rbuf), self.MINIBATCH_SIZE))

                # Train the model, decay epsilon
                self.actor.train(mbatch, eps)
                self.actor.epsilon_decay()

            # Saving model at given interval, as well as plotting stats
            if (eps + 1) % self.SAVE_INTERVAL == 0 and self.SAVE:
                self.actor.save_model(
                    self.SAVE_NAME + str(self.save_ind), self.SAVE_PATH
                )

                h = self.actor.get_history()
                x = []
                y = []
                y_val = []
                y_val_acc = []
                ay = []
                ay_val = []
                ay_val_acc = []
                for i, hist in enumerate(h):
                    x.append(i)
                    y.append(hist.history["loss"][-1])
                    y_val.append(hist.history["val_loss"][-1])
                    y_val_acc.append(hist.history["val_categorical_accuracy"][-1])
                    ay.append(np.average(y[-9:]))
                    ay_val.append(np.average(y_val[-9:]))
                    ay_val_acc.append(np.average(y_val_acc[-9:]))

                plt.figure(figsize=(10, 6))
                plt.plot(x, ay, label="10ep Mean Loss")
                plt.plot(x, ay_val, label="10ep Mean Val_Loss")
                plt.plot(x, ay_val_acc, ":", label="10ep Mean Val Acc")
                plt.legend()
                plt.ylim(0, max(y) + 1)

                plt.savefig(f"{self.SAVE_PATH}/{self.SAVE_NAME}_loss.pdf")
                plt.close()

                self.save_ind += 1
