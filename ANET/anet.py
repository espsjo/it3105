import tensorflow as tf
from config import ANET_config
from Environments.simworld import SimWorldAbs
from keras.optimizers import Adam, Adagrad, SGD, RMSprop
import numpy as np
from time import time


class ANET:
    def __init__(self, Environment: SimWorldAbs, model=None):
        self.EPSILON_DECAY = ANET_config["EPSILON_DECAY"]
        self.epsilon = ANET_config["EPSILON"]
        self.MIN_EPSILON = ANET_config["MIN_EPSILON"]
        self.Environment = Environment

        self.all_moves = self.Environment.get_actions()

        if model != None:
            # LOAD SOME MODEL
            # self.model = load(model)
            pass
        else:
            self.LEARNING_RATE = ANET_config["EPSILON"]
            self.HIDDEN_LAYERS = ANET_config["HIDDEN_LAYERS"]
            self.ACTIVATION = ANET_config["ACTIVATION"]
            self.OPTIMIZER = ANET_config["OPTIMIZER"]
            self.LOSS_FUNC = ANET_config["LOSS_FUNC"]
            self.EPOCHS = ANET_config["EPOCHS"]
            self.BATCH_SIZE = ANET_config["BATCH_SIZE"]

            self.INPUT_SIZE = len(
                self.Environment.get_state(flatten=True, include_turn=True)
            )
            self.OUTPUT_SIZE = len(self.all_moves)
            self.construct_model()

    def construct_model(self):
        optimizers = {
            "Adam": Adam(learning_rate=self.LEARNING_RATE),
            "Adagrad": Adagrad(learning_rate=self.LEARNING_RATE),
            "SGD": SGD(learning_rate=self.LEARNING_RATE),
            "RMSprop": RMSprop(learning_rate=self.LEARNING_RATE),
        }

        model = tf.keras.models.Sequential(
            [
                (tf.keras.layers.InputLayer((self.INPUT_SIZE,))),
                *[
                    (tf.keras.layers.Dense(l_size, activation=self.ACTIVATION))
                    for l_size in self.HIDDEN_LAYERS
                ],
                (tf.keras.layers.Dense(self.OUTPUT_SIZE, activation="softmax")),
            ]
        )

        model.compile(
            optimizer=optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=["accuracy", "mse"],
        )
        self.model = model

    def train(self, minibatch):
        state = np.array([x[0] for x in minibatch])
        distr = np.array([x[1] for x in minibatch])

        self.model.fit(state, distr, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)

    def epsilon_decay(self):
        x = self.epsilon * self.EPSILON_DECAY
        self.epsilon = min(x, self.MIN_EPSILON)

    def save_model(self, name, path):
        self.model.save(str(path) + "/" + str(name))

    def action_distrib(self, state, legal_moves):
        pred = self.model(np.expand_dims(state, axis=0))[0].numpy()

        for i, j in enumerate(self.all_moves):
            if j not in legal_moves:
                pred[i] = 0.0000000000001

        return pred / np.sum(pred)

    def get_action(self, state, legal_moves, choose_greedy: bool):

        if not choose_greedy:
            if np.random.random() < self.epsilon:
                return int(np.random.choice(legal_moves))

        norm_distr = self.action_distrib(state, legal_moves)
        move = self.all_moves[np.argmax(norm_distr)]
        return move

    def random_action(self, legal_moves):
        return int(np.random.choice(legal_moves))

    def evaluate(self, target_distr, state, legal_moves):
        distr = self.action_distrib(state, legal_moves)
        x = np.absolute(target_distr - distr)

        print(
            f"""
        list: {x}, 
        max: {max(x)}, 
        min: {min(x)}, 
        avg: {np.average(x)}"""
        )
