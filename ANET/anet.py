import tensorflow as tf
from tensorflow import keras as ks
from Environments.simworld import SimWorldAbs
import numpy as np


class ANET:
    """
    Initialise key parameters
    Either loads a model or creates a new one based on parameters
    Returns void
    """

    def __init__(self, ANET_config, Environment: SimWorldAbs, model_name=None):
        self.EPSILON_DECAY = ANET_config["EPSILON_DECAY"]
        self.epsilon = ANET_config["EPSILON"]
        self.MIN_EPSILON = ANET_config["MIN_EPSILON"]
        self.LOAD_PATH = ANET_config["LOAD_PATH"]
        self.Environment = Environment

        self.all_moves = self.Environment.get_actions()

        if model_name != None:
            self.load(model_name + ".h5")
        else:
            self.LEARNING_RATE = ANET_config["LEARNING_RATE"]
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

    """
    Creates a new model based on parameters
    Returns void
    """

    def construct_model(self):
        if self.LEARNING_RATE is None:
            optimizers = {
                "Adam": ks.optimizers.Adam(),
                "Adagrad": ks.optimizers.Adagrad(),
                "SGD": ks.optimizers.SGD(),
                "RMSprop": ks.optimizers.RMSprop(),
            }
        else:
            optimizers = {
                "Adam": ks.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                "Adagrad": ks.optimizers.Adagrad(learning_rate=self.LEARNING_RATE),
                "SGD": ks.optimizers.SGD(learning_rate=self.LEARNING_RATE),
                "RMSprop": ks.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            }

        model = ks.Sequential()
        model.add(ks.layers.Input(shape=(self.INPUT_SIZE,)))
        for dim in self.HIDDEN_LAYERS:
            model.add(ks.layers.Dense(dim, activation=self.ACTIVATION))
        model.add(ks.layers.Dense(self.OUTPUT_SIZE, activation="softmax"))

        model.compile(
            optimizer=optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=["accuracy", "mse"],
        )
        self.model = model
        self.model.summary()

    """
    Trains the model on a minibatch of cases (x: state, y: target)
    Returns void
    """

    def train(self, minibatch):
        x, y = zip(*minibatch)

        self.model.fit(
            np.array(x), np.array(y), batch_size=self.BATCH_SIZE, epochs=self.EPOCHS
        )

    """
    Decays epsilon according to parameters
    Returns void
    """

    def epsilon_decay(self):
        x = self.epsilon * self.EPSILON_DECAY
        self.epsilon = max(x, self.MIN_EPSILON)

    """
    Saves the model to a given path
    Returns void
    """

    def save_model(self, name, path):
        self.model.save(str(path) + "/" + str(name) + ".h5")

    """
    Uses the model to predict a distribution based on input state
    Normalizes to sum to 1
    Returns array
    """

    def action_distrib(self, state, legal_moves):
        pred = self.model(np.expand_dims(state, axis=0))[0].numpy()

        for i, j in enumerate(self.all_moves):
            if j not in legal_moves:
                pred[i] = 0

        return pred / max(np.sum(pred), 0.00000000001)

    """
    Returns an action based on the distribution from the model. If not choosing greedy, returns random legal move with specified chance
    Returns array
    """

    def get_action(self, state, legal_moves, choose_greedy: bool):

        if not choose_greedy:
            if np.random.random() < self.epsilon:
                return int(np.random.choice(legal_moves))

        norm_distr = self.action_distrib(state, legal_moves)
        if not norm_distr.any():
            return int(np.random.choice(legal_moves))

        return self.all_moves[np.argmax(norm_distr)]

    """
    Loads a given model from the load path specified in the parameters
    Returns void
    """

    def load(self, model_name):
        x = tf.keras.models.load_model(f"{self.LOAD_PATH}/{model_name}", compile=False)
        self.model = x