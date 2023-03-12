import tensorflow as tf
from tensorflow import keras as ks
from Environments.simworld import SimWorldAbs
import numpy as np
from .litemodel import LiteModel


class ANET:
    """
    Class for initializing and loading neural network models. Also implements logic for training the model as well as predicting new samples.
    Parameters:
        ANET_config: (dict) Dictionary containing key settings for the neural network
        Environment: (SimWorldAbs) The simulated world to extract key parameters as input and output size
        model_name: (str) Name of model to load, inits new if None
    """

    def __init__(self, ANET_config, Environment: SimWorldAbs, model_name=None):
        self.EPSILON_DECAY = ANET_config["EPSILON_DECAY"]
        self.epsilon = ANET_config["EPSILON"]
        self.MIN_EPSILON = ANET_config["MIN_EPSILON"]
        self.LOAD_PATH = ANET_config["LOAD_PATH"]
        self.MODIFY_STATE = ANET_config["MODIFY_STATE"]
        self.Environment = Environment
        self.history = []
        self.TEMPERATURE = ANET_config["TEMPERATURE"]
        self.EPISODES_BEFORE_LR_RED = ANET_config["EPISODES_BEFORE_LR_RED"]
        self.LR_SCALE_FACTOR = ANET_config["LR_SCALE_FACTOR"]

        self.all_moves = self.Environment.get_actions()
        self.epnr = 0

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

    def construct_model(self) -> None:
        """
        Creates a new model based on parameters
        Parameters:
            None
        Returns:
            None
        """
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

        model.add(
            ks.layers.Input(
                shape=(self.INPUT_SIZE,),
            )
        )
        for i in range(len(self.HIDDEN_LAYERS)):
            dim = self.HIDDEN_LAYERS[i]
            model.add(
                ks.layers.Dense(
                    dim,
                    activation=self.ACTIVATION,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )
            if i + 1 != len(self.HIDDEN_LAYERS):
                model.add(ks.layers.Dropout(0.1))

        model.add(
            ks.layers.Dense(
                self.OUTPUT_SIZE,
                activation="softmax",
            )
        )

        model.compile(
            optimizer=optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=[tf.keras.metrics.categorical_accuracy],
        )
        self.model = model
        self.model.summary()

    def train(self, minibatch, epnr) -> None:
        """
        Trains the model on a minibatch of cases
        Parameters:
            minibatch: (list) A minibatch of cases to train network on (x: state, y: target)
            epnr: (int) The current episode number
        Returns:
            None
        """
        self.epnr = epnr
        x, y = zip(*minibatch)
        callback = ks.callbacks.EarlyStopping(
            monitor="loss", patience=5, min_delta=0.001
        )

        def scheduler(epoch, lr):
            if epoch < 6:
                if self.epnr >= self.EPISODES_BEFORE_LR_RED:
                    return self.LEARNING_RATE * self.LR_SCALE_FACTOR
                return self.LEARNING_RATE
            else:
                return lr * tf.math.exp(-0.1)

        lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if self.MODIFY_STATE:
            x = [self.modify_state(i) for i in x]

        if self.TEMPERATURE != None:
            y = [i ** (1 / self.TEMPERATURE) for i in y]
            y = [i / sum(y) for i in y]

        hist = self.model.fit(
            np.array(x),
            np.array(y),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[callback, lrs],
            validation_split=0.2,
        )
        self.history.append(hist)

    def epsilon_decay(self) -> None:
        """
        Decays epsilon according to parameters
        Parameters:
            None
        Returns:
            None
        """
        x = self.epsilon * self.EPSILON_DECAY
        self.epsilon = max(x, self.MIN_EPSILON)

    def save_model(self, name, path) -> None:
        """
        Saves the model to a given path
        Parameters:
            name: (str) Name to save
            path: (str) Path to save model to
        Returns:
            None
        """
        self.model.save(str(path) + "/" + str(name) + ".h5")

    def action_distrib(
        self, state, legal_moves, litemodel: LiteModel = None
    ) -> np.ndarray:
        """
        Uses the model to predict a distribution based on input state. Normalizes to sum to 1
        Parameters:
            state: (np.ndarray) The state representation flattened
            legal_moves: (np.ndarray / list) A set of legal moves in current state
            litemodel: (LiteModel) If created a litemodel, use this to carry out predicitions for faster runtime
        Returns:
            np.ndarray
        """
        if self.MODIFY_STATE:
            state = self.modify_state(state)
        if litemodel != None:
            pred = litemodel.predict_single(state)
        else:
            pred = self.model(np.expand_dims(state, axis=0))[0].numpy()

        for i, j in enumerate(self.all_moves):
            if j not in legal_moves:
                pred[i] = 0

        return pred / max(np.sum(pred), 0.00000000001)

    def get_action(
        self, state, legal_moves, choose_greedy: bool, litemodel: LiteModel = None
    ) -> np.ndarray:
        """
        Returns an action based on the distribution from the model. If not choosing greedy, returns random legal move with specified chance
        Parameters:
            state: (np.ndarray) The state representation flattened
            legal_moves: (np.ndarray / list) A set of legal moves in current state
            choose_greedy: (bool) If the function should choose the greedy best move or random at a given epsilon interval
            litemodel: (LiteModel) If created a litemodel, use this to carry out predicitions for faster runtime
        Returns:
            np.ndarray
        """

        if not choose_greedy:
            if np.random.random() < self.epsilon:
                return int(np.random.choice(legal_moves))

        norm_distr = self.action_distrib(state, legal_moves, litemodel)
        if not norm_distr.any():
            return int(np.random.choice(legal_moves))

        return self.all_moves[np.argmax(norm_distr)]

    def load(self, model_name) -> None:
        """
        Loads a given model from the load path specified in the parameters
        Parameters:
            model_name: (str) Name of model to load
        Returns:
            None
        """
        x = tf.keras.models.load_model(f"{self.LOAD_PATH}/{model_name}", compile=False)
        self.model = x

    def get_history(self) -> list:
        """
        Returns the history of training
        Parameters:
            None
        Returns:
            list(History)
        """

        return self.history

    def modify_state(self, x) -> np.ndarray:
        """
        Modifies states to represent player 2 as -1
        Parameters:
            x: (np.ndarray / list) List to modify
        Returns:
            np.ndarray
        """
        return np.array([i if i != 2 else -1 for i in x])
