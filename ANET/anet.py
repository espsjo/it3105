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
        self.MIN_LR = ANET_config["MIN_LR"]
        self.BUILD_CONV = ANET_config["BUILD_CONV"]

        self.ALL_MOVES = self.Environment.get_actions()
        self.epnr = 0

        self.OUTPUT_SIZE = len(self.ALL_MOVES)
        self.BOARD_SIZE = int(np.sqrt(self.OUTPUT_SIZE))

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
        if self.BUILD_CONV:
            model = self.build_conv_model()
        else:
            model = self.build_normal()

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
            monitor="val_loss", patience=4, min_delta=0.001
        )

        def scheduler(epoch, lr):
            if self.epnr >= self.EPISODES_BEFORE_LR_RED:
                return max(
                    self.LEARNING_RATE * (self.LR_SCALE_FACTOR**self.epnr),
                    self.MIN_LR,
                )
            return self.LEARNING_RATE

        lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

        if self.MODIFY_STATE:
            x = [self.modify_state(i) for i in x]

        x = [self.convolute(s, player2_rep_as_2=not self.MODIFY_STATE) for s in x]

        if self.TEMPERATURE != None:
            y = [i ** (1 / self.TEMPERATURE) for i in y]
            y = [i / sum(y) for i in y]

        hist = self.model.fit(
            np.array(x),
            np.array(y),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[callback, lrs],
            validation_split=0.25,
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
        state = self.convolute(state, player2_rep_as_2=not self.MODIFY_STATE)

        if litemodel != None:
            pred = litemodel.predict_single(state)
        else:
            pred = self.model(np.expand_dims(state, axis=0))[0].numpy()

        for i, j in enumerate(self.ALL_MOVES):
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

        return self.ALL_MOVES[np.argmax(norm_distr)]

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

    def unflatten_state(self, state):
        """
        Method for unflattening state - similar to the one implemented in hex.py
        Parameters:
            state: (np.ndarray) List of the state
        Returns:
            np.ndarray
        """
        board_size = self.BOARD_SIZE
        s = []
        r = 0
        for i in range(board_size):
            s.append([state[n] for n in range(r, board_size * (i + 1))])
            r += board_size
        state = np.array(s)
        return state

    def convolute(self, state, player2_rep_as_2: bool == False):
        """
        Method for convoluting a board game state. Layer 0/1/2 are tiles occupied by player1/player2/empty.
        Layer 3/4 represent whose turn it is.
        Returns a Board_size x Board_size matrix x 5 array
        Parameters:
            state: (np.ndarray) List of the state including the player turn as the first element
            player2_rep_as_2: (bool) If player two is represented with 2: True, with -1: False
        Returns:
            np.ndarray
        """
        final_matrix = []
        turn = state[0]
        state = state[1:]
        size = len(state)

        features = [1, 2, 0] if player2_rep_as_2 else [1, -1, 0]
        for feat in features:
            state_list = [1 if state[tile] == feat else 0 for tile in range(size)]
            final_matrix.append(self.unflatten_state(state_list))

        features = [1, 2] if player2_rep_as_2 else [1, -1]
        for feat in features:
            turn_list = [1 if turn == feat else 0 for i in range(size)]
            final_matrix.append(self.unflatten_state(turn_list))

        final_matrix = np.rollaxis(np.array(final_matrix), 0, 3)
        return np.array(final_matrix)

    def build_normal(self) -> ks.models.Sequential:
        """
        Method for building a standard sequential model. Can be used with NIM or other games as well.
        Parameters:
            None
        Returns:
            ks.models.Sequential
        """
        model = ks.models.Sequential()
        hidden_dense = self.HIDDEN_LAYERS[0]

        model.add(
            ks.layers.Input(
                shape=(self.INPUT_SIZE,),
            )
        )
        for i in range(len(hidden_dense)):
            dim = hidden_dense[i]
            model.add(
                ks.layers.Dense(
                    dim,
                    activation=self.ACTIVATION,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )
            # if i + 1 != len(hidden_dense):
            #     model.add(ks.layers.Dropout(0.5))

        model.add(
            ks.layers.Dense(
                self.OUTPUT_SIZE,
                activation="softmax",
            )
        )

    def build_conv_model(self) -> ks.models.Sequential:
        """
        Method for building a sequential convolutional network
        Parameters:
            None
        Returns:
            ks.models.Sequential
        """
        activ = eval("ks.activations." + self.ACTIVATION)
        hidden_dense, hidden_conv = self.HIDDEN_LAYERS

        model = ks.models.Sequential()

        model.add(
            ks.layers.Input(
                shape=(self.BOARD_SIZE, self.BOARD_SIZE, 5),
            )
        )

        model.add(
            ks.layers.Conv2D(
                hidden_conv[0],
                kernel_size=(3, 3),
                input_shape=(self.BOARD_SIZE, self.BOARD_SIZE, 5),
                padding="same",
            )
        )
        model.add(ks.layers.BatchNormalization())
        model.add(ks.layers.Activation(activ))

        for i in range(1, len(hidden_conv)):
            units = hidden_conv[i]
            model.add(
                ks.layers.Conv2D(
                    units,
                    kernel_size=(3, 3),
                    activation=self.ACTIVATION,
                    padding="same",
                )
            )
            if i != len(hidden_conv) - 1:
                model.add(ks.layers.BatchNormalization())
        model.add(ks.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
        model.add(ks.layers.Dropout(0.4))

        model.add(ks.layers.Flatten())

        for i in range(len(hidden_dense)):
            dim = hidden_dense[i]
            model.add(
                ks.layers.Dense(
                    dim,
                    activation=self.ACTIVATION,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )

        model.add(ks.layers.Dense(self.OUTPUT_SIZE, activation="softmax"))

        return model
