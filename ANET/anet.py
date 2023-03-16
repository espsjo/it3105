import tensorflow as tf
from tensorflow import keras as ks
from Environments.simworld import SimWorldAbs
import numpy as np
from .litemodel import LiteModel
from ANET.network import Network


class ANET:
    """
    Class for initializing and loading neural network models. Also implements logic for training the model as well as predicting new samples.
    Parameters:
        ANET_config: (dict) Dictionary containing key settings for the neural network
        Environment: (SimWorldAbs) The simulated world to extract key parameters as input and output size
        model_name: (str) Name of model to load, inits new if None
    """

    def __init__(self, ANET_config, Environment: SimWorldAbs, model_name=None):
        self.ANET_config = ANET_config
        self.EPSILON_DECAY = ANET_config["EPSILON_DECAY"]
        self.epsilon = ANET_config["EPSILON"]
        self.MIN_EPSILON = ANET_config["MIN_EPSILON"]
        self.LOAD_PATH = ANET_config["LOAD_PATH"]
        self.MODIFY_STATE = ANET_config["MODIFY_STATE"]
        self.Environment = Environment
        self.history = []
        self.EPISODES_BEFORE_LR_RED = ANET_config["EPISODES_BEFORE_LR_RED"]
        self.LR_SCALE_FACTOR = ANET_config["LR_SCALE_FACTOR"]
        self.MIN_LR = ANET_config["MIN_LR"]
        self.BUILD_CONV = ANET_config["BUILD_CONV"]
        self.LEARNING_RATE = ANET_config["LEARNING_RATE"]
        self.EPOCHS = ANET_config["EPOCHS"]
        self.BATCH_SIZE = ANET_config["BATCH_SIZE"]
        self.EARLY_STOP_PAT = ANET_config["EARLY_STOP_PAT"]

        self.ALL_MOVES = self.Environment.get_actions()
        self.epnr = 0

        self.OUTPUT_SIZE = len(self.ALL_MOVES)
        self.BOARD_SIZE = int(np.sqrt(self.OUTPUT_SIZE))
        self.INPUT_SIZE = self.OUTPUT_SIZE + 1

        if model_name != None:
            self.load(model_name + ".h5")

        else:
            self.construct_model(ANET_config=ANET_config)

    def construct_model(self, ANET_config) -> None:
        """
        Creates a new model based on parameters
        Parameters:
            ANET_config: (dict) Config containing anet params
        Returns:
            None
        """
        network = Network(ANET_config=ANET_config)
        if self.BUILD_CONV:
            self.model = network.build_conv_model(
                in_size=self.BOARD_SIZE, out_size=self.OUTPUT_SIZE
            )
        else:
            self.model = network.build_normal(
                in_size=self.BOARD_SIZE, out_size=self.OUTPUT_SIZE
            )

    def train(self, minibatch, epnr) -> None:
        """
        Trains the model on a minibatch of cases
        Parameters:
            minibatch: (list) A minibatch of cases to train network on (x: state, y: target, y_crit: target eval)
            epnr: (int) The current episode number
        Returns:
            None
        """
        self.epnr = epnr
        # Load data
        x, y, y_crit = zip(*minibatch)

        # Callbacks
        callback = ks.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.EARLY_STOP_PAT, min_delta=0.001
        )

        def scheduler(epoch, lr):
            if self.epnr >= self.EPISODES_BEFORE_LR_RED:
                return max(
                    self.LEARNING_RATE * (self.LR_SCALE_FACTOR**self.epnr),
                    self.MIN_LR,
                )
            return self.LEARNING_RATE

        lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

        # State augmentation
        if self.MODIFY_STATE:
            x = [self.modify_state(i) for i in x]

        if self.BUILD_CONV:
            l = [
                [s[0], np.array(s[1:]).reshape(self.BOARD_SIZE, self.BOARD_SIZE)]
                for s in x
            ]
            x = [
                self.convolute(tup[1], tup[0], player2_rep_as_2=not self.MODIFY_STATE)
                for tup in l
            ]

        # Training
        hist = self.model.fit(
            np.array(x),
            np.array(y),
            batch_size=self.BATCH_SIZE,
            epochs=self.EPOCHS,
            callbacks=[callback, lrs],
            validation_split=0.3,
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
        if self.BUILD_CONV:
            turn = state[0]
            state = np.array(state[1:]).reshape(self.BOARD_SIZE, self.BOARD_SIZE)
            state = self.convolute(state, turn, player2_rep_as_2=not self.MODIFY_STATE)

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
        self.LEARNING_RATE = self.ANET_config["LEARNING_RATE"]
        self.OPTIMIZER = self.ANET_config["OPTIMIZER"]
        self.LOSS_FUNC = self.ANET_config["LOSS_FUNC"]

        if self.LEARNING_RATE is None:
            self.optimizers = {
                "Adam": ks.optimizers.Adam(),
                "Adagrad": ks.optimizers.Adagrad(),
                "SGD": ks.optimizers.SGD(),
                "RMSprop": ks.optimizers.RMSprop(),
            }
        else:
            self.optimizers = {
                "Adam": ks.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                "Adagrad": ks.optimizers.Adagrad(learning_rate=self.LEARNING_RATE),
                "SGD": ks.optimizers.SGD(learning_rate=self.LEARNING_RATE),
                "RMSprop": ks.optimizers.RMSprop(learning_rate=self.LEARNING_RATE),
            }

        x = tf.keras.models.load_model(f"{self.LOAD_PATH}/{model_name}", compile=False)
        x.compile(
            optimizer=self.optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=[tf.keras.metrics.categorical_accuracy],
        )
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

    ### DATA AUGMENTATION ###

    def modify_state(self, x) -> np.ndarray:
        """
        Modifies states to represent player 2 as -1
        Parameters:
            x: (np.ndarray / list) List to modify
        Returns:
            np.ndarray
        """
        return np.array([i if i != 2 else -1 for i in x])

    def convolute(self, state, turn, player2_rep_as_2: bool = False):
        """
        Method for convoluting a board game state. Layer 0/1/2 are tiles occupied by player1/player2/empty.
        Layer 3/4 represent whose turn it is.
        Rolls the axis to correct dimensions.
        Returns a Board_size x Board_size x 5 array
        Parameters:
            state: (np.ndarray) List of the state including the player turn as the first element
            player2_rep_as_2: (bool) If player two is represented with 2: True, with -1: False
        Returns:
            np.ndarray
        """
        final_matrix = []
        board_size = self.BOARD_SIZE

        features = [1, 2, 0] if player2_rep_as_2 else [1, -1, 0]
        for feat in features:
            x = np.where(state == feat, 1, 0)
            final_matrix.append(x)

        if turn == 1:
            final_matrix.append(np.ones((board_size, board_size)))
            final_matrix.append(np.zeros((board_size, board_size)))
        else:
            final_matrix.append(np.zeros((board_size, board_size)))
            final_matrix.append(np.ones((board_size, board_size)))

        return np.rollaxis(np.array(final_matrix), 0, 3)
