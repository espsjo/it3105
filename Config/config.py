from dotenv import dotenv_values

"""
File containg all parameters deemed necessary to be easily accessible.
Some parameters in how the program operates are fixed, but the most significant can be changed here. 
"""

config = {
    "GAME": "hex",  # 'hex', 'nim': Game to be played
    "UI_ON": True,  # bool: Toggles GUI for Hex, verbose for NIM
}

game_configs = {
    "hex": {
        "BOARD_SIZE": 7,  # int: Specifies the board size in Hex
        "ANIMATION_SPEED": 0.1,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON does not override)
        "DISPLAY_INDEX": True,  # bool: Specifies if the GUI should display indexes (useful for human play)
    },
    "nim": {
        "STONES": 20,  # int: Specifies number of stones in NIM
        "MIN_STONES": 1,  # int: Specifies the min number of stones you must take each turn
        "MAX_STONES": 3,  # int: Specifies the max number of stones you can take each turn (unless there are fewer stones left)
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON overrides this if True)
        "DELAY": 0,  # float: delay between moves for some reason? (UNSTABLE USE AT OWN RISK)
    },
}

MCTS_config = {
    "UCT_C": 1.3,  # float: Variable for weighting the Upper Confidence Bound for Tree
    "MAX_TIME": 0.5,  # float: Variable for controlling how much time the algorithm is allowed to spend (seconds) (overwritten by MIN_SIMS)
    "MIN_SIMS": 1000,  # int: How many simulations per move at minimum
    "KEEP_SUBTREE": True,  # bool: Specify if to keep the subtree after update
}
# Helpers
save_hex = (
    f"_hex_{game_configs['hex']['BOARD_SIZE']}x{game_configs['hex']['BOARD_SIZE']}_"
)
save_nim = f"_nim_{game_configs['nim']['STONES']}_{game_configs['nim']['MIN_STONES']}_{game_configs['nim']['MAX_STONES']}_"

RLLearner_config = {
    "EPISODES": 1000,  # int: Specify the number of actual games to run
    "BUFFER_SIZE": 2048,  # int: Specify the size of the replay buffer
    "MINIBATCH_SIZE": 256,  # int: Specify the number of samples to be retrived from the buffer
    "SAVE": True,  # bool: Specify to save nets or not
    "SAVE_INTERVAL": 50,  # int: Save the target policy at each x episodes
    "SAVE_PATH": "Models/ModelCache",  # str: Path to save nets to
    "SAVE_NAME": "LETSGO"
    + (save_hex if config["GAME"] == "hex" else save_nim),  # str: Name for saved models
    "TRAIN_UI": False,  # bool: Specify if UI on while training
    "GREEDY_BIAS": 0.2,  # float: A bias to if the RLLearner should choose greedy or 2,3 best (1 -> Always choose best; -1 -> Never choose best)
}

ANET_config = {
    ##################################################
    # Must be tuple of tuples
    "HIDDEN_LAYERS": (
        (128, 64, 64),
        (64, 64, 64, 64, 64),
    )  # tuple: [0] Dense layers; [1] Conv2D layers
    if config["GAME"] == "hex"
    else ((128, 64, 32),),  # tuple: Size of hidden layers
    ##################################################
    "BUILD_CONV": True,  # bool: To build a convolutional model or "normal". Built specifically for Hex (or any square two player board game)
    "EPSILON": 1,  # float: Variable for choosing a random move compared to the greedy best move
    "EPSILON_DECAY": 0.99,  # float: Variable for choosing how fast epsilon should decay
    "MIN_EPSILON": 0.1,  # float: minimum for epsilon
    "LEARNING_RATE": 0.001,  # float: Learning rate (None: Default learning rate)
    # After EPISODES_BEFORE_LR_RED: LR -> LR * (LR_SCALE_FACTOR ** EpNr)
    "EPISODES_BEFORE_LR_RED": 50,  # int: Number of episodes before LR is scaled with a factor
    "LR_SCALE_FACTOR": 0.998,  # float: Factor to scale LR with (1 -> keep the same) #0.993 for α = 0.005, 0.998 for α = 0.001
    "MIN_LR": 0.0001,  # float: Minimum LR
    "ACTIVATION": "relu",  # str: relu, tanh, sigmoid, selu
    "OPTIMIZER": "Adam",  # str: SGD, Adagrad, Adam, RMSprop
    "LOSS_FUNC": "categorical_crossentropy",  # str: categorical_crossentropy, kl_divergence, mse
    "EPOCHS": 5,  # int: Epochs to run each fit
    "REGULARIZATION_CONST": 0.001,  # float: L2 Reg const in CNN
    "EARLY_STOP_PAT": 3,  # int: Early stopping patience
    "BATCH_SIZE": 32,  # int: Number of batches to use in fitting
    "LOAD_PATH": "Models/Stored",  # str: Folder to load models from
    ##########################################################################
    "MODIFY_STATE": (
        config["GAME"]
        in [
            "hex",  # Add other games here if nn should modify state
        ]
    ),  # bool: Specify if we should change state representation from 2 to -1
    ##########################################################################
    "GAME": config["GAME"],  # str: Retrieving game for different ANET inits
    ##########################################################################
}

TOPP_config = {
    "LOAD_PATH": "Models/TOPP",  # str: Folder to load models from
    "GAMES": 24,  # int: Games to play against each other player (Even number)
    "PLOT_STATS": True,  # bool: To plot the final stats or not
    "TOPP_UI": False,  # bool: Toggle UI during TOPP
    "PROBABILISTIC": True,  # bool: If always choosing the best move or not
    "GREEDY_BIAS": 0.3,  # float: Bias towards choosing the best move in probabilistic reasoning (0.3 -> Always choose best if >70% certain)
}

secrets = dotenv_values("Config/secret.env")
OHT_config = {
    "GAME": "hex",  # str: Override config
    "BOARD_SIZE": 7,  # int: Override game_config for hex
    "LOAD_PATH": "Models/OHT",  # str: Where to load model from
    "MODEL_NAME": "LETSGO_hex_7x7_8",  # str: Name of model to load
    "EPSILON": 0,  # float: Epsilon, if we for some reason wanted to choose random moves
    "UI_ON": False,  # bool: If UI should be on
    ################
    ##   DANGER   ##
    ################
    "AUTH": secrets["AUTH"],  # str: Certificate
    "QUALIFY": False,  # bool: To qualify or not (2 attempts left)
    "MODE": "league",  # str: qualifiers, league
}
