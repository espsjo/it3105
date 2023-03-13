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
        "ANIMATION_SPEED": 0.5,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON does not override)
        "DISPLAY_INDEX": True,  # bool: Specifies if the GUI should display indexes (useful for human play)
    },
    "nim": {
        "STONES": 10,  # int: Specifies number of stones in NIM
        "MIN_STONES": 1,  # int: Specifies the min number of stones you must take each turn
        "MAX_STONES": 4,  # int: Specifies the max number of stones you can take each turn (unless there are fewer stones left)
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON overrides this if True)
        "DELAY": 0,  # float: delay between moves for some reason? (UNSTABLE USE AT OWN RISK)
    },
}

MCTS_config = {
    "UCT_C": 1.3,  # float: Variable for weighting the Upper Confidence Bound for Tree
    "MAX_TIME": 1.5,  # float: Variable for controlling how much time the algorithm is allowed to spend (seconds) (overwritten by MIN_SIMS)
    "MIN_SIMS": 1000,  # int: How many simulations per move at minimum
    "KEEP_SUBTREE": True,  # bool: Specify if to keep the subtree after update
}

save_hex = (
    f"_hex_{game_configs['hex']['BOARD_SIZE']}x{game_configs['hex']['BOARD_SIZE']}_"
)
save_nim = f"_nim_{game_configs['nim']['STONES']}_{game_configs['nim']['MIN_STONES']}_{game_configs['nim']['MAX_STONES']}_"
RLLearner_config = {
    "EPISODES": 2000,  # int: Specify the number of actual games to run
    "BUFFER_SIZE": 1024,  # int: Specify the size of the replay buffer
    "MINIBATCH_SIZE": 256,  # int: Specify the number of samples to be retrived from the buffer
    "SAVE": True,  # bool: Specify to save nets or not
    "SAVE_INTERVAL": 50,  # int: Save the target policy at each x episodes
    "SAVE_PATH": "Models/ModelCache",  # str: Path to save nets to
    "SAVE_NAME": "PRAY"
    + (save_hex if config["GAME"] == "hex" else save_nim),  # str: Name for saved models
    "TRAIN_UI": False,  # bool: Specify if UI on while training
    "GREEDY_BIAS": 0.2,  # float: A bias to if the RLLearner should choose greedy or 2,3 best (1 -> Always choose best; -1 -> Never choose best)
}

ANET_config = {
    "BUILD_CONV": True,  # bool: To build a convolutional model or "normal". Built specifically for Hex (or any square two player board game)
    "EPSILON": 1,  # float: Variable for choosing a random move compared to the greedy best move
    "EPSILON_DECAY": 0.99,  # float: Variable for choosing how fast epsilon should decay
    "MIN_EPSILON": 0.1,  # float: minimum for epsilon
    "LEARNING_RATE": 0.005,  # float: Learning rate (None: Default learning rate)
    "HIDDEN_LAYERS": (
        32,
        64,
        100,
    )  # If BUILD_CONV [-1] is Dense, the rest Conv2D. Else: All Dense
    if config["GAME"] == "hex"
    else (32, 32),  # tuple: Size of hidden layers
    "ACTIVATION": "relu",  # str: relu, tanh, sigmoid, selu
    "OPTIMIZER": "Adam",  # str: SGD, Adagrad, Adam, RMSprop
    "LOSS_FUNC": "categorical_crossentropy",  # str: categorical_crossentropy, kl_divergence, mse
    "EPOCHS": 10,  # int: Epochs to run each fit
    "BATCH_SIZE": 16,  # int: Number of batches to use in fitting
    "LOAD_PATH": "Models/Stored",  # str: Folder to load models from
    "MODIFY_STATE": (
        config["GAME"]
        in [
            "hex",  # Add other games here if nn should modify state
        ]
    ),  # bool: Specify if we should change state representation from 2 to -1
    "TEMPERATURE": None,  # float: A temperature to encode the target values with (None / 1 to do nothing)
    # After EPISODES_BEFORE_LR_RED: LR -> LR * (LR_SCALE_FACTOR ** EpNr)
    "EPISODES_BEFORE_LR_RED": 50,  # int: Number of episodes before LR is scaled with a factor
    "LR_SCALE_FACTOR": 0.993,  # float: Factor to scale LR with (1 -> keep the same)
    "MIN_LR": 0.0001,  # float: Minimum LR
}

TOPP_config = {
    "LOAD_PATH": "Models/TOPP",  # str: Folder to load models from
    "GAMES": 20,  # int: Games to play against each other player
    "PLOT_STATS": True,  # bool: To plot the final stats or not
    "TOPP_UI": False,  # bool: Toggle UI during TOPP
    "PROBABILISTIC": True,  # bool: If always choosing the best move or not
    "GREEDY_BIAS": 0.3,  # float: Bias towards choosing the best move in probabilistic reasoning (0.3 -> Always choose best if >70% certain)
}
