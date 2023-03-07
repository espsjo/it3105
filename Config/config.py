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
        "BOARD_SIZE": 4,  # int: Specifies the board size in Hex
        "ANIMATION_SPEED": 0.5,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
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
    "MAX_TIME": 1.5,  # float: Variable for controlling how much time the algorithm is allowed to spend (seconds) (overwritten by MIN_SIMS)
    "MIN_SIMS": 1000,  # int: How many simulations per move at minimum
    "KEEP_SUBTREE": True,  # bool: Specify if to keep the subtree after update
}

save_hex = (
    f"_hex_{game_configs['hex']['BOARD_SIZE']}x{game_configs['hex']['BOARD_SIZE']}_"
)
save_nim = f"_nim_{game_configs['nim']['STONES']}_{game_configs['nim']['MIN_STONES']}_{game_configs['nim']['MAX_STONES']}"
RLLearner_config = {
    "EPISODES": 300,  # int: Specify the number of actual games to run
    "BUFFER_SIZE": 512,  # int: Specify the size of the replay buffer
    "MINIBATCH_SIZE": 256,  # int: Specify the number of samples to be retrived from the buffer
    "SAVE": True,  # bool: Specify to save nets or not
    "SAVE_INTERVAL": 50,  # int: Save the target policy at each x episodes
    "SAVE_PATH": "Models/ModelCache",  # str: Path to save nets to
    "SAVE_NAME": "LONG"
    + (save_hex if config["GAME"] == "hex" else save_nim),  # str: Name for saved models
    "TRAIN_UI": False,  # bool: Specify if UI on while training
    "GREEDY_BIAS": 0.1,  # float: A bias to if the RLLearner should choose greedy or 2,3 best (1 -> Always choose best; -1 -> Never choose best)
}

ANET_config = {
    "EPSILON": 1,  # float: Variable for choosing a random move compared to the greedy best move
    "EPSILON_DECAY": 0.99,  # float: Variable for choosing how fast epsilon should decay
    "MIN_EPSILON": 0.1,  # float: minimum for epsilon
    "LEARNING_RATE": 0.001,  # float: Learning rate (None: Default learning rate <-- Please use)
    "HIDDEN_LAYERS": (32, 48, 32)
    if config["GAME"] == "hex"
    else (512, 256, 128, 128),  # tuple: Size of hidden layers
    "ACTIVATION": "relu",  # str: relu, tanh, sigmoid
    "OPTIMIZER": "SGD",  # str: SGD, Adagrad, Adam, RMSprop
    "LOSS_FUNC": "kl_divergence",  # str: categorical_crossentropy, kl_divergence, mse
    "EPOCHS": 15,  # int: Epochs to run each fit
    "BATCH_SIZE": 16,  # int: Number of batches to use in fitting
    "LOAD_PATH": "Models",  # str: Folder to load models from
    "MODIFY_STATE": (
        config["GAME"]
        in [
            "hex",  # Add other games here if nn should modify state
        ]
    ),  # bool: Specify if we should change state representation from 2 to -1
    "TEMPERATURE": None,  # float: A temperature to encode the target values with (None / 1 to do nothing)
}

TOPP_config = {}
