config = {
    "GAME": "hex",  # 'hex', 'nim': Game to be played
    "UI_ON": True,  # bool: Toggles GUI for Hex, verbose for NIM
}

game_configs = {
    "hex": {
        "BOARD_SIZE": 5,  # int: Specifies the board size in Hex
        "ANIMATION_SPEED": 0.1,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON does not override)
        "DISPLAY_INDEX": True,  # bool: Specifies if the GUI should display indexes (useful for human play)
    },
    "nim": {
        "STONES": 10,  # int: Specifies number of stones in NIM
        "MIN_STONES": 1,  # int: Specifies the min number of stones you must take each turn
        "MAX_STONES": 5,  # int: Specifies the max number of stones you can take each turn (unless there are fewer stones left)
        "WON_MSG": False,  # bool: Specifies if the winning player should be printed to console (UI_ON overrides this if True)
        "DELAY": 0.2,  # float, delay between moves for some reason? (UNSTABLE USE AT OWN RISK)
    },
}

MCTS_config = {
    "UCT_C": 1,  # float: Variable for weighting the Upper Confidence Bound for Tree
    "MAX_TIME": 3,  # float: Variable for controlling how much time the algorithm is allowed to spend (seconds) (overwritten by MIN_SIMS)
    "MIN_SIMS": 100,  # int: How many simulations per move at minimum
    "KEEP_SUBTREE": True,  # bool: Specify if to keep the subtree after update
}

RLLearner_config = {
    "EPISODES": 200,  # int: Specify the number of actual games to run
    "BUFFER_SIZE": 2000,  # int: Specify the size of the replay buffer
    "MINIBATCH_SIZE": 300,  # int: Specify the number of samples to be retrived from the buffer
    "SAVE": True,  # bool: Specify to save nets or not
    "SAVE_INTERVAL": 50,  # int: Save the target policy at each x episodes
    "SAVE_PATH": "Models",  # str: Path to save nets to
    "SAVE_NAME": "Test5x5" + "_" + config["GAME"] + "_",  # str: Name for saved models
}

ANET_config = {
    "EPSILON": 0.99,  # float: Variable for choosing a random move compared to the greedy best move
    "EPSILON_DECAY": 0.99,  # float: Variable for choosing how fast epsilon should decay
    "MIN_EPSILON": 0.2,  # float: minimum for epsilon
    "LEARNING_RATE": None,  # float: Learning rate (None: Default learning rate <-- Please use)
    "HIDDEN_LAYERS": (256, 128, 128),  # tuple: Size of hidden layers
    "ACTIVATION": "relu",  # str: relu, tanh, sigmoid
    "OPTIMIZER": "Adam",  # str: SGD, Adagrad, Adam, RMSprop
    "LOSS_FUNC": "kl_divergence",  # str: categorical_crossentropy, kl_divergence, mse
    "EPOCHS": 8,  # int: Epochs to run each fit
    "BATCH_SIZE": 64,  # int: Number of batches to use in fitting
    "LOAD_PATH": RLLearner_config["SAVE_PATH"],  # str: Folder to load models from
}

TOPP_config = {}
