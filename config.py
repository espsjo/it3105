config = {
    "GAME": "hex",  # 'hex', 'nim': Game to be played
    "UI_ON": False,  # True, False: Toggles GUI for Hex, verbose for NIM
}

game_configs = {
    "hex": {
        "BOARD_SIZE": 6,  # int: Specifies the board size in Hex
        "ANIMATION_SPEED": 0.1,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
        "WON_MSG": False,  # True, False: Specifies if the winning player should be printed to console (UI_ON does not override)
        "DISPLAY_INDEX": True,  # True, False: Specifies if the GUI should display indexes (useful for human play)
    },
    "nim": {
        "STONES": 15,  # int: Specifies number of stones in NIM
        "MIN_STONES": 1,  # int: Specifies the min number of stones you must take each turn
        "MAX_STONES": 4,  # int: Specifies the max number of stones you can take each turn (unless there are fewer stones left)
        "WON_MSG": True,  # True, False: Specifies if the winning player should be printed to console (UI_ON overrides this if True)
        "DELAY": 0,  # float, delay between moves for some reason?
    },
}

MCTS_config = {
    "EPSILON": 0.5,  # Variable for choosing a random move compared to the greedy best move
    "UCT_C": 1,  # Variable for weighting the Upper Confidence Bound for Tree
    # Can use either time or num rollouts, but can also be combined
    "MAX_TIME": 0,  # Variable for controlling how much time the algorithm is allowed to spend (seconds)
    "MAX_SIMS": 0,  # How many simulations per move
}
