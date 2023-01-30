config = {
    "GAME": "nim",  # 'hex', 'nim': Game to be played
    "UI_ON": True,  # True, False: Toggles GUI for Hex, verbose for NIM
}

hex_config = {
    "BOARD_SIZE": 5,  # int: Specifies the board size in Hex
    "ANIMATION_SPEED": 0.25,  # float: Specifies the min_speed of moves in GUI. Can be slower due to machine processing
    "WON_MSG": False,  # True, False: Specifies if the winning player should be printed to console (UI_ON does not override)
}

nim_config = {
    "STONES": 15,  # int: Specifies number of stones in NIM
    "MIN_STONES": 1,  # int: Specifies the min number of stones you must take each turn
    "MAX_STONES": 4,  # int: Specifies the max number of stones you can take each turn (unless there are fewer stones left)
    "WON_MSG": True,  # True, False: Specifies if the winning player should be printed to console (UI_ON overrides this if True)
    "DELAY": 0,  # float, delay between moves for some reason?
}
