from RLLearner.rllearner import RLLearner
from Config.config import (
    config,
    game_configs,
    MCTS_config,
    RLLearner_config,
    ANET_config,
)

"""
File for initializing the training of a new ANET, saved at given intervals in config
"""

r = RLLearner(config, game_configs, MCTS_config, RLLearner_config, ANET_config)
r.train()
