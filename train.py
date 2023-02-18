from RLLearner.rllearner import RLLearner
from Config.config import (
    config,
    game_configs,
    MCTS_config,
    RLLearner_config,
    ANET_config,
)

r = RLLearner(config, game_configs, MCTS_config, RLLearner_config, ANET_config)
r.train()
