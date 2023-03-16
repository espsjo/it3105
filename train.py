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


def train_main(load: bool = False) -> None:
    """
    Main function which starts the training process.
    As default, it will create a new model and start training. Can however load an old model, but unpolished due to not being used.
    Parameters:
        load: (bool) To load a model or not
    Returns:
        None
    """
    r = RLLearner(config, game_configs, MCTS_config, RLLearner_config, ANET_config)
    if load:
        r.load_model(
            model_name="7x7/LETSGO/LETSGO_hex_7x7_7",
            epsilon=0.1,
            epnr=350,
            RBUF_MIN_SIZE=0,
        )
    r.train()


if __name__ == "__main__":
    train_main(load=False)
