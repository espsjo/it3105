import tensorflow as tf
from tensorflow import keras as ks
import numpy as np


class Network:
    """
    Method for building a standard sequential model. Can be used with NIM or other games as well.
    Parameters:
        ANET_config: (dict) Config containing anet params
    """

    def __init__(
        self,
        ANET_config,
    ) -> None:
        self.LEARNING_RATE = ANET_config["LEARNING_RATE"]
        self.HIDDEN_LAYERS = ANET_config["HIDDEN_LAYERS"]
        self.ACTIVATION = ANET_config["ACTIVATION"]
        self.OPTIMIZER = ANET_config["OPTIMIZER"]
        self.LOSS_FUNC = ANET_config["LOSS_FUNC"]
        self.REGULARIZATION_CONST = ANET_config["REGULARIZATION_CONST"]

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

    def build_normal(self, in_size, out_size) -> ks.models.Sequential:
        """
        Method for building a standard sequential model. Can be used with NIM or other games as well.
        Parameters:
            in_size: (int) The board_size**2+1
            out_size: (int) The number of moves (board_size**2)
        Returns:
            ks.models.Sequential
        """

        model = ks.models.Sequential()
        hidden_dense = self.HIDDEN_LAYERS[0]

        model.add(
            ks.layers.Input(
                shape=(in_size,),
            )
        )
        for i in range(len(hidden_dense)):
            dim = hidden_dense[i]
            model.add(
                ks.layers.Dense(
                    dim,
                    activation=self.ACTIVATION,
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                )
            )

        model.add(
            ks.layers.Dense(
                out_size,
                activation="softmax",
            )
        )

        model.compile(
            optimizer=self.optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=[tf.keras.metrics.categorical_accuracy],
        )
        self.model = model
        self.model.summary()

        return model

    def build_conv_model(self, in_size, out_size) -> ks.models.Sequential:
        """
        Method for building a sequential convolutional network
        Parameters:
            in_size: (int) The board_size
            out_size: (int) The number of moves (board_size**2)
        Returns:
            ks.models.Sequential
        """
        activ = eval("ks.activations." + self.ACTIVATION)
        hidden_dense, hidden_conv = self.HIDDEN_LAYERS

        ##############################
        #                            #
        #   Based on AlphaGo paper   #
        #                            #
        ##############################
        model = ks.models.Sequential()
        model.add(
            ks.layers.Input(
                shape=(in_size, in_size, 5),
            )
        )

        model.add(
            ks.layers.Conv2D(
                hidden_conv[0],
                kernel_size=(5, 5),
                padding="same",
                kernel_regularizer=ks.regularizers.l2(self.REGULARIZATION_CONST),
            )
        )
        model.add(ks.layers.BatchNormalization(axis=1))
        model.add(ks.layers.Activation(activ))

        for i in range(1, len(hidden_conv)):
            units = hidden_conv[i]
            model.add(
                ks.layers.Conv2D(
                    units,
                    kernel_size=(3, 3),
                    padding="same",
                    kernel_regularizer=ks.regularizers.l2(self.REGULARIZATION_CONST),
                )
            )
            model.add(ks.layers.BatchNormalization(axis=1))
            model.add(ks.layers.Activation(activ))

        model.add(
            ks.layers.Conv2D(
                1,
                kernel_size=(1, 1),
                # kernel_regularizer=ks.regularizers.l2(self.REGULARIZATION_CONST),
            )
        )
        model.add(ks.layers.Flatten())
        model.add(ks.layers.Softmax())

        model.compile(
            optimizer=self.optimizers[self.OPTIMIZER],
            loss=self.LOSS_FUNC,
            metrics=[tf.keras.metrics.categorical_accuracy, "KLDivergence"],
        )
        self.model = model
        self.model.summary()

        return model
