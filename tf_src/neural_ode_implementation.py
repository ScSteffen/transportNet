"""
Author: Steffen Schotthöfer
Date: 29.03.2022
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


def main():
    print(" --- Starting model construction ---")

    model = create_transport_resnet(input_dim=1, output_dim=1, x_steps=10, t_steps=5)

    print("--- Start model training ---")

    x = np.linspace(-5.0, 5.0, 5000)
    y = np.sin(x)

    model.fit(x=x[::2], y=y[::2], batch_size=64, epochs=1000, verbose=2)
    model.save("model")

    pred = model(x)

    plt.plot(x, y, '-')
    plt.plot(x, pred, '--')
    plt.savefig("result.png")

    return 0


def create_transport_resnet(input_dim, output_dim, x_steps, t_steps):
    # Weight initializer
    initializer = keras.initializers.LecunNormal()
    # Weight regularizer
    l2_regularizer = tf.keras.regularizers.L2(l2=0.0001)  # L1 + L2 penalties

    def residual_block(x: tf.Tensor, layer_dim: int = 10, layer_idx: int = 0) -> tf.Tensor:
        # ResNet architecture by https://arxiv.org/abs/1603.05027

        # 1) activation
        y = keras.activations.softplus(x)
        # 2) layer without activation
        y = keras.layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                               bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                               bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_0")(y)
        # 3) add skip connection
        out = keras.layers.Add()([x, y])
        return out

    input_ = keras.Input(shape=(input_dim,))
    hidden = input_
    # build resnet blocks
    for idx in range(0, t_steps):
        hidden = residual_block(hidden, layer_dim=x_steps, layer_idx=idx)

    output_ = keras.layers.Dense(output_dim, activation=None, kernel_initializer=initializer, name="dense_output",
                                 kernel_regularizer=l2_regularizer, bias_initializer='zeros')(hidden)

    core_model = keras.Model(inputs=[input_], outputs=[output_], name="TransportNet")
    print("The core model overview")
    core_model.summary()
    # build graph
    batch_size: int = 3  # dummy entry
    core_model.build(input_shape=(batch_size, input_dim))

    core_model.compile(
        loss={'dense_output': tf.keras.losses.MeanSquaredError()},
        loss_weights={'dense_output': 1}, optimizer=keras.optimizers.Adam(),
        metrics=['mean_absolute_error', 'mean_squared_error'])

    return core_model


if __name__ == '__main__':
    main()
