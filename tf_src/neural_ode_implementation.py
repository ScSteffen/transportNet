"""
Author: Steffen Schotthöfer
Date: 29.03.2022
"""

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

from ODENet import ODENet, create_lotka_volterra_data


def main():
    # Just an ODE system solve

    t_init, t0, t1 = 0., 0.5, 1.
    y_init = tf.constant([1., 1.], dtype=tf.float64)
    A = tf.constant([[-1., -2.], [-3., -4.]], dtype=tf.float64)

    def ode_fn(t, y):
        return tf.linalg.matvec(A, y)

    results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                       solution_times=[t0, t1])
    y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
    y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)

    # Now with learnable weight matrices
    def ode_fn(t, y, A):
        return tf.linalg.matvec(A, y)

    with tf.GradientTape() as tape:
        tape.watch(A)
        results = tfp.math.ode.BDF().solve(ode_fn, t_init, y_init,
                                           solution_times=[t0, t1],
                                           constants={'A': A})
    grads = tape.gradient(results.states, A)  # Fine.

    y0 = results.states[0]  # == dot(matrix_exp(A * t0), y_init)
    y1 = results.states[1]  # == dot(matrix_exp(A * t1), y_init)

    return 0


def main2():
    model = ODENet(input_dim=2)  # Build Model

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    # Create data from a rea dynamical system
    n_data = 10
    n_time = 100
    final_time = 20
    data = create_lotka_volterra_data(n_data=n_data, n_time=n_time, final_time=final_time)
    plt.plot(np.linspace(0, final_time, n_time), data[0, :, :])
    plt.show()

    # Build dataset
    split = 2  # (n_data / 10)
    train_x = data[:n_data - split, 0, :]
    train_y = data[:n_data - split, -1, :]

    test_x = data[- split:, -1, :]
    test_y = data[- split:, -1, :]

    # config training
    epochs = 10
    batch = 1

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch)

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # tape.watch(modA)
                out = model(batch_train[0])
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

    test = model(test_x)
    plt.plot(test_x, test.numpy(), '-.')
    plt.plot(test_x, test_y, '--')
    plt.show()
    return 0


if __name__ == '__main__':
    main2()
