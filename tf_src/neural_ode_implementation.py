"""
Author: Steffen Schotthöfer
Date: 29.03.2022
"""

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import matplotlib.pyplot as plt

from ODENet import LinODENet, NonLinODENet, create_lotka_volterra_data, create_pendulum_data


def main():

    train = True
    test = False
    # Create Model
    model = NonLinODENet(input_dim=2)
    #model = NonLinearOdeNet(input_dim=2)
    save_name = 'NonLinearODENet/lotka_volterra'

    if train:
        # Load Model
        #model.load(save_name)

        # Create data from a rea dynamical system
        n_data = 1
        n_time = 20
        final_time = 10
        data = create_lotka_volterra_data(
            n_data=n_data, n_time=n_time, final_time=final_time, deterministic=True)

        #data = create_pendulum_data(
        #    n_data=n_data, n_time=n_time, final_time=final_time)

        # Train Model
        training_loop(model, data, final_time, n_time,save_name)

        # Test Model
        test_model(
            model, ic=data[0, 0, :], time_data=data[0, :, :], t_f=final_time, n_time=n_time)

    if test:
        # Load Model
        model.load('linearODENet/lotka_volterra')
        # Create data from a rea dynamical system
        n_data = 1
        n_time = 200
        final_time = 5
        data = create_lotka_volterra_data(
            n_data=n_data, n_time=n_time, final_time=final_time,deterministic=True)
        # Evaluate model
        test_model(
            model, ic=data[0, 0, :], time_data=data[0, :, :], t_f=final_time, n_time=n_time)
    # train_nonlinear_ODENet()
    return 0


def training_loop(model, data, t_f, n_time, save_name):

    # config training
    epochs = 100
    batch = 1
    t_0 = 0.0
    dt = t_f/float(n_time)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    train_x = data[0, :-1, :]
    train_y = data[0, 1:, :]  # label is the solution at the next time step

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    # Iterate over epochs.
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # tape.watch(modA)
                results = model(
                    inputs=batch_train[0], t_0=step*dt, t_f=(step+1)*dt)
                x_f = results.states[0]
                # Compute reconstruction loss
                loss = mse_loss_fn(batch_train[1], x_f)
                loss += sum(model.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 10 == 0:
                print("Training at t=" + str(step*dt) + " with x(t)=" +
                      str(batch_train[0].numpy()) + " and x(t+1)=" + str(batch_train[1].numpy()))
                print("Prediction at this step: " + str(x_f.numpy()))
                print("step %d: mean loss = %.4f" %
                      (step, loss_metric.result()))
             # Save Model
            model.save(save_name)

    return 0

    # extract model weights:
    weights = [model.tr]


def test_model(model, ic, time_data, t_f, n_time):
    print(time_data.shape)

    t = np.linspace(0, t_f, n_time)
    results = model.predict(inputs=tf.constant(
        ic, dtype=tf.float32),  t_0=0, t_f=t_f, n_time=n_time)

    x_time_series = results.states.numpy()
    print(x_time_series.shape)

    plt.plot(t, x_time_series, '-.')
    plt.plot(t, time_data, '--')
    plt.savefig("result.png")
    return 0


if __name__ == '__main__':
    main()
