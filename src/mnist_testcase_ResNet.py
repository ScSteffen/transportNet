from src.networks.resNet import ResNet, create_csv_logger_cb

import tensorflow as tf
from tensorflow import keras
import numpy as np
from optparse import OptionParser
from os import path, makedirs


def train(num_layers, units, batch_size, load_model, epochs):
    # specify training
    filename = "resNet_" + str(num_layers) + "_" + str(units)
    folder_name = filename + '/latest_model'
    folder_name_best = filename + '/best_model'

    # check if dir exists
    if not path.exists(folder_name):
        makedirs(folder_name)
    if not path.exists(folder_name_best):
        makedirs(folder_name_best)

    print("save model as: " + filename)

    # Create Model
    input_dim = 784  # 28x28  pixel per image
    output_dim = 10  # one-hot vector of digits 0-9

    model = ResNet(num_layers=num_layers, input_dim=input_dim, units=units, output_dim=output_dim,
                   batch_size=batch_size)

    # Build optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # Choose loss
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # Choose metrics (to monitor training, but not to optimize on)
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Accuracy()
    loss_metric_acc_val = tf.keras.metrics.Accuracy()

    # Build dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, input_dim))
    x_test = np.reshape(x_test, (-1, input_dim))

    # Reserve 10,000 samples for validation.
    val_size = 10000
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    (x_val, y_val) = normalize_img(x_val, y_val)

    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]
    (x_train, y_train) = normalize_img(x_train, y_train)

    (x_test, y_test) = normalize_img(x_test, y_test)
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Create logger
    log_file, file_name = create_csv_logger_cb(folder_name=filename)

    # print headline
    log_string = "loss_train;acc_train;loss_val;acc_val;loss_test;acc_test\n"
    with open(file_name, "a") as log:
        log.write(log_string)

    # load weights
    if load_model == 1:
        model.load(folder_name=folder_name)

    best_acc = 0
    best_loss = 10

    # initialize model state
    for step, batch_train in enumerate(train_dataset):
        # model.initialize(batch_train[0])
        break

    # Iterate over epochs. (Training loop)
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch,))
        # Iterate over the batches of the dataset.

        for step, batch_train in enumerate(train_dataset):

            if batch_train[0].shape[0] != batch_size:

                bx = batch_train[0]
                by = batch_train[1]

                z = bx.shape[0]
                while z < batch_size:
                    bx = tf.concat([bx, bx], axis=0)
                    by = tf.concat([by, by], axis=0)

                    z = bx.shape[0]
                batch_train = (bx[:batch_size, :], by[:batch_size])

            # 1)  linear step
            with tf.GradientTape() as tape:
                out = model(batch_train[0], training=True)
                # softmax activation for classification
                out = tf.keras.activations.softmax(out)
                # Compute reconstruction loss
                loss = loss_fn(batch_train[1], out)
                loss += sum(model.losses)  # Add KLD regularization loss
            grads = tape.gradient(loss, model.trainable_weights)
            model.set_none_grads_to_zero(grads, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # 2) Relaxation step
            # model.relax(batch_train[0])

            # Network monotoring and verbosity
            loss_metric.update_state(loss)
            prediction = tf.math.argmax(out, 1)
            acc_metric.update_state(prediction, batch_train[1])

            loss_value = loss_metric.result().numpy()
            acc_value = acc_metric.result().numpy()
            if step % 100 == 0:
                print("step %d: mean loss  = %.4f" % (step, loss_value))
                print("Accuracy: " + str(acc_value))

            # Reset metrics
            loss_metric.reset_state()
            acc_metric.reset_state()

        # Compute vallidation loss and accuracy
        print("---validation---")
        # Validate model
        for step, batch_val in enumerate(val_dataset):
            if batch_val[0].shape[0] != batch_size:

                bx = batch_val[0]
                by = batch_val[1]

                z = bx.shape[0]
                while z < batch_size:
                    bx = tf.concat([bx, bx], axis=0)
                    by = tf.concat([by, by], axis=0)

                    z = bx.shape[0]
                batch_val = (bx[:batch_size, :], by[:batch_size])

            out = model(batch_val[0], training=False)
            out = tf.keras.activations.softmax(out)
            loss = loss_fn(batch_val[1], out)
            loss_metric.update_state(loss)
            prediction = tf.math.argmax(out, 1)
            acc_metric.update_state(prediction, batch_val[1])

        loss_val = loss_metric.result().numpy()
        acc_val = acc_metric.result().numpy()
        print("Val Accuracy: " + str(acc_val))

        # save current model if it's the best
        if acc_val >= best_acc and loss_val <= best_loss:
            best_acc = acc_val
            best_loss = loss_val
            print("new best model with accuracy: " + str(best_acc) + " and loss " + str(best_loss))

            # model.save(folder_name=folder_name_best)
        # model.save(folder_name=folder_name)

        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Test model
        for step, batch_test in enumerate(test_dataset):
            if batch_test[0].shape[0] != batch_size:

                bx = batch_test[0]
                by = batch_test[1]

                z = bx.shape[0]
                while z < batch_size:
                    bx = tf.concat([bx, bx], axis=0)
                    by = tf.concat([by, by], axis=0)

                    z = bx.shape[0]
                batch_test = (bx[:batch_size, :], by[:batch_size])

            out = model(batch_test[0], training=False)
            out = tf.keras.activations.softmax(out)
            loss = loss_fn(batch_test[1], out)
            loss_metric.update_state(loss)
            prediction = tf.math.argmax(out, 1)
            acc_metric.update_state(prediction, batch_test[1])

        loss_test = loss_metric.result().numpy()
        acc_test = acc_metric.result().numpy()
        log_string = "Loss: " + str(loss_test) + "| Accuracy" + str(acc_test) + "\n"
        print("Test :" + log_string)
        # Reset metrics
        loss_metric.reset_state()
        acc_metric.reset_state()

        # Log Data of current epoch
        log_string = str(loss_value) + ";" + str(acc_value) + ";" + str(
            loss_val) + ";" + str(acc_val) + ";" + str(
            loss_test) + ";" + str(acc_test) + ";"
        with open(file_name, "a") as log:
            log.write(log_string)
        print("Epoch Data :" + log_string)

    return 0


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-u", "--units", dest="units", default=200)
    parser.add_option("-x", "--epsilon", dest="epsilon", default=0.1)
    parser.add_option("-l", "--load_model", dest="load_model", default=1)
    parser.add_option("-t", "--train", dest="train", default=1)
    parser.add_option("-b", "--batch_size", dest="batch_size", default=32)
    parser.add_option("-e", "--epochs", dest="epochs", default=100)
    parser.add_option("-n", "--num_layers", dest="num_layers", default=100)

    (options, args) = parser.parse_args()
    options.units = int(options.units)
    options.epsilon = float(options.epsilon)
    options.load_model = int(options.load_model)
    options.train = int(options.train)
    options.batch_size = int(options.batch_size)
    options.epochs = int(options.epochs)
    options.num_layers = int(options.num_layers)

    if options.train == 1:
        train(num_layers=options.num_layers, units=options.units,
              batch_size=options.batch_size, load_model=options.load_model, epochs=options.epochs)
