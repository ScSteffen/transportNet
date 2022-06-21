import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime


def main():
    def hide_spines(intx=False, inty=False, logscale=0):
        """Hides the top and rightmost axis spines from view for all active
        figures and their respective axes."""

        # Retrieve a list of all current figures.
        figures = [x for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        if (plt.gca().get_legend()):
            plt.setp(plt.gca().get_legend().get_texts(), fontproperties=font)
        for figure in figures:
            # Get all Axis instances related to the figure.
            for ax in figure.canvas.figure.get_axes():
                # Disable spines.
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                # Disable ticks.
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
                for label in ax.get_xticklabels():
                    label.set_fontproperties(font)
                for label in ax.get_yminorticklabels():
                    label.set_fontproperties(font)
                for label in ax.get_ymajorticklabels():
                    label.set_fontproperties(font)
                # ax.set_xticklabels(ax.get_xticks(), fontproperties = font)
                ax.set_xlabel(ax.get_xlabel(), fontproperties=font)
                ax.set_ylabel(ax.get_ylabel(), fontproperties=font)
                ax.set_title(ax.get_title(), fontproperties=font)
                if (inty):
                    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
                if (intx):
                    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
                if (logscale):
                    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: ("10$^{%d}$" % (math.log(v, 10)))))

    def show(nm, a=0, b=0, logscale=0, facecolor=0):
        hide_spines(a, b, logscale)
        # ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v,_: ("10$^{%d}$" % math.log(v,10)) ))
        # plt.yticks([1,1e-2,1e-4,1e-6,1e-8,1e-10,1e-12], labels)
        if (type(facecolor) == str):
            plt.savefig(nm, facecolor=facecolor)
        else:
            plt.savefig(nm);
        plt.show()

    # Setup output and datasets
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    print(test_labels.shape)
    x_train = train_images  # /255
    y_train = train_labels
    x_valid = test_images  # /255
    y_valid = test_labels
    w, h = 28, 28
    x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
    x_train = x_train.reshape(x_train.shape[0], w, h, 1)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    print(x_train.shape)
    print(y_valid.shape, y_train.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(120000)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    # train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(5000)  # .shuffle(10000)
    valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    # valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    valid_dataset = valid_dataset.repeat()

    # Define network blocks
    def trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3):
        densev1_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')
        ldt = tf.keras.layers.Lambda(lambda x: x * dt)
        diag_lam = tf.keras.layers.Lambda(lambda x: x / (1 + dt / epsilon))
        off_lam = tf.keras.layers.Lambda(lambda x: dt * x / epsilon / (1 + dt / epsilon))
        tfact = tf.keras.layers.Activation(activation="relu")
        tfa = tf.keras.layers.Add()
        tfm = tf.keras.layers.Subtract()
        for step in range(steps):
            v1 = tfa([v1, ldt(densev1_1(v2))])
            tp = tf.keras.layers.Permute(dims=(2, 1, 3))(v1)
            v2 = tfm([v2, ldt(densev1_1(tp))])
            v2 = diag_lam(v2)
            v_o = tfact(v1)
            v_o = off_lam(v_o)
            v2 = tfa([v2, v_o])
        return v1, v2

    def res_block(dt, steps, v1, filters, kernel_size=3):
        densev1_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                                           padding='same', activation="relu")
        ldt = tf.keras.layers.Lambda(lambda x: x * dt)
        diag_lam = tf.keras.layers.Lambda(lambda x: x / (1 + dt / epsilon))
        tfa = tf.keras.layers.Add()
        for step in range(steps):
            v1 = tfa([v1, ldt(densev1_1(v1))])
        return v1

    def baseline_model(x):
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same', activation="relu")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)
        x = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(x))
        outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
        return outputs

    if False:
        # create network
        inputs = keras.Input(shape=(28, 28, 1,), name='MNIST')

        output_base = baseline_model(inputs)

        baseline = keras.Model(inputs, output_base)
        baseline.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                         # Loss function to minimize
                         loss=keras.losses.CategoricalCrossentropy(),
                         # List of metrics to monitor
                         metrics=['accuracy'])
        baseline.summary()

        # Train baseline
        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(
                log_dir='./log/base_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                write_images=True),
            keras.callbacks.ModelCheckpoint(filepath='./log/baseline_fashion.hdf5', verbose=1, save_best_only=True)]

        history = baseline.fit(train_dataset,
                               epochs=5, steps_per_epoch=120000 // 64,
                               # We pass some validation for
                               # monitoring validation loss and metrics
                               # at the end of each epoch
                               validation_data=(valid_dataset), validation_steps=4,
                               callbacks=callbacks)

    if False:
        init_split = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same')(inputs)
        v1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)

        # first trans block
        epsilon = 0.01
        steps = 5
        dt = 1.0 / steps
        filters = 64
        v1 = res_block(dt, steps, v1, filters, kernel_size=3)

        # pool, conv to 32 filters and normalize
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)

        # second trans block
        filters = 32
        v1 = res_block(dt, steps, v1, filters, kernel_size=3)

        # pool and flatten
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(v1))
        dense_final = tf.keras.layers.Dense(10, activation="softmax")
        outputs = dense_final(v1)

        model_res = keras.Model(inputs=inputs, outputs=outputs)

        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(
                log_dir='./log/res_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                write_images=True),
            keras.callbacks.ModelCheckpoint(filepath='./log/resnet_fashion.hdf5', verbose=1, save_best_only=True)]

        model_res.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                          # Loss function to minimize
                          loss=keras.losses.CategoricalCrossentropy(),
                          # List of metrics to monitor
                          metrics=['accuracy'])
        model_res.summary()
        history = model_res.fit(train_dataset,
                                epochs=5, steps_per_epoch=120000 // 64,
                                # We pass some validation for
                                # monitoring validation loss and metrics
                                # at the end of each epoch
                                validation_data=(valid_dataset), validation_steps=4,
                                callbacks=callbacks)

        print('\nhistory dict:', history.history)

    if True:
        inputs = keras.Input(shape=(28, 28, 1,), name='MNIST')

        init_split = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same')(inputs)
        v1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)
        v2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)

        # first trans block
        epsilon = 0.1
        steps = 5
        dt = 1.0 / steps
        filters = 64
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool, conv to 32 filters and normalize
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)
        v2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v2)
        v2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v2)
        v2 = tf.keras.layers.BatchNormalization()(v2)

        # second trans block
        filters = 32
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool and flatten
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(v1))
        dense_final = tf.keras.layers.Dense(10, activation="softmax")
        outputs = dense_final(v1)

        model01 = keras.Model(inputs=inputs, outputs=outputs)
        model01.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                        # Loss function to minimize
                        loss=keras.losses.CategoricalCrossentropy(),
                        # List of metrics to monitor
                        metrics=['accuracy'])
        model01.summary()
        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(
                log_dir='./log/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
            keras.callbacks.ModelCheckpoint(filepath='./log/steps5e01_fashion.hdf5', verbose=1, save_best_only=True)]
        history = model01.fit(train_dataset,
                              epochs=5, steps_per_epoch=120000 // 64,
                              # We pass some validation for
                              # monitoring validation loss and metrics
                              # at the end of each epoch
                              validation_data=(valid_dataset), validation_steps=4,
                              callbacks=callbacks)

        print('\nhistory dict:', history.history)

    if False:
        init_split = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same')(inputs)
        v1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)
        v2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)

        # first trans block
        epsilon = 0.001
        steps = 5
        dt = 1.0 / steps
        filters = 64
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool, conv to 32 filters and normalize
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)
        v2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v2)
        v2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v2)
        v2 = tf.keras.layers.BatchNormalization()(v2)

        # second trans block
        filters = 32
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool and flatten
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(v1))
        dense_final = tf.keras.layers.Dense(10, activation="softmax")
        outputs = dense_final(v1)

        model0p001 = keras.Model(inputs=inputs, outputs=outputs)
        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(
                log_dir='./log/steps5e0p001_fashion_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                write_images=True),
            keras.callbacks.ModelCheckpoint(filepath='./log/steps5e0p001_fashion.hdf5', verbose=1, save_best_only=True)]

        model0p001.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                           # Loss function to minimize
                           loss=keras.losses.CategoricalCrossentropy(),
                           # List of metrics to monitor
                           metrics=['accuracy'])
        model0p001.summary()
        history = model0p001.fit(train_dataset,
                                 epochs=5, steps_per_epoch=120000 // 64,
                                 # We pass some validation for
                                 # monitoring validation loss and metrics
                                 # at the end of each epoch
                                 validation_data=(valid_dataset), validation_steps=4,
                                 callbacks=callbacks)

        print('\nhistory dict:', history.history)

    if False:
        # 10 steps res
        init_split = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same')(inputs)
        v1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)

        # first trans block
        steps = 10
        dt = 1.0 / steps
        filters = 64
        v1 = res_block(dt, steps, v1, filters, kernel_size=3)

        # pool, conv to 32 filters and normalize
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)

        # second trans block
        filters = 32
        v1 = res_block(dt, steps, v1, filters, kernel_size=3)

        # pool and flatten
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(v1))
        dense_final = tf.keras.layers.Dense(10, activation="softmax")
        outputs = dense_final(v1)

        model_res_10 = keras.Model(inputs=inputs, outputs=outputs)

        callbacks = [
            # Write TensorBoard logs to `./logs` directory
            keras.callbacks.TensorBoard(
                log_dir='./log/res_10_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
                write_images=True),
            keras.callbacks.ModelCheckpoint(filepath='./log/resnet_fashion_10.hdf5', verbose=1, save_best_only=True)]

        model_res_10.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                             # Loss function to minimize
                             loss=keras.losses.CategoricalCrossentropy(),
                             # List of metrics to monitor
                             metrics=['accuracy'])
        model_res_10.summary()
        history = model_res_10.fit(train_dataset,
                                   epochs=5, steps_per_epoch=120000 // 64,
                                   # We pass some validation for
                                   # monitoring validation loss and metrics
                                   # at the end of each epoch
                                   validation_data=(valid_dataset), validation_steps=4,
                                   callbacks=callbacks)

        print('\nhistory dict:', history.history)
        init_split = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=(1, 1), padding='same')(inputs)
        v1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)
        v2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(init_split)

        # first trans block
        epsilon = 0.01
        steps = 10
        dt = 1.0 / steps
        filters = 64
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool, conv to 32 filters and normalize
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v1)
        v1 = tf.keras.layers.BatchNormalization()(v1)
        v2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v2)
        v2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same')(v2)
        v2 = tf.keras.layers.BatchNormalization()(v2)

        # second trans block
        filters = 32
        v1, v2 = trans_block(dt, epsilon, steps, v1, v2, filters, kernel_size=3)

        # pool and flatten
        v1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(v1)
        v1 = tf.keras.layers.Flatten()(tf.keras.layers.Dropout(0.01)(v1))
        dense_final = tf.keras.layers.Dense(10, activation="softmax")
        outputs = dense_final(v1)

        model_10 = keras.Model(inputs=inputs, outputs=outputs)
    return 0


if __name__ == '__main__':
    main()
