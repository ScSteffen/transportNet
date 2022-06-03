import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import path, makedirs


class RelaxationLayer(keras.layers.Layer):
    units: int
    input_dim: int
    batch_size: int
    epsilon: float

    B0: tf.Tensor
    B1: tf.Tensor
    q: tf.Tensor
    activation: tf.keras.activations.relu

    v0_n: tf.Tensor
    v1_n: tf.Tensor

    def __init__(self, input_dim=32, units=32, batch_size=32, epsilon=0.1, name="RelaxationLayer", **kwargs):
        super(RelaxationLayer, self).__init__(**kwargs)

        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.B0 = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        # self.B1 = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.q = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.activation = tf.keras.activations.tanh

        self.v0_n = self.add_weight(shape=(self.batch_size, self.units), initializer="zeros", trainable=False)
        self.v1_n = self.add_weight(shape=(self.batch_size, self.units), initializer="zeros", trainable=False)

    def initialize(self, v0_n) -> bool:
        # call to initialize state of layer
        self.v0_n.assign(v0_n)
        self.v1_n.assign(self.activation(v0_n))
        return True

    @tf.function
    def call(self, z_in) -> tf.Tensor:
        self.v0_n.assign(z_in)
        z = z_in + tf.matmul(self.v1_n, self.B0) + self.q
        return z

    @tf.function
    def relax(self, z_in) -> tf.Tensor:
        self.v1_n.assign(z_in)
        z = z_in - tf.matmul(z_in, tf.transpose(self.B0)) - 1 / self.epsilon * (
                z_in - self.activation(self.v0_n))
        return z


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name="linear", **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

    def save(self, folder_name, layer_id):
        w_np = self.w.numpy()
        np.save(folder_name + "/w_" + str(layer_id) + ".npy", w_np)
        b_np = self.b.numpy()
        np.save(folder_name + "/b_" + str(layer_id) + ".npy", b_np)
        return 0

    def load(self, folder_name, layer_id):
        a_np = np.load(folder_name + "/w_" + str(layer_id) + ".npy")
        self.w = tf.Variable(initial_value=a_np,
                             trainable=True, name="w_", dtype=tf.float32)
        b_np = np.load(folder_name + "/b_" + str(layer_id) + ".npy")
        self.b = tf.Variable(initial_value=b_np,
                             trainable=True, name="b_", dtype=tf.float32)


class TransNet(keras.Model):
    # v1_ic: tf.Variable
    input_dim: int
    output_dim: int
    units: int
    epsilon: float
    batch_size: int

    linearInput: Linear
    relaxBlock1: RelaxationLayer
    relaxBlock2: RelaxationLayer
    relaxBlock3: RelaxationLayer
    relaxBlock4: RelaxationLayer
    linearOutput: Linear

    def __init__(self, input_dim=784, units=32, output_dim=10, batch_size=32, epsilon=0.1, name="transNet", **kwargs):
        super(TransNet, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.output_dim = output_dim
        self.units = units
        self.batch_size = batch_size

        self.linearInput = Linear(input_dim=self.input_dim, units=self.units)
        self.relaxBlock1 = RelaxationLayer(input_dim=self.units, units=self.units, batch_size=batch_size,
                                           epsilon=self.epsilon)
        self.relaxBlock2 = RelaxationLayer(input_dim=self.units, units=self.units, batch_size=batch_size,
                                           epsilon=self.epsilon)
        self.relaxBlock3 = RelaxationLayer(input_dim=self.units, units=self.units, batch_size=batch_size,
                                           epsilon=self.epsilon)
        self.relaxBlock4 = RelaxationLayer(input_dim=units, units=self.units, batch_size=batch_size,
                                           epsilon=self.epsilon)
        self.linearOutput = Linear(input_dim=self.units, units=self.output_dim)

    def initialize(self, inputs):
        z = self.linearInput(inputs)

        # self.v1_ic = self.relaxBlock1.activation(z)

        self.relaxBlock1.initialize(z)
        z = self.relaxBlock1(z)
        self.relaxBlock2.initialize(z)
        z = self.relaxBlock2(z)
        self.relaxBlock3.initialize(z)
        z = self.relaxBlock3(z)
        self.relaxBlock4.initialize(z)

        return True

    @tf.function
    def call(self, inputs):
        z = self.linearInput(inputs)
        z = self.relaxBlock1(z)
        z = self.relaxBlock2(z)
        z = self.relaxBlock3(z)
        z = self.relaxBlock4(z)
        z = self.linearOutput(z)
        return z

    @tf.function
    def relax(self, inputs):
        z = self.linearInput(inputs)
        z = self.relaxBlock1.activation(z)  # This is the IC for the relaxation
        z = self.relaxBlock1.relax(z)
        z = self.relaxBlock2.relax(z)
        z = self.relaxBlock3.relax(z)
        z = self.relaxBlock4.relax(z)
        return z

    def deactivate_linear_layers(self):
        self.layers[0].trainable = False  # Dense input
        self.layers[-1].trainable = False  # Dense output

    def activate_linear_layers(self):
        self.layers[0].trainable = True  # Dense input
        self.layers[-1].trainable = True  # Dense output

    @staticmethod
    def set_none_grads_to_zero(grads, weights):
        """
        :param grads: gradients of current tape
        :param weights: weights of current model
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if grads[i] is None:
                grads[i] = tf.zeros(shape=weights[i].shape, dtype=tf.float32)
        return 0


# ------ utils below

def create_csv_logger_cb(folder_name: str):
    '''
    dynamically creates a csvlogger and tensorboard logger
    '''
    # check if dir exists
    if not path.exists(folder_name + '/historyLogs/'):
        makedirs(folder_name + '/historyLogs/')

    # checkfirst, if history file exists.
    logName = folder_name + '/historyLogs/history_001_'
    count = 1
    while path.isfile(logName + '.csv'):
        count += 1
        logName = folder_name + \
                  '/historyLogs/history_' + str(count).zfill(3) + '_'

    logFileName = logName + '.csv'
    # create logger callback
    f = open(logFileName, "a")

    return f, logFileName
