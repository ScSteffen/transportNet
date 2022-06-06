import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import path, makedirs


class RelaxationLayerImplicit(keras.layers.Layer):
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

    def __init__(self, input_dim=32, units=32, batch_size=32, epsilon=0.1, name="RelaxationLayerImplicit", **kwargs):
        super(RelaxationLayerImplicit, self).__init__(**kwargs)

        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.B0 = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.q = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.activation = tf.keras.activations.tanh

        # For implicit system solve
        self.sys_matrix = self.add_weight(shape=(2 * input_dim, 2 * units), initializer="zeros", trainable=True)
        self.rhs = self.add_weight(shape=(2 * input_dim, batch_size), initializer="zeros", trainable=False)

    def assemble_sys_mat(self):
        """
        assembles the system matrix from the weight matrix
        :return: Void
        """
        self.sys_matrix.assign(tf.zeros((2 * self.units, 2 * self.units)))
        self.sys_matrix[:self.units, self.units:].assign(self.B0)
        self.sys_matrix[self.units:, :self.units].assign(-tf.transpose(self.B0))

        return 0

    # @tf.function
    def call(self, v0, v1) -> (tf.Tensor, tf.Tensor):
        # create rhs
        self.rhs[:self.units, :].assign(v0 + self.q)
        self.rhs[self.units:, :].assign(v1)

        v_np1 = tf.linalg.solve(self.sys_matrix, self.rhs)

        return v_np1[:self.units, :], v_np1[self.units:, :]


class LinearLayer(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, name="linear", **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.units = units
        self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal",
                                 trainable=True, )
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(LinearLayer, self).get_config()
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


class TransNetImplicit(keras.Model):
    input_dim: int
    output_dim: int
    units: int
    epsilon: float
    batch_size: int
    num_layers: int

    linearInput: LinearLayer
    linearOutput: LinearLayer
    relaxLayers: list  # [RelaxationLayer]

    def __init__(self, num_layers=4, input_dim=784, units=32, output_dim=10, batch_size=32, epsilon=0.1,
                 name="transNet", **kwargs):
        super(TransNetImplicit, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.epsilon = epsilon
        self.output_dim = output_dim
        self.units = units
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.linearInput = LinearLayer(input_dim=self.input_dim, units=self.units)
        self.linearOutput = LinearLayer(input_dim=self.units, units=self.output_dim)

        self.relaxLayers = []
        for i in range(num_layers):
            self.relaxLayers.append(
                RelaxationLayerImplicit(input_dim=self.units, units=self.units, batch_size=batch_size,
                                        epsilon=self.epsilon))

    # @tf.function
    def call(self, inputs):

        v_0 = self.linearInput(inputs)

        v_0 = tf.transpose(v_0)
        v_1 = self.relaxLayers[0].activation(v_0)

        for i in range(self.num_layers):
            self.relaxLayers[i].assemble_sys_mat()
            v_0, v_1 = self.relaxLayers[i](v_0, v_1)

        v_0 = tf.transpose(v_0)
        z = self.linearOutput(v_0)
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
