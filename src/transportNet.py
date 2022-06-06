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
        self.q = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.activation = tf.keras.activations.tanh

    @tf.function
    def call(self, v_0_in, v_1_in) -> (tf.Tensor, tf.Tensor):
        v_0_out = v_0_in + tf.matmul(v_1_in, self.B0) + self.q
        v_1_out = v_1_in - tf.matmul(v_0_in, tf.transpose(self.B0)) - 1 / self.epsilon * (
                v_1_in - self.activation(v_0_in))
        return v_0_out, v_1_out


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
        self.sys_matrix = self.add_weight(shape=(2 * input_dim, 2 * units), initializer="random_normal",
                                          trainable=True)
        self.activation = tf.keras.activations.tanh

    def assemble_sys_mat(self):
        """
        assembles the system matrix from the weight matrix
        :return: Void
        """

        tensor = [[1, 1], [1, 1], [1, 1]]  # tf.rank(tensor) == 2
        indices = [[0, 1], [2, 0]]  # num_updates == 2, index_depth == 2
        updates = [5, 10]
        print(tf.tensor_scatter_nd_update(tensor, indices, updates))

        self.sys_matrix

    @tf.function
    def call(self, v_0_in, v_1_in) -> (tf.Tensor, tf.Tensor):
        v_0_out = v_0_in + tf.matmul(v_1_in, self.B0) + self.q
        v_1_out = v_1_in - tf.matmul(v_0_in, tf.transpose(self.B0)) - 1 / self.epsilon * (
                v_1_in - self.activation(v_0_in))
        return v_0_out, v_1_out


class ResNetLayer(keras.layers.Layer):
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
        super(ResNetLayer, self).__init__(**kwargs)

        self.units = units
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.B0 = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
        self.q = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)
        self.activation = tf.keras.activations.tanh

    @tf.function
    def call(self, z_in) -> tf.Tensor:
        z = z_in - tf.matmul(self.activation(z_in), tf.transpose(self.B0)) + self.q
        return z


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


class TransNetExplicit(keras.Model):
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
        super(TransNetExplicit, self).__init__(name=name, **kwargs)

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
            self.relaxLayers.append(RelaxationLayer(input_dim=self.units, units=self.units, batch_size=batch_size,
                                                    epsilon=self.epsilon))

    @tf.function
    def call(self, inputs):
        v_0 = self.linearInput(inputs)
        v_1 = self.relaxLayers[0].activation(v_0)

        for i in range(self.num_layers):
            v_0, v_1 = self.relaxLayers[i](v_0, v_1)

        z = self.linearOutput(v_0)
        return z

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

    @tf.function
    def call(self, inputs):

        v_0 = self.linearInput(inputs)
        v_1 = self.relaxLayers[0].activation(v_0)

        for i in range(self.num_layers):
            v_0, v_1 = self.relaxLayers[i](v_0, v_1)

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


class ResNet(keras.Model):
    input_dim: int
    output_dim: int
    units: int
    batch_size: int
    num_layers: int

    linearInput: LinearLayer
    linearOutput: LinearLayer
    relaxLayers: list  # [RelaxationLayer]

    def __init__(self, num_layers=4, input_dim=784, units=32, output_dim=10, batch_size=32, name="ResNet", **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.linearInput = LinearLayer(input_dim=self.input_dim, units=self.units)
        self.linearOutput = LinearLayer(input_dim=self.units, units=self.output_dim)

        self.relaxLayers = []
        for i in range(num_layers):
            self.relaxLayers.append(ResNetLayer(input_dim=self.units, units=self.units, batch_size=batch_size))

    @tf.function
    def call(self, inputs):
        z = self.linearInput(inputs)

        for i in range(self.num_layers):
            z = self.relaxLayers[i](z)
        z = self.linearOutput(z)
        return z

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
