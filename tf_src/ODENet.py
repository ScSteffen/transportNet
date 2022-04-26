import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras


class LinODENet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="partDLRANet", **kwargs):
        super(LinODENet, self).__init__(name=name, **kwargs)
        self.odeBlock = LinearOdeBlock(input_dim=input_dim)

    def call(self, inputs, t_0, t_f):
        z = self.odeBlock(inputs=inputs,t_0=t_0,t_f=t_f)
        return z

class NonLinODENet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="partDLRANet", **kwargs):
        super(LinODENet, self).__init__(name=name, **kwargs)
        self.odeBlock = NonLinearOdeBlock(input_dim=input_dim)

    def call(self, inputs, t_0, t_f):
        z = self.odeBlock(inputs=inputs,t_0=t_0,t_f=t_f)
        return z

class LinearOdeBlock(keras.layers.Layer):
    # self.layer1 = Linear(units=units)

    def __init__(self, input_dim=32):
        super(LinearOdeBlock, self).__init__()
        # Right now, only quadratic matrices
        self.A = self.add_weight(shape=(input_dim, input_dim), initializer="random_normal", name="_A", trainable=True)
        self.b = self.add_weight(shape=(input_dim,), initializer="zeros", name="_b", trainable=True)

    def call(self, inputs, t_0, t_f):
        z = tfp.math.ode.BDF().solve(self.ode_fn, t_0, inputs, solution_times=[t_f],
                                     constants={'A': self.A, 'b': self.b})
        return z

    def ode_fn(self, t, y, A, b):
        # Right Hand side of the Ode that defines the network
        return tf.linalg.matvec(A, y) + b
    
class NonLinearOdeBlock(keras.layers.Layer):
    # self.layer1 = Linear(units=units)

    def __init__(self, input_dim=32):
        super(NonLinearOdeBlock, self).__init__()
        # Right now, only quadratic matrices
        self.A = self.add_weight(shape=(input_dim, input_dim), initializer="random_normal", name="_A", trainable=True)
        self.b = self.add_weight(shape=(input_dim,), initializer="zeros", name="_b", trainable=True)

    def call(self, inputs, t_0, t_f):
        z = tfp.math.ode.BDF().solve(self.ode_fn, t_0, inputs, solution_times=[t_f],
                                     constants={'A': self.A, 'b': self.b})
        return z

    def ode_fn(self, t, y, A, b):
        # Right Hand side of the Ode that defines the network
        return tf.keras.activations.relu(tf.linalg.matvec(A, y) + b)


def create_lotka_volterra_data(n_data: int, n_time: int, final_time: float) -> np.ndarray:
    """
    :param n_data: number of different initial conditions
           n_time: number of time steps of the solution
           final_time: final time of computed system
    :return: time series data of the systems dynamics: dims=(2 x n_time x n_data)
    """
    a = 2.0 / 3.0
    b = 4.0 / 3.0
    c = 1.0
    d = 1.0

    t_0 = 0.0
    t_f = final_time

    times = np.linspace(t_0, t_f, n_time).tolist()
    if n_time == 1:
        times = [t_f]

    def lotka_volterra_ode(t, x):
        res = np.asarray([a * x[0] - b * x[0] * x[1], c * x[0] * x[1] - d * x[1]])
        z = tf.constant(res, dtype=x.dtype, shape=x.shape)
        return z

    data = np.zeros(shape=(n_data, n_time, 2))
    tf.random.set_seed(5);

    for i_data in range(n_data):
        x_0 = tf.random.uniform(shape=[2, ], minval=0.5, maxval=5, dtype=tf.float64, seed=1)
        # x_0 = tf.constant([1., 1.], dtype=tf.float64)

        results = tfp.math.ode.BDF().solve(lotka_volterra_ode, t_0, x_0, solution_times=times)
        data[i_data, :, :] = results.states

    return data


def create_pendulum_data(n_data: int, n_time: int, final_time: float) -> np.ndarray:
    """
    :param n_data: number of different initial conditions
           n_time: number of time steps of the solution
           final_time: final time of computed system
    :return: time series data of the systems dynamics: dims=(2 x n_time x n_data)
    """
    a = 2.0 / 3.0
    b = 4.0 / 3.0
    c = 1.0
    d = 1.0

    t_0 = 0.0
    t_f = final_time

    times = np.linspace(t_0, t_f, n_time).tolist()
    if n_time == 1:
        times = [t_f]

    A = tf.constant([[-1., -2.], [-3., -4.]], dtype=tf.float64)

    def oscillator_ode(t, x):
        return tf.linalg.matvec(A, x)


    data = np.zeros(shape=(n_data, n_time, 2))
    tf.random.set_seed(5);

    for i_data in range(n_data):
        x_0 = tf.random.uniform(shape=[2, ], minval=0.5, maxval=5, dtype=tf.float64, seed=1)
        # x_0 = tf.constant([1., 1.], dtype=tf.float64)

        results = tfp.math.ode.BDF().solve(lotka_volterra_ode, t_0, x_0, solution_times=times)
        data[i_data, :, :] = results.states

    return data
