import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras


# ---- Network definitions ---

class ODENet(keras.Model):
    # virtual base class

    odeBlock: tf.Tensor

    def __init__(self, input_dim=1, name="NonLinODENet", **kwargs):
        super(ODENet, self).__init__(name=name, **kwargs)

    def call(self, inputs, t_0, t_f):
        z = self.odeBlock(inputs=inputs, t_0=t_0, t_f=t_f)
        return z

    def predict(self, inputs, t_0, t_f, n_time):
        z = self.odeBlock.predict(
            inputs=inputs, t_0=t_0, t_f=t_f, n_time=n_time)
        return z

    def save(self, folder_name):
        self.odeBlock.save(folder_name)
        return 0

    def load(self, folder_name):
        self.odeBlock.load(folder_name)
        return 0


class NonLinODENet(ODENet):

    def __init__(self, input_dim=1, name="NonLinODENet", **kwargs):
        super(NonLinODENet, self).__init__(name=name, **kwargs)
        self.odeBlock = NonLinearOdeBlock(input_dim=input_dim)


class LinODENet(ODENet):

    def __init__(self, input_dim=1, name="LinODENet", **kwargs):
        super(LinODENet, self).__init__(name=name, **kwargs)
        self.odeBlock = LinearOdeBlock(input_dim=input_dim)


# ---- Layer definitions ----


class ODEBlock(keras.layers.Layer):
    A: tf.Tensor
    b: tf.Tensor

    # virtual base class

    def __init__(self, input_dim=2):
        super(ODEBlock, self).__init__()
        # Right now, only quadratic matrices
        self.A = self.add_weight(shape=(
            input_dim, input_dim), initializer="random_normal", name="_A", trainable=True)
        self.b = self.add_weight(
            shape=(input_dim,), initializer="zeros", name="_b", trainable=True)

    def call(self, inputs, t_0, t_f):
        z = tfp.math.ode.BDF().solve(self.ode_fn, t_0, inputs, solution_times=[t_f],
                                     constants={'A': self.A, 'b': self.b})
        return z

    def predict(self, inputs, t_0, t_f, n_time):
        times = np.linspace(t_0, t_f, n_time).tolist()

        z = tfp.math.ode.BDF().solve(self.ode_fn, t_0, inputs, solution_times=times,
                                     constants={'A': self.A, 'b': self.b})
        return z

    def save(self, folder_name):
        a_np = self.A.numpy()
        # print(a_np)
        np.save(folder_name + "/A.npy", a_np)

        b_np = self.b.numpy()
        np.save(folder_name + "/b.npy", b_np)

        return 0

    def load(self, folder_name):
        a_np = np.load(folder_name + "/A.npy")
        self.A = tf.Variable(initial_value=a_np,
                             trainable=True, name="_A", dtype=tf.float32)
        b_np = np.load(folder_name + "/b.npy")
        self.b = tf.Variable(initial_value=b_np,
                             trainable=True, name="_b", dtype=tf.float32)
        return 0


class LinearOdeBlock(ODEBlock):

    def __init__(self, input_dim=2):
        super(LinearOdeBlock, self).__init__(input_dim=input_dim)

    def ode_fn(self, t, y, A, b):
        # Right Hand side of the Ode that defines the network
        return tf.linalg.matvec(A, y) + b


class NonLinearOdeBlock(ODEBlock):

    def __init__(self, input_dim=2):
        super(NonLinearOdeBlock, self).__init__(input_dim=input_dim)

    def ode_fn(self, t, y, A, b):
        # Right Hand side of the Ode that defines the network
        return tf.keras.activations.tanh(tf.linalg.matvec(A, y) + b)


class NonLinearOdeBlockV2(ODEBlock):

    def __init__(self, input_dim=2):
        super(NonLinearOdeBlockV2, self).__init__(input_dim=input_dim)

    def ode_fn(self, t, y, A, b):
        # Right Hand side of the Ode that defines the network
        return tf.linalg.matvec(A, tf.keras.activations.relu(y)) + b


def create_lotka_volterra_data(n_data: int, n_time: int, final_time: float, deterministic=True) -> np.ndarray:
    """
    :param n_data: number of different initial conditions
           n_time: number of time steps of the solution
           final_time: final time of computed system
    :return: time series data of the systems dynamics: dims=(n_data x n_time x 2)
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
        res = np.asarray([a * x[0] - b * x[0] * x[1],
                          c * x[0] * x[1] - d * x[1]])
        z = tf.constant(res, dtype=x.dtype, shape=x.shape)
        return z

    data = np.zeros(shape=(n_data, n_time, 2))
    tf.random.set_seed(5)

    for i_data in range(n_data):
        if deterministic:
            x_0 = tf.constant([1., 1.], dtype=tf.float64)
        else:
            x_0 = tf.random.uniform(
                shape=[2, ], minval=0.5, maxval=5, dtype=tf.float64, seed=1)

        results = tfp.math.ode.BDF().solve(
            lotka_volterra_ode, t_0, x_0, solution_times=times)
        data[i_data, :, :] = results.states

    return data


def create_pendulum_data(n_data: int, n_time: int, final_time: float) -> np.ndarray:
    """
    :param n_data: number of different initial conditions
           n_time: number of time steps of the solution
           final_time: final time of computed system
    :return: time series data of the systems dynamics: dims=(2 x n_time x n_data)
    """

    t_0 = 0.0
    t_f = final_time

    times = np.linspace(t_0, t_f, n_time).tolist()
    if n_time == 1:
        times = [t_f]

    A = tf.constant([[0., -1.], [-1., 0.]], dtype=tf.float64)

    @tf.function
    def linear_ode(t, x):
        return tf.linalg.matvec(A, x)

    data = np.zeros(shape=(n_data, n_time, 2))
    tf.random.set_seed(5)

    for i_data in range(n_data):
        x_0 = tf.random.uniform(
            shape=[2, ], minval=0.5, maxval=5, dtype=tf.float64, seed=1)
        # x_0 = tf.constant([1., 1.], dtype=tf.float64)

        results = tfp.math.ode.BDF().solve(linear_ode, t_0, x_0, solution_times=times)
        data[i_data, :, :] = results.states

    return data
