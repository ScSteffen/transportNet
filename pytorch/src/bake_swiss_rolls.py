import numpy as np
import matplotlib.pyplot as plt


def create_dataset(num_samples: int = 1000, test_rate=0.1, plotting=False, shuffle=True):
    """
    :brief creates a dataset, with 50-50 class balance

    :param num_samples: size of the complete dataset
    :param validation_rate: ratio of validation data
    :param test_rate:  ratio of test data
    :return: [[x_train,y_train],[x_val,y_val],[x_test,y_test]]
    """

    def f1(x):
        """
        :param r_theta: r_theta[:,0] = r_s,r_theta[:,1] = thetas_s,
        :return:
        """
        z = np.zeros(shape=(num_samples, 3))
        label = np.zeros(shape=(num_samples, 1))

        z[:, 0] = x * np.cos(4 * np.pi * x)
        z[:, 1] = x * np.sin(4 * np.pi * x)
        z[:, 2] = x * 0.0

        return z, label

    def f2(x):
        """
        :param r_theta: r_theta[:,0] = r_s,r_theta[:,1] = thetas_s,
        :return:
        """
        z = np.zeros(shape=(num_samples, 3))
        label = np.ones(shape=(num_samples, 1))
        z[:, 0] = (x + 0.2) * np.cos(4 * np.pi * x)
        z[:, 1] = (x + 0.2) * np.sin(4 * np.pi * x)
        z[:, 2] = x * 0.0
        return z, label

    x = np.linspace(0, 1, num_samples)

    inner_roll, inner_labels = f1(x)
    outer_roll, outer_labels = f2(x)

    if plotting:
        plt.plot(inner_roll[:, 0], inner_roll[:, 1], "r")
        plt.plot(outer_roll[:, 0], outer_roll[:, 1], "k")
        plt.show()
    data = np.concatenate([inner_roll, outer_roll], axis=0)
    labels = np.concatenate([inner_labels, outer_labels], axis=0)

    if shuffle:
        shuffler = np.random.permutation(2 * num_samples)
        data = data[shuffler]
        labels = labels[shuffler]

    return (data[int(test_rate * 2 * num_samples):], labels[int(test_rate * 2 * num_samples):]), (
        data[:int(test_rate * 2 * num_samples)], labels[:int(test_rate * 2 * num_samples)])
