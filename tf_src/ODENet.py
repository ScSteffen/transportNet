import tensorflow as tf
from tensorflow import keras


class FullDLRANet(keras.Model):

    def __init__(self, input_dim=1, output_dim=1, name="partDLRANet", tol=0.4, low_rank=20, rmax_total=100, **kwargs):
        super(FullDLRANet, self).__init__(name=name, **kwargs)
        dlra_layer_dim = 250
        self.denseBlock = DenseBlock(units=250, input_dim=input_dim)
        self.dlraBlock1 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock2 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )
        self.dlraBlock3 = DLRALayer(input_dim=dlra_layer_dim, units=dlra_layer_dim, low_rank=low_rank, epsAdapt=tol,
                                    rmax_total=rmax_total, )

        self.outputBlock = DenseBlockOutputSmall(output_dim=output_dim)

    def call(self, inputs, step: int):
        z = self.denseBlock(inputs)
        z = self.dlraBlock1(z, step=step)
        z = self.dlraBlock2(z, step=step)
        z = self.dlraBlock3(z, step=step)
        z = self.outputBlock(z)
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

    @staticmethod
    def set_dlra_bias_grads_to_zero(grads):
        """
        :param grads: gradients of current tape
        :return: sets the nonexistent gradients to zero (i K step, the grads of S and L step are None, which throws annoying warnings)
        """
        for i in range(len(grads)):
            if len(grads[i].shape) == 1:
                grads[i] = tf.math.scalar_mul(0.0, grads[i])
        return 0

    def toggle_non_s_step_training(self):
        self.layers[0].trainable = False  # Dense input
        self.layers[-1].trainable = False  # Dense output

        return 0

    def toggle_s_step_training(self):
        self.layers[0].trainable = True  # Dense input
        self.layers[-1].trainable = True  # Dense output
        return 0
