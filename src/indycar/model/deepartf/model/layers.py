import tensorflow as tf
from keras import backend as K
from keras.initializers import glorot_normal
from keras.layers import Layer


class GaussianLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        n_weight_rows = input_shape[-1]
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        #output_mu = K.dot(x, self.kernel_1) + self.bias_1
        #output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_mu = tf.matmul(x, self.kernel_1) + self.bias_1
        output_sig = tf.matmul(x, self.kernel_2) + self.bias_2


        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        """
        The assumption is the output ts is always one-dimensional
        """
        return [(input_shape[0], input_shape[1],self.output_dim), 
                (input_shape[0], input_shape[1],self.output_dim)]
        #return [(input_shape[0], self.output_dim), 
        #        (input_shape[0], self.output_dim)]


#
# studentT
#
class StudentTLayer(Layer):
    """
    mu
        Tensor containing the means, of shape `(*batch_shape, *event_shape)`.
    sigma
        Tensor containing the standard deviations, of shape
        `(*batch_shape, *event_shape)`.
    nu
        Nonnegative tensor containing the degrees of freedom of the distribution,
        of shape `(*batch_shape, *event_shape)`.
 
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.num_parameters = 3
        self.kernels = [[] for x in range(3)]
        self.biases = [[] for x in range(3)]

        super(StudentTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input: shape (NTC)
        """
        n_weight_rows = input_shape[2]
    
        for i in range(self.num_parameters):
            self.kernels[i] = self.add_weight(name='kernel_%d'%i,
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        for i in range(self.num_parameters):
            self.biases[i] = self.add_weight(name='bias_%d'%i,
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(StudentTLayer, self).build(input_shape)

    def call(self, x):
        """
        return mu, sigma, nu
        """
        output_mu = tf.matmul(x, self.kernels[0]) + self.biases[0]
        output_sig = tf.matmul(x, self.kernels[1]) + self.biases[1]
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        output_nu = tf.matmul(x, self.kernels[2]) + self.biases[2]
        output_nu_pos = K.log(1 + K.exp(output_nu)) + 1e-06 + 2.0

        return [output_mu, output_sig_pos, output_nu_pos]

    def compute_output_shape(self, input_shape):
        """
        The assumption is the output ts is always one-dimensional
        """
        return [(input_shape[0], input_shape[1], self.output_dim),
                (input_shape[0], input_shape[1], self.output_dim),
                (input_shape[0], input_shape[1], self.output_dim)]
