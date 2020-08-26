#from deepar.model import NNModel
#from deepar.model.layers import GaussianLayer
#from deepar.model.loss import gaussian_likelihood
from . import NNModel
from .layers import GaussianLayer, StudentTLayer
from .loss import gaussian_likelihood, gaussian_sampler
from .loss import studentt_likelihood, studentt_sampler

from tensorflow.keras.layers import Input, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
import logging
import numpy as np

logger = logging.getLogger('deepar')

_debug_itercnt = 0
_debug_profile_start = 5

class DeepAR(NNModel):
    def __init__(self, ts_obj, steps_per_epoch=50, epochs=100, 
           distribution = 'Gaussian',
           optimizer='adam', with_custom_nn_structure=None,
           use_generator=False):

        self.ts_obj = ts_obj
        self.inputs, self.z_sample = None, None
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.optimizer = optimizer
        self.keras_model = None
    
        self.use_generator = use_generator

        if distribution == 'Gaussian':
            self.loss = gaussian_likelihood
            self.distrib = GaussianLayer
            self.sampler = gaussian_sampler
        elif distribution == 'StudentT':
            self.loss = studentt_likelihood
            self.distrib = StudentTLayer
            self.sampler = studentt_sampler
        else:
            pass

 


        if with_custom_nn_structure:
            self.nn_structure = with_custom_nn_structure
        else:
            self.nn_structure = DeepAR.basic_structure
        self._output_layer_name = 'main_output'
        self.get_intermediate = None

    @staticmethod
    def basic_structure(**kwargs):
        """
        This is the method that needs to be patched when changing NN structure
        :return: inputs_shape (tuple), inputs (Tensor), [loc, scale] (a list of theta parameters
        of the target likelihood)
        """
        context_len = kwargs['context_len']
        prediction_len = kwargs['prediction_len']
        input_dim = kwargs['input_dim']
        distrib = kwargs['distrib']

        seqlen = context_len + prediction_len
        input_shape = (seqlen, input_dim)
 
        inputs = Input(shape=input_shape)
        x = LSTM(4, return_sequences=True)(inputs)
        #x = Dense(3, activation='relu')(x)
        #loc, scale = GaussianLayer(1, name='main_output')(x)
        theta = distrib(1, name='main_output')(x)
        return input_shape, inputs, theta

    def instantiate_and_fit(self, verbose=False):
        print('instantiate_and_fit(self, verbose=False)')

        input_shape, inputs, theta = self.nn_structure()
        model = Model(inputs, theta[0])
        model.compile(loss=self.loss(theta[1:]), optimizer=self.optimizer,
                experimental_run_tf_function = False
                )
        #model.fit_generator(ts_generator(self.ts_obj,
        model.fit(ts_generator(self.ts_obj,
                                         input_shape[0]),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs)
        if verbose:
            logger.debug('Model was successfully trained')
        self.keras_model = model
        self.get_intermediate = K.function(inputs=[self.model.input],
                                           outputs=self.model.get_layer(self._output_layer_name).output)

    @property
    def model(self):
        return self.keras_model

    def predict_theta_from_input(self, input_list):
        """
        This function takes an input of size equal to the n_steps specified in 'Input' when building the
        network
        :param input_list:
        :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
        corresponding to [[mu_values], [sigma_values]]
        """
        if not self.get_intermediate:
            raise ValueError('TF model must be trained first!')

        return self.get_intermediate(input_list)



    @staticmethod
    def encoder_decoder(**kwargs):
        #context_len=40, prediction_len=2, input_dim=1,
        #    num_cells = 40, num_layers = 2, dropout_rate = 0.1):
        """
        follow the deepar 2-layers lstm encoder-decoder
        This is the method that needs to be patched when changing NN structure
        :return: inputs_shape (tuple), inputs (Tensor), [loc, scale] (a list of theta parameters
        of the target likelihood)
        """
        context_len = kwargs['context_len']
        prediction_len = kwargs['prediction_len']
        input_dim = kwargs['input_dim']
        num_cells = kwargs['num_cells']
        num_layers = kwargs['num_layers']
        dropout_rate = kwargs['dropout_rate']
        distrib = kwargs['distrib']
 
        seqlen = context_len + prediction_len

        input_shape = (seqlen, input_dim)
        inputs = Input(shape=input_shape)

        x = inputs
        for l in range(num_layers):
            x = LSTM(num_cells, return_sequences=True, dropout=dropout_rate, unroll=True)(x)
            #x = LSTM(num_cells, return_sequences=True, dropout=dropout_rate, unroll=False)(x)

        #x = Dense(3, activation='relu')(x)
        #loc, scale = GaussianLayer(1, name='main_output')(x)
        #return input_shape, inputs, [loc, scale]
        theta = distrib(1, name='main_output')(x)
        #theta = StudentTLayer(1, name='main_output')(x)
        return input_shape, inputs, theta


    def fit(self, verbose=False, 
            context_len=20, prediction_len=2, input_dim=1,
            num_cells = 40, num_layers = 2, dropout_rate = 0.1, 
            batch_size = 32,
            callbacks=None):

        input_shape, inputs, theta = self.nn_structure(
             distrib = self.distrib,
             context_len=context_len, prediction_len=prediction_len, input_dim=input_dim,
             num_cells = num_cells, num_layers = num_layers, dropout_rate = dropout_rate)

        #model = Model(inputs, theta[0])
        model = Model(inputs, theta)
        model.compile(loss=self.loss(), optimizer=self.optimizer)
                #experimental_run_tf_function = True)
                #experimental_run_tf_function = False)

        model.summary()

        _debug_itercnt = 0
        #model.fit_generator(ts_generator(self.ts_obj,
        if self.use_generator:
            model.fit(ts_generator(self.ts_obj,
                                         input_shape[0]),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            callbacks=callbacks)
        else:

            datax = np.empty((0, self.ts_obj.n_steps, self.ts_obj.n_features))
            datay = np.empty((0, self.ts_obj.n_steps, 1))
            curbatch = 0
            while curbatch < self.ts_obj.n_batches:
                x,y = self.ts_obj.next_batch(32, input_shape[0])
                datax = np.append(datax, x, axis=0)
                datay = np.append(datay, y, axis=0)
                curbatch = curbatch + 1


            model.fit(datax, datay,
                      batch_size = batch_size,
                      steps_per_epoch=self.steps_per_epoch,
                      epochs=self.epochs,
                      callbacks=callbacks)

        if verbose:
            logger.debug('Model was successfully trained')
        self.keras_model = model
        self.get_intermediate = K.function(inputs=[self.model.input],
                                           outputs=self.model.get_layer(self._output_layer_name).output)


    def predict(self, input_list):
        """
        This function takes an input of size equal to the n_steps specified in 'Input' when building the
        network
        :param input_list:
        :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
        corresponding to [[mu_values], [sigma_values]]
        """
        if not self.get_intermediate:
            raise ValueError('TF model must be trained first!')

        return self.get_intermediate(input_list)

    def get_sample(self, theta):
        return self.sampler(theta)

def ts_generator(ts_obj, n_steps):
    """
    This is a util generator function for Keras
    :param ts_obj: a Dataset child class object that implements the 'next_batch' method
    :param n_steps: parameter that specifies the length of the net's input tensor
    :return:
    """
    global _debug_itercnt, _debug_profile_start
    while 1:
        #batch = ts_obj.next_batch(1, n_steps)
        #batch = ts_obj.next_batch(32, n_steps)
        batch = ts_obj.next_batch(32, n_steps)

        #if _debug_itercnt == _debug_profile_start:
        #    with open('./vtune-flag.txt','w') as flagf:
        #        flagf.write('hi')

        _debug_itercnt += 1

        yield batch[0], batch[1]
