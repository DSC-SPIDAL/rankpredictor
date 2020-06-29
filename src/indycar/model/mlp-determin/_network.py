# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import List

# Third-party imports
import mxnet as mx
from mxnet.gluon import loss as gloss

# First-party imports
from gluonts.block.scaler import MeanScaler, NOPScaler
from gluonts.core.component import validated
from gluonts.distribution import Distribution, DistributionOutput
from gluonts.model.common import Tensor


class DMLPNetworkBase(mx.gluon.HybridBlock):
    """
    Abstract base class to implement feed-forward networks for probabilistic
    time series prediction.

    This class does not implement hybrid_forward: this is delegated
    to the two subclasses DMLPTrainingNetwork and
    DMLPPredictionNetwork, that define respectively how to
    compute the loss and how to generate predictions.

    Parameters
    ----------
    num_hidden_dimensions
        Number of hidden nodes in each layer.
    prediction_length
        Number of time units to predict.
    context_length
        Number of time units that condition the predictions.
    batch_normalization
        Whether to use batch normalization.
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse.
    distr_output
        Distribution to fit.
    kwargs
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        mean_scaling: bool,
        dropout: float,
        #distr_output: DistributionOutput,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        #self.distr_output = distr_output

        self.loss = gloss.L2Loss() 

        with self.name_scope():
            #self.distr_args_proj = self.distr_output.get_args_proj()
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims[:-1]):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation="relu"))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
                #dropout
                self.mlp.add(mx.gluon.nn.Dropout(dropout))

            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * dims[-1]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, dims[-1])
                    )
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

    #def get_distr(self, F, feat: Tensor, target: Tensor) -> Distribution:
    #def get_distr(self, F, feat: Tensor) -> Distribution:
    #    """
    #    Given past target values, applies the feed-forward network and
    #    maps the output to a probability distribution for future observations.

    #    Parameters
    #    ----------
    #    F
    #    target
    #        Tensor containing past target observations.
    #        Shape: (batch_size, context_length, target_dim).

    #    Returns
    #    -------
    #    Distribution
    #        The predicted probability distribution for future observations.
    #    """

    #    # (batch_size, seq_len, target_dim) and (batch_size, seq_len, target_dim)
    #    #scaled_target, target_scale = self.scaler(
    #    #    past_target,
    #    #    F.ones_like(past_target),  # TODO: pass the actual observed here
    #    #)
    #    target_scale = F.ones_like(feat).mean(axis=1)

    #    mlp_outputs = self.mlp(feat)
    #    distr_args = self.distr_args_proj(mlp_outputs)
    #    return self.distr_output.distribution(
    #        distr_args, scale=target_scale.expand_dims(axis=1)
    #    )

    def get_output(self, F, feat: Tensor) -> Tensor:
        """
        """
        target_scale = F.ones_like(feat).mean(axis=1)

        mlp_outputs = self.mlp(feat)

        #distr_args = self.distr_args_proj(mlp_outputs)
        #noret = self.distr_output.distribution(
        #    distr_args, scale=target_scale.expand_dims(axis=1)
        #)

        return mlp_outputs


class DMLPTrainingNetwork(DMLPNetworkBase):
    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, target: Tensor, feat: Tensor
    ) -> Tensor:
        """
        Computes a probability distribution for future data given the past,
        and returns the loss associated with the actual future observations.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, prediction_length, target_dim).

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size, ).
        """
        #distr = self.get_distr(F, feat)

        ## (batch_size, prediction_length, target_dim)
        #loss = distr.loss(target)

        ## (batch_size, )
        #return loss.mean(axis=1)

        output = self.get_output(F, feat)
        # (batch_size, prediction_length, target_dim)
        l = self.loss(target, output)
        # (batch_size, )
        return l


class DMLPPredictionNetwork(DMLPNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(self, F, feat: Tensor) -> Tensor:
        """
        Computes a probability distribution for future data given the past,
        and draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).

        Returns
        -------
        Tensor
            Prediction sample. Shape: (batch_size, samples, prediction_length).
        """
        #distr = self.get_distr(F, feat)

        ## (num_samples, batch_size, prediction_length)
        #samples = distr.sample(self.num_parallel_samples)

        ## (batch_size, num_samples, prediction_length)
        #return samples.swapaxes(0, 1)


        # (batch_size, prediction_length, target_dim)
        output = self.get_output(F, feat)
        return output

