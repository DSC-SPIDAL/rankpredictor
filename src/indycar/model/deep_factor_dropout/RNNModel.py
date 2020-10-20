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

# Third-party imports
from mxnet.gluon import HybridBlock, nn

import mxnet as mx

import numpy as np
# First-party imports
from gluonts.block.rnn import RNN
from gluonts.core.component import validated


class RNNModel(HybridBlock):
    @validated()
    def __init__(
        self,
        mode,
        num_hidden,
        num_layers,
        num_output,
        bidirectional=False,
        dropout_rate = 0.1,
        context_length = 60,
        **kwargs,
    ):
        super(RNNModel, self).__init__(**kwargs)
        self.num_output = num_output
        self.context_length = context_length

        RnnCell = mx.gluon.rnn.LSTMCell

        with self.name_scope():
            #self.rnn = RNN(
            #    mode=mode,
            #    num_hidden=num_hidden,
            #    num_layers=num_layers,
            #    bidirectional=bidirectional,
            #)
            self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
            for k in range(num_layers):
                cell = RnnCell(hidden_size=num_hidden)
                cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
                cell = (
                    mx.gluon.rnn.ZoneoutCell(cell, zoneout_states=dropout_rate)
                    if dropout_rate > 0.0
                    else cell
                )
                self.rnn.add(cell)
            self.rnn.cast(dtype=np.float32)
 
            self.decoder = nn.Dense(
                num_output, in_units=num_hidden, flatten=False
            )

    def hybrid_forward(self, F, inputs):
        outputs, state = self.rnn.unroll(
            inputs=inputs,
            #length=subsequences_length,
            length= self.context_length,
            layout="NTC",
            merge_outputs=True,
            begin_state=self.rnn.begin_state(
                func=F.zeros,
                #dtype=self.dtype,
                batch_size=inputs.shape[0]
                if isinstance(inputs, mx.nd.NDArray)
                else 0,
            ),
        )

       
        return self.decoder(outputs)
        #return self.decoder(self.rnn(inputs))
