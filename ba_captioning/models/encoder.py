import keras.backend as K
from keras.layers import InputSpec
from keras import activations, initializations, regularizers
from keras.layers.recurrent import Recurrent
import theano.tensor as T
import theano
import numpy as np
from theano import gof, tensor
from theano.gof import Apply

class StochasticStep(gof.Op):
    nin = 1
    nout = 1
    __props__ = ()

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        sm = (np.random.uniform(low=.0, high=1.0, size=x.shape) > (1/(1+np.exp(-x)))).astype(x.dtype)
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        y = T.nnet.sigmoid(x)
        return [g_sm * y * (1 - y)]

    def infer_shape(self, node, shape):
        return shape

class Step(gof.Op):
    nin = 1
    nout = 1
    __props__ = ()

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got %s' %
                             x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)

        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        sm = (.5 > (1 / (1 + np.exp(-x)))).astype(x.dtype)
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        y = T.nnet.sigmoid(x)
        return [g_sm * y * (1 - y)]

    def infer_shape(self, node, shape):
        return shape

stochastic_switch = StochasticStep()
switch = Step()

def rnn(step_function, inputs, initial_states,
        go_backwards=False, mask=None, constants=None,
        unroll=False, input_length=None):

    ndim = inputs.ndim
    assert ndim >= 3, 'Input should be at least 3D.'

    if unroll:
        if input_length is None:
            raise Exception('When specifying `unroll=True`, an `input_length` '
                            'must be provided to `rnn`.')

    axes = [1, 0] + list(range(2, ndim))
    inputs = inputs.dimshuffle(axes)

    if constants is None:
        constants = []

    if mask is not None:
        if mask.ndim == ndim - 1:
            mask = K.expand_dims(mask)
        assert mask.ndim == ndim
        mask = mask.dimshuffle(axes)

        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, new_states = step_function(inputs[i], states + constants)

                if len(successive_outputs) == 0:
                    prev_output = K.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = T.switch(mask[i], output, prev_output)
                kept_states = []
                for state, new_state in zip(states, new_states):
                    kept_states.append(T.switch(mask[i], new_state, state))
                states = kept_states

                successive_outputs.append(output)
                successive_states.append(states)

            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))
        else:
            # build an all-zero tensor of shape (samples, output_dim)
            initial_output = step_function(inputs[0], initial_states + constants)[0] * 0
            # Theano gets confused by broadcasting patterns in the scan op
            initial_output = T.unbroadcast(initial_output, 0, 1)


            def _step(input, mask, output_tm1, *states):
                output, new_states = step_function(input, states)
                # output previous output if masked.
                output = T.switch(mask, output, output_tm1)
                return_states = []
                for state, new_state in zip(states, new_states):
                    return_states.append(T.switch(mask, new_state, state))
                return [output] + return_states


            results, _ = theano.scan(
                _step,
                sequences=[inputs, mask],
                outputs_info=[initial_output] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []
    else:
        if unroll:
            indices = list(range(input_length))
            if go_backwards:
                indices = indices[::-1]

            successive_outputs = []
            successive_states = []
            states = initial_states
            for i in indices:
                output, states = step_function(inputs[i], states + constants)
                successive_outputs.append(output)
                successive_states.append(states)
            outputs = T.stack(*successive_outputs)
            states = []
            for i in range(len(successive_states[-1])):
                states.append(T.stack(*[states_at_step[i] for states_at_step in successive_states]))

        else:
            def _step(input, *states):
                output, new_states = step_function(input, states)
                return [output] + new_states


            results, _ = theano.scan(
                _step,
                sequences=inputs,
                outputs_info=[None] + initial_states,
                non_sequences=constants,
                go_backwards=go_backwards)

            # deal with Theano API inconsistency
            if type(results) is list:
                outputs = results[0]
                states = results[1:]
            else:
                outputs = results
                states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    axes = [1, 0] + list(range(2, outputs.ndim))
    outputs = outputs.dimshuffle(axes)
    states = [T.squeeze(state.dimshuffle(axes)) for state in states]
    return last_output, outputs, states


class BA_LSTM(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 first = False, disable_shots = False, dropout_W = .0, dropout_U = .0,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.first = first
        self.disable_shots = disable_shots
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.uses_learning_phase = True

        self.input_dim = None
        self.input_spec = [InputSpec(ndim=3)]

        super(BA_LSTM, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.output_dim+1)
        else:
            return (input_shape[0], self.output_dim)

    def build(self, input_shape):
        self.T = input_shape[1]
        if self.first:
            self.input_dim = input_shape[2]
        else:
            self.input_dim = input_shape[2] - 1

        self.input_spec = [InputSpec(shape=input_shape)]
        self.states = [None, None, None]

        self.W_i = self.init((self.input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))

        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((self.input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((self.input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((self.input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))


        self.W_s = self.init((self.input_dim, 1),
                             name='{}_W_s'.format(self.name))

        self.U_s = self.inner_init((self.output_dim, 1),
                                   name='{}_U_s'.format(self.name))
        self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o,
                                                        self.W_s]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                        self.U_f,
                                                        self.U_c,
                                                        self.U_o,
                                                        self.U_s]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o,
                                                        self.b_s]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o,
                                  self.W_s, self.U_s, self.b_s]

    def preprocess_input(self, input):
        if self.first:
            seq = input
            shot_begin = K.ones_like(K.sum(seq, axis=-1, keepdims=True))
            shot_end = K.ones_like(K.sum(seq, axis=-1, keepdims=True))
        else:
            seq = input[:, :, T.arange(self.input_dim)]
            shot_begin = K.expand_dims(input[:, :, self.input_dim], -1)
            shot_end = K.expand_dims(input[:, T.arange(1, self.T), self.input_dim], -1)
            shot_end = K.concatenate([shot_end, K.zeros_like(K.sum(shot_end, axis=1, keepdims=True))], axis=1)
        return K.concatenate([seq, shot_begin, shot_end], axis=-1)

    def step(self, input, states):
        x = input[:, T.arange(self.input_dim)]
        s_lm1_begin = input[:, T.arange(self.input_dim, self.input_dim+1)]
        s_lm1_end = input[:, T.arange(self.input_dim+1, self.input_dim+2)]
        s_lm1_end_rep = K.repeat_elements(s_lm1_end, self.output_dim, -1)

        B_U = states[3]
        B_W = states[4]

        x_s = K.dot(x * B_W[4], self.W_s) + self.b_s
        s_l_begin = K.in_train_phase(stochastic_switch(x_s + K.dot(states[0] * B_U[4], self.U_s)) * s_lm1_end, switch(x_s + K.dot(states[0] * B_U[4], self.U_s)) * s_lm1_end)
        if self.disable_shots:
            s_l_begin *=  0

        s_l_begin_rep = K.repeat_elements(s_l_begin, self.output_dim, -1)

        h_tm1 = states[0]*(1-s_l_begin_rep)
        c_tm1 = states[1]*(1-s_l_begin_rep)

        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
        x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))

        h = o * self.activation(c)

        h_ok = (1-s_lm1_end_rep)*h_tm1 + s_lm1_end_rep*h
        c_ok = (1-s_lm1_end_rep)*c_tm1 + s_lm1_end_rep*c

        if self.disable_shots:
            s_l_begin += 1

        return h_ok, [h_ok, c_ok, s_l_begin]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        if self.first:
            reducer = K.zeros((self.input_dim, self.output_dim))
        else:
            reducer = K.zeros((self.input_dim+1, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)

        initial_state_s = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state_s = K.sum(initial_state_s, axis=1)  # (samples, input_dim)
        if self.first:
            reducer_s = K.zeros((self.input_dim, 1))
        else:
            reducer_s = K.zeros((self.input_dim+1, 1))
        initial_state_s = K.dot(initial_state_s, reducer_s)  # (samples, 1)

        initial_states = [initial_state for _ in range(len(self.states)-1)] + [initial_state_s, ]
        return initial_states

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(5)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])

        if 0 < self.dropout_W < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.input_dim))
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(5)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])
        return constants

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.return_sequences:
            mask = K.expand_dims(mask, -1)
            shots = states[-1]*mask
            shots += K.concatenate([K.zeros_like(shots[:,T.arange(1)]), mask[:,T.arange(self.T-1)] - mask[:,T.arange(1,self.T)]], axis=1)
            return K.concatenate([outputs, shots], axis=-1)
        else:
            return last_output