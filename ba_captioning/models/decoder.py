import keras.backend as K
from keras.layers import InputSpec, Layer
from keras import activations, initializations, regularizers
from keras.initializations import normal

class GRU_Decoder(Layer):
    def __init__(self, output_dim,
                 init='normal', inner_init='orthogonal', activation='tanh', inner_activation='sigmoid',
                 W_regularizer=None, U_regularizer=None,
                 weights=None, go_backwards=False,
                 **kwargs):
        self.seq_d_in = None
        self.vd_d_in = None
        self.T = None
        self.seq_d_out = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.initial_weights = weights
        self.go_backwards = go_backwards

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer

        self.input_spec = [InputSpec(ndim=2), InputSpec(ndim=3)]

        super(GRU_Decoder, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        seq_input_shape = input_shapes[1]
        return (seq_input_shape[0], seq_input_shape[1], self.seq_d_out)

    def compute_mask(self, input, mask):
        return mask[1]

    def build(self, input_shapes):
        vid_input_shape = input_shapes[0]
        seq_input_shape = input_shapes[1]
        self.seq_d_in = seq_input_shape[2]
        self.vd_d_in = vid_input_shape[1]
        self.input_spec = [InputSpec(shape=vid_input_shape), InputSpec(shape=seq_input_shape)]
        self.states = [None, None]

        # GRU weights
        self.W_z = normal((self.seq_d_in, self.seq_d_out), scale=0.01)
        self.U_z = self.inner_init((self.seq_d_out, self.seq_d_out))
        self.C_z = normal((self.vd_d_in, self.seq_d_out), scale=0.01)
        self.b_z = K.zeros((self.seq_d_out))

        self.W_r = normal((self.seq_d_in, self.seq_d_out), scale=0.01)
        self.U_r = self.inner_init((self.seq_d_out, self.seq_d_out))
        self.C_r = normal((self.vd_d_in, self.seq_d_out), scale=0.01)
        self.b_r = K.zeros((self.seq_d_out))

        self.W = normal((self.seq_d_in, self.seq_d_out), scale=0.01)
        self.U = self.inner_init((self.seq_d_out, self.seq_d_out))
        self.C = normal((self.vd_d_in, self.seq_d_out), scale=0.01)
        self.b = K.zeros((self.seq_d_out))

        self.trainable_weights = [self.W_z, self.U_z, self.C_z, self.b_z,
                                  self.W_r, self.U_r, self.C_r, self.b_r,
                                  self.W, self.U, self.C, self.b]

    def preprocess_input(self, inputs):
        seq = inputs[1]
        return seq

    def get_constants(self, inputs):
        video = inputs[0]
        return [video]

    def step(self, x, states):
        h_tm1 = states[0]
        video = states[1]

        # GRU
        z_i = self.inner_activation(K.dot(x, self.W_z) + K.dot(h_tm1, self.U_z) + K.dot(video, self.C_z) + self.b_z)
        r_i = self.inner_activation(K.dot(x, self.W_r) + K.dot(h_tm1, self.U_r) + K.dot(video, self.C_r) + self.b_r)
        hh = self.activation(K.dot(x, self.W) + K.dot(r_i * h_tm1, self.U) + K.dot(video, self.C) + self.b)
        h = (1-z_i) * h_tm1 + z_i * hh
        return h, [h]

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        x = inputs[1]
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.seq_d_in, self.seq_d_out))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)

        initial_states = [initial_state]
        return initial_states

    def call(self, x, mask=None):
        # input shape: (b_s, C, T, H, W)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[1].shape
        initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[1],
                                             constants=constants,
                                             unroll=False,
                                             input_length=input_shape[1])

        return outputs