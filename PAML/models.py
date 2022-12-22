import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Implementation of the residual-sup network used as encoder in PAML

class ResGRUCell(tf.keras.layers.Layer):
    def __init__(self,
                num_units = 3,
                layer = 3,
                num_outputs=1):
        super(ResGRUCell, self).__init__()

        #self.rnn_cells = [tf.keras.layers.GRUCell(num_units) for _ in range(layer)]
        #self.stacked_lstm = tf.keras.layers.StackedRNNCells(self.rnn_cells)
        self.stacked_lstm = tf.keras.layers.GRUCell(num_units)
        
        self.emb_layer = tf.keras.layers.Dense(num_outputs)
        self.tanh = tf.keras.layers.Activation('tanh')
        
    @property
    def state_size(self):
        return self.stacked_lstm.state_size
    @property
    def output_size(self):
        return self.stacked_lstm.output_size

    def call(self, inputs, state, scope=None):
        output, new_state = self.stacked_lstm(inputs, state, scope)
        
        outputs = self.emb_layer(output)
        outputs = tf.add(outputs, inputs)
        return outputs, new_state
        

class EncoderModel(Model):
    def __init__(self, gru_layer):
        super(EncoderModel, self).__init__()
        self.gru_layer = gru_layer

    def call(self, x):
        _,x = self.gru_layer(x)
        return x

class DecoderModel(Model):
    def __init__(self, gru_layer):
        super(DecoderModel, self).__init__()
        self.gru_layer = gru_layer

    def call(self, x, s):
        x = self.gru_layer(x,initial_state=s)
        return x[0],x[1:]

