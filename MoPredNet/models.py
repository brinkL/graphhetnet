#our own implementation of MoPredNet layers
import tensorflow as tf
import numpy as np

from MoPredNet.utils import ConvOffset2D


class DSTCEncoder(tf.keras.Model):
    def __init__(self,drop=0.5):
        super(DSTCEncoder, self).__init__()
        
        
        self.conv0 = tf.keras.layers.Conv2D(16, kernel_size=[3,3], strides=3, activation=None,padding="same")
        
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=[2,3], strides=2, activation=tf.keras.layers.LeakyReLU(0.5),padding="same")
        self.drop1 = tf.keras.layers.Dropout(drop)
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=[2,3], strides=2, activation=tf.keras.layers.LeakyReLU(0.5),padding="same")
        self.drop2 = tf.keras.layers.Dropout(drop)
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=[2,3], strides=2, activation=tf.keras.layers.LeakyReLU(0.5),padding="same")
        self.drop3 = tf.keras.layers.Dropout(drop)
        
        
        self.conv_offset1 = ConvOffset2D(16, time_axis=0)
        self.conv_offset2 = ConvOffset2D(64, time_axis=0)
        self.conv_offset3 = ConvOffset2D(128,time_axis=0)
        
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense_out = tf.keras.layers.Dense(512, activation=None)


    def call(self, x, training = False):
        
        
        x = self.conv0(x)
        
        #DSTCN
        offset_1 = self.conv_offset1(x)
        x = self.conv1(offset_1)
        x = self.drop1(x,training)
        
        #DSTCN
        offset_2 = self.conv_offset2(x)
        x = self.conv2(offset_2)
        x = self.drop2(x,training)
        
        #DSTCN
        offset_3 = self.conv_offset3(x)
        x = self.conv3(offset_3)
        x = self.drop3(x,training)
        
        
        x = self.flatten(x)
        
        x = self.dense_out(x)
        
        return x
    
    
class FCNet(tf.keras.Model):
    def __init__(self,drop=0.5,layer_sizes = [512,128,54],activation="lrelu"):
        super(FCNet, self).__init__()
        
        self.layer_sizes  = layer_sizes
        self.forward = []

        for idx,num_neurons in enumerate(layer_sizes[:-1]):

            if activation == "lrelu":
                act = tf.keras.layers.LeakyReLU(0.5)
            elif activation == "relu":
                act = tf.keras.layers.ReLU()

            self.forward.append(tf.keras.layers.Dense(num_neurons, activation=act))

            if drop is not None and drop != 0.0:
                self.forward.append(tf.keras.layers.Dropout(drop))

        # Output layer
        self.forward.append(tf.keras.layers.Dense(layer_sizes[-1], activation=None))
        self.seq = tf.keras.Sequential(layers=self.forward)

    def call(self, x, training = False):
        
        x = self.seq(x,training = training)
        
        return x
    
class Masking(tf.keras.Model):
    def __init__(self,factor =  0.0666,max_k = 20, joints=54):
        super(Masking, self).__init__()
        
        self.factor = factor # masking threshold value
        self.maxK   = max_k  # Big K variable, max number of look-backs
        self.joints = joints # channel size
        
    def call(self,x,x_hat,debug=False):
        
        i = 0
        length = tf.shape(x)[1]
        
        # x_hat is tiled to the lenght of x so it can multiply to every time-step in (5)
        x_hat_tile = tf.tile(x_hat,[1,length,1,1])
        
        # this is another way of doing exp(xt_k^T x_t+k-1) in (5)
        inner = tf.multiply(x_hat_tile,x)
        inner = tf.exp(tf.reduce_sum(inner,axis=2))
        
        #find the summation of the bottom part of the fraction in (5) then execute the division
        bottom= tf.reduce_sum(inner[:,length-self.maxK:,:],axis=1,keepdims=True)
        a = inner / tf.tile(bottom,[1,length,1])
        
        # apply mask threshold
        mask = tf.cast(tf.greater(a,self.factor),tf.float32)
        
        #expand the mask to all channels and apply it
        mask = tf.expand_dims(mask,axis=2)
        mask = tf.tile(mask,[1,1,self.joints,1])
        new_x = tf.multiply(x,mask)
        
        if debug:
            return tf.concat([new_x,x_hat],axis=1),mask
        # concat the original input and return
        return tf.concat([new_x,x_hat],axis=1)