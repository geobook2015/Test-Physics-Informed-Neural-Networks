import sys 
sys.path.insert(0, 'Utilities/')

import tensorflow as tf 
import numpy as np 
import scipy.io 
import sampling 
import time

class Linear: # Initialize the class 
    def __init__(self, layers): 
        self.weights, self.biases = self.initialize_NN(layers)

    def forward(self, X):
        num_layers = len(self.weights) + 1
    
        H = X
        for l in range(0, num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = self.weights[-1]
        b = self.biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def load(self, path):
    
        pass

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers-1):
    
            name = str(l)
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        
        return weights, biases


    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
















