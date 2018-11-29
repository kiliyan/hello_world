#!/usr/bin/python
# -*- coding: utf-8 -*-
# -- coding: utf-8 --


import numpy as np
import tensorflow as tf

data = np.genfromtxt('data.csv', delimiter=',')
np.random.shuffle(data)


# Parameters
X_train=np.array(data[0:120,0:6])
Y=np.array(data[0:120,6])
Y_train = Y.reshape((Y.shape[0], 1))

X_test=np.array(data[0:120,0:6])
Y=np.array(data[0:120,6])
Y_test = Y.reshape((Y.shape[0], 1))


#print(Y.shape)
#print(X.shape)

#Normalizing the data


def compute_cost(Z3, Y):
  
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###
    
    return cost


def initialize_parameters(arr[]):
  
   for i in range(len(arr)):
      parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
      parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

return parameters
 
def L_modal_forward(X,parameters)
  for i in range(len(parameters)/2):
    Z = tf.add(tf.matmul(W1,X),b1)                                           
    A1 = tf.nn.relu(Z1)                            


 l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache=linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="relu")
        caches.append(cache)
        ### END CODE HERE ###
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],activation="sigmoid")
    caches.append(cache)

