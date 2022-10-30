#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:59:36 2022

@author: sudarsun
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.Variable(tf.random.uniform(shape=(1,1)))
b = tf.Variable(tf.zeros(shape=(1,)))

# create a 100,1 X matrix
x_train = np.random.rand(100,1)
y_train = np.array([0 if i < 0.5 else 1 for i in x_train]).reshape(-1,1)

plt.plot(x_train, y_train, 'b.')
plt.show()

def output(x):
    return tf.sigmoid(W*x + b)

def loss_function(yhat,y):
    return tf.reduce_sum(tf.square(yhat - y))

learning_rate = 0.2
steps = 1000
prev_loss = -1
for i in range(steps):
    with tf.GradientTape() as tape:
        predictions = output(x_train)
        if np.mod(i,40) == 0:
            plt.plot(x_train, y_train, 'b.')
            plt.plot(x_train, predictions, '+', color='r')
            plt.show()
        loss = loss_function(predictions, y_train)
        dloss_dw, dloss_db = tape.gradient(loss, [W,b])
    #print(f"epoch [{i}]: loss={loss.numpy()} W={W.numpy()} b={b.numpy()}")
    W.assign_sub(learning_rate * dloss_dw)
    b.assign_sub(learning_rate * dloss_db)
    print(f"epoch [{i}]: loss={loss.numpy()} W={W.numpy()} b={b.numpy()}")
    
    if i != 0 and np.fabs(prev_loss - loss.numpy()) < 0.00001:
        print("converged, stopping!")
        break
    
    prev_loss = loss.numpy()
    
