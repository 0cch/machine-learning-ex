# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True)

with tf.Graph().as_default() as g:
    INPUT_PARAMETERS = 784
    L1_PARAMETERS = 300
    W1 = tf.Variable(tf.truncated_normal([INPUT_PARAMETERS, L1_PARAMETERS], stddev=0.1), name='w1')
    b1 = tf.Variable(tf.truncated_normal([L1_PARAMETERS], stddev=0.1), name='b1')
    W2 = tf.Variable(tf.truncated_normal([L1_PARAMETERS, 10], stddev=0.1), name='W2')
    b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name='b2')
    
    x = tf.placeholder(tf.float32, [None, INPUT_PARAMETERS])
    
    hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    y =  tf.nn.sigmoid(tf.matmul(hidden1, W2) + b2)
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    
    
    with tf.Session() as sess:
        state = tf.train.get_checkpoint_state('.\\mnist_checkpoint\\')
        if state and state.model_checkpoint_path:
            print(state.model_checkpoint_path)
            saver.restore(sess, state.model_checkpoint_path)
            step_accuracy = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
        print(step_accuracy)

