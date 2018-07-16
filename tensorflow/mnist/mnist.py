# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import PIL.Image as im

def minst_to_image(data, name):
    two_d = 255 - (np.reshape(data, (28, 28)) * 255).astype(np.uint8)
    img = im.fromarray(two_d, 'L')
    img.save('.\\'+name, 'PNG')
    

mnist = input_data.read_data_sets(".", one_hot=True)

INPUT_PARAMETERS = 784
L1_PARAMETERS = 300
W1 = tf.Variable(tf.truncated_normal([INPUT_PARAMETERS, L1_PARAMETERS], stddev=0.1), name='w1')
b1 = tf.Variable(tf.truncated_normal([L1_PARAMETERS], stddev=0.1), name='b1')
W2 = tf.Variable(tf.truncated_normal([L1_PARAMETERS, 10], stddev=0.1), name='W2')
b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name='b2')

x = tf.placeholder(tf.float32, [None, INPUT_PARAMETERS])

hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y =  tf.matmul(hidden1, W2) + b2

y_ = tf.placeholder(tf.float32, [None, 10])
loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=y, labels=y_)))
train_step = tf.train.GradientDescentOptimizer(3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

loss_array = []
accuracy_array = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, step_loss = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 500 == 0:
            step_accuracy = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
            loss_array.append(step_loss)
            accuracy_array.append(step_accuracy)
            print(step_loss, step_accuracy)
    saver.save(sess, '.\\mnist_checkpoint\\mnist.save')
loss_array.append(step_loss)
accuracy_array.append(step_accuracy)
print(step_loss, step_accuracy)



plt.plot([i*500 for i in range(len(loss_array))], loss_array, 'b-',
          [i*500 for i in range(len(accuracy_array))], accuracy_array, 'r-')