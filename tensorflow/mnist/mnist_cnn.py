# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True)

train_batch_size = 100
test_batch_size = 10000

tf.reset_default_graph()

def variable_summary(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/'+name, stddev)

def inference(input_x, is_reuse):
    with tf.variable_scope('inference', reuse = is_reuse):
        conv_w = tf.get_variable('conv_w', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_b = tf.get_variable('conv_b', [8], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summary(conv_w, 'conv_w')
        variable_summary(conv_b, 'conv_b')
        conv = tf.nn.conv2d(input_x, conv_w, strides=[1,1,1,1],padding='SAME')
        conv = tf.nn.bias_add(conv, conv_b)
        a = tf.nn.relu(conv)
        a = tf.nn.max_pool(a, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
        
        CONV_OUT_PARAMETERS = a.shape[1]*a.shape[2]*a.shape[3]
        w = tf.get_variable('w', [CONV_OUT_PARAMETERS, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', [10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summary(w, 'w')
        variable_summary(b, 'b')
        
        a = tf.reshape(a, [tf.shape(a)[0], CONV_OUT_PARAMETERS])
        y = tf.matmul(a, w) + b
        
        return y

x = tf.placeholder(tf.float32, [train_batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [train_batch_size, 10])

test_x = tf.placeholder(tf.float32, [test_batch_size, 28, 28, 1])
test_y_ = tf.placeholder(tf.float32, [test_batch_size, 10])

y = inference(x, False)
#y = tf.nn.dropout(y, 0.8)
loss = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.3, global_step, 1000, 0.9, True)
tf.summary.scalar('learning_rate', learning_rate)
#regularizer = tf.nn.l2_loss(w)
#train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss+0.01*regularizer)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

test_y = tf.nn.softmax(inference(test_x, True))
correct_prediction = tf.equal(tf.argmax(test_y, 1), tf.argmax(test_y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_array = []
accuracy_array = []

summary_merge = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('.\\mnist_cnn_log\\', sess.graph)
    tf.global_variables_initializer().run()
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs = batch_xs.reshape((100,28,28,1))
        _, step_loss, step, summary_buf = sess.run(
                [train_step, loss, global_step, summary_merge], 
                feed_dict={x: batch_xs, y_: batch_ys})
        summary_writer.add_summary(summary_buf, step)
        if i % 500 == 0:
            step_accuracy = accuracy.eval({test_x: mnist.test.images.reshape((10000,28,28,1)), test_y_: mnist.test.labels})
            loss_array.append(step_loss)
            accuracy_array.append(step_accuracy)
            print(step, step_loss, step_accuracy)

summary_writer.close()
loss_array.append(step_loss)
accuracy_array.append(step_accuracy)
print(step_loss, step_accuracy)

plt.plot([i*500 for i in range(len(loss_array))], loss_array, 'b-',
          [i*500 for i in range(len(accuracy_array))], accuracy_array, 'r-')

plt.show()