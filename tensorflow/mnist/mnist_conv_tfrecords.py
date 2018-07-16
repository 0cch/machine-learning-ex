# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot=True)

def _int64_feature(val):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[val]))

def _bytes_feature(val):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[val]))

def mnist_write_tfrecords(path, mnist, name):
    writer = tf.python_io.TFRecordWriter(path+'\\mnist_'+name+'.tfrecords')
    image_num = mnist.num_examples
    for index in range(image_num):
        features = tf.train.Features(feature = {'size': _int64_feature(mnist.images.shape[1]),
                    'label': _int64_feature(np.argmax(mnist.labels[index])),
                    'image_raw': _bytes_feature(mnist.images[index].tostring())})
        example = tf.train.Example(features = features)
        writer.write(example.SerializeToString())
    writer.flush()
    writer.close()
    

mnist_write_tfrecords('.', mnist.train, 'train')
mnist_write_tfrecords('.', mnist.validation, 'validation')
mnist_write_tfrecords('.', mnist.test, 'test')