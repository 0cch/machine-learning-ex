# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from skimage import transform
import numpy as np

def read_mnist_tfrecord(path, num):
    file_queue = tf.train.string_input_producer([path], shuffle=False)
    coord = tf.train.Coordinator()
    reader = tf.TFRecordReader()
    retval = []
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess, coord)
        
        for i in range(num):
            _, record_data = reader.read(file_queue)
            features = tf.parse_single_example(record_data, features = {
                    'size':tf.FixedLenFeature([], tf.int64),
                    'label':tf.FixedLenFeature([], tf.int64),
                    'image_raw':tf.FixedLenFeature([], tf.string)})
            size_t = tf.cast(features['size'], tf.int32)
            label_t = tf.cast(features['label'], tf.int32)
            image_raw_t = tf.decode_raw(features['image_raw'], tf.float32)
            size, label, image_raw = sess.run([size_t, label_t, image_raw_t])
            retval.append({'size':size, 'label':label, 'image_raw':image_raw})
        coord.request_stop()
        coord.wait_for_stop()
    return retval

example = read_mnist_tfrecord('../mnist/mnist_test.tfrecords', 1)
model = keras.models.load_model('./mnist_inception_v3.h5')
model.load_weights('./mnist_inception_v3_weights.h5')

model.summary()

x = transform.resize(example[0]['image_raw'].reshape(28,28,1), (150,150,3), mode='reflect')
print(example[0]['label'])
y = model.predict(x[np.newaxis])
print(np.argmax(y))