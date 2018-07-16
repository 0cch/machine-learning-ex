# -*- coding: utf-8 -*-
import tensorflow as tf
import glob

IMAGE_SIZE = 32
IMAGE_DEP = 3
def cifar10_loaddata2image(num):
    tf.reset_default_graph()
    input_bin_path = glob.glob(r'cifar10_data\cifar-10-batches-bin\data_batch_*.bin')
    input_queue = tf.train.string_input_producer(input_bin_path, shuffle=False)
    reader = tf.FixedLengthRecordReader(IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEP + 1)
    _, value = reader.read(input_queue)

    image_raw_data = tf.decode_raw(value, tf.uint8)
    label = image_raw_data[0]
    image_data = image_raw_data[1:]
    reshaped_data = tf.reshape(image_data, [IMAGE_DEP, IMAGE_SIZE, IMAGE_SIZE])
    reshaped_data = tf.transpose(reshaped_data, [1, 2, 0])
    output_image_data = tf.image.encode_png(tf.cast(reshaped_data, tf.uint8))
    
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.train.start_queue_runners(sess)
        for i in range(num):
            l, d=sess.run([label, output_image_data])
            with open('sample-%u-label-%u.png' % (i, l), 'wb') as f:
                f.write(d)
    
    