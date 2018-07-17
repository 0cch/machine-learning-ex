import os
import numpy as np
import PIL.Image as im
from tensorflow import keras
# from tensorflow.examples.tutorials.mnist import input_data
from skimage import transform as skitf

# mnist = input_data.read_data_sets(".", one_hot=True)

def mnist_write_img(path, num, images, labels):
    file_name_index = [0] * 10
    for i in range(num):
        if (i+1) % 1000 == 0:
            print('\rprogress %u / %u' % (i+1, num), end='')
        label = np.argmax(labels[i])
        directory = path+'\\'+str(label)
        if not os.path.exists(directory):
            os.makedirs(directory)
        image_data = images[i].reshape(28,28,1)
        resize_data = skitf.resize(image_data, (150,150,3), mode='reflect')
        resize_data = (resize_data * 255).astype(np.uint8)
        img = im.fromarray(resize_data)
        img.save(directory+'\\'+str(file_name_index[label])+'.png', 'PNG')
        file_name_index[label]+=1

model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
for layer in model.layers:
    layer.trainable = False


top_model = keras.models.Sequential()
top_model.add(keras.layers.Flatten(input_shape=model.output_shape[1:]))
#top_model.add(keras.layers.Dense(30, activation='relu'))
#top_model.add(keras.layers.Dropout(0.5))
top_model.add(keras.layers.Dense(10, activation='softmax'))

model = keras.models.Model(inputs=model.input, outputs=top_model(model.output))

print(model.summary())

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
train_generator = train_datagen.flow_from_directory('.\\train', target_size=(150,150), batch_size=50)

model.fit_generator(train_generator, steps_per_epoch=1100, epochs=1)

model.save('.\\mnist_inception_v3.h5')
model.save_weights('.\\mnist_inception_v3_weights.h5')