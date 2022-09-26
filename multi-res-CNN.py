%matplotlib inline
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPool2D, concatenate, Resizing, Cropping2D, BatchNormalization, CenterCrop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True,
                                   samplewise_center=True)
test_datagen = ImageDataGenerator(rescale =1./255)
training_set = train_datagen.flow_from_directory('sports-video-data/train_images',
                                                target_size=(128,128),
                                                batch_size= 32,
                                                class_mode='categorical')
test_set = test_datagen.flow_from_directory('sports-video-data/test_images',
                                           target_size = (128,128),
                                           batch_size = 32,
                                           class_mode ='categorical')


# Converting the generator to the required format for .fit() function

def train_gen():
    gen = train_datagen.flow_from_directory('sports-video-data/train_images',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')
    while True:
        X = gen.next()
        yield [X[0], X[0]], X[1]


train_generator = train_gen()


def test_gen():
    gen = test_datagen.flow_from_directory('sports-video-data/test_images',
                                           target_size=(128, 128),
                                           batch_size=32,
                                           class_mode='categorical')
    while True:
        X = gen.next()
        yield [X[0], X[0]], X[1]


test_generator = test_gen()

# Multiresolution CNN - original architecture

# Input 1
fovea_stream = Input(shape=(178,178,3))
x = Cropping2D(cropping=((45, 44), (45, 44)), input_shape=(178, 178, 3))(fovea_stream)
x = Conv2D(96,11,strides=(3, 3),padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
x = Conv2D(256,5,padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
x = Conv2D(384,3,padding="same", activation="relu")(x)
x = Conv2D(384,3,padding="same", activation="relu")(x)
x = Conv2D(256,3,padding="same", activation="relu")(x)

# Input 2
context_stream = Input(shape=(178,178,3))
y = Resizing(89,89)(context_stream)
y = Conv2D(96,11,strides=(3, 3),padding="same", activation="relu")(y)
y = BatchNormalization()(y)
y = MaxPool2D()(y)
y = Conv2D(256,5,padding="same", activation="relu")(y)
y = BatchNormalization()(y)
y = MaxPool2D()(y)
y = Conv2D(384,3,padding="same", activation="relu")(y)
y = Conv2D(384,3,padding="same", activation="relu")(y)
y = Conv2D(256,3,padding="same", activation="relu")(y)

final = concatenate([x, y])
final = Flatten()(final)
final = Dense(4096, activation='relu')(final)
final = Dense(4096, activation='relu')(final)
outputs = Dense(5, activation='softmax')(final)

model = Model([fovea_stream, context_stream], outputs)
model.summary()

rate_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=4000,
    decay_rate=0.9)

model.compile(optimizer=SGD(rate_scheduler),loss='categorical_crossentropy',metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)

batch_size = 32
num_img = 10843
num_img_test = 2398
multires_model = model.fit_generator(train_generator, steps_per_epoch=int(num_img/batch_size), validation_data=test_generator, validation_steps=int(num_img_test/batch_size), epochs=50)

acc = multires_model.history['accuracy']
val_acc = multires_model.history['val_accuracy']
loss = multires_model.history['loss']
val_loss = multires_model.history['val_loss']

epochs = range(50)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

