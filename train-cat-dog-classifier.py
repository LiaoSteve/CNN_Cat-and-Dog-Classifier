import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
classes = ['cats', 'dogs']
train_path = 'dataset/training_set/'
test_path = 'dataset/test_set/'

train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224),
     classes=classes, batch_size=6)

test_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), 
    classes=classes, batch_size=10, shuffle=False)

imgs, labels = next(train_batches)

def plotImages(images):
    plt.figure(figsize=(8,8))
    for i in range(len(images)):
        plt.subplot(8,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])  
    plt.tight_layout()     
    plt.show()   

plotImages(imgs)
print(labels)

inputs = keras.Input(shape=(224,224,3), name="images-input")
x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = layers.MaxPool2D(pool_size=2, strides=2)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=2, strides=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(units=2, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='cat-vs-dog-cnn-model')
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
     loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, epochs=5, verbose=2)