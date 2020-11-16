import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model("cifar_10.h5")
model.summary()
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x = model.predict(np.expand_dims(x_test[0],0), batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)

predict = np.argmax(x)
label = y_test[0][0]

import matplotlib.pyplot as plt
plt.imshow(x_test[0])
plt.title(f'predict: {predict}, label: {label}')
plt.show()