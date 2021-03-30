
import tensorflow as tf
from tensorflow import keras
import numpy as np



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.01):
            print("\n Reached 60% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()


mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

"""
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
"""

training_images = training_images/255.0
testing_images = testing_images/255.0

# flatten takes the square images and convert them into a 1 dimensional set
# dense adds a layer of neurons
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

model.evaluate(testing_images, testing_labels)
















