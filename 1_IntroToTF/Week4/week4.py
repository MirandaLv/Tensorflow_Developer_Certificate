
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tf.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

"""
train_dir -> folder1/folder2
test_dir -> folder1/folder2
"""

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dictionary(
    train_dir,
    target_size=(300, 300), # image resize for you as you load the image, so no need for pre-processing
    batch_size=128, # check on different batch size to see the difference
    class_mode='binary'
)


model = tf.keras.models.Sequences([
    Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid') # sigmoid is for binary classification
])

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    # there are 1024 training images, so we load them in a 128 at a time,
    # 128 is similar to the batch_size when defining the training generator
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8, #256 images for batch 32, so steps is 8
    verbose=2 #how much to display while training is going on
)

"""
Prediction the model
"""
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path)











