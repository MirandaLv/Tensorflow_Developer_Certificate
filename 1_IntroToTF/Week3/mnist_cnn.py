
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

"""
That's because the first convolution expects a single tensor containing everything, 
so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1
"""
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/255.0

testing_images = testing_images.reshape(10000, 28, 28, 1)
testing_images = testing_images/255.0

# build a model with conv layer and maxpooling
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(testing_images, testing_labels)
print(test_loss)