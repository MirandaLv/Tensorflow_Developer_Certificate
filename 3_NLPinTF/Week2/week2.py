
import tensorflow as tf
print(tf.__version__)
import numpy as np

# needed for tf 1x
tf.enable_eager_execution()

# pip install -q tensorflow-datasets if using

import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb["train"], imdb["test"]
training_sentences = list()
training_labels = list()

testing_sentences = list()
testing_labels = list()

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())





