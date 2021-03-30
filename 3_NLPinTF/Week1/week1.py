
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Read the sarcastic dataset
with open("./sarcasm.json", "r") as f:
    datastore = json.load(f)

sentences = list()
labels = list()
urls = list()

for item in datastore:

    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])
    urls.append(item["article_link"])



"""
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
"""

# Each word will have a token number, the word is key, and the token value is value
tokenizer = Tokenizer(oov_token="<OOV>") # for the word that are not in the token, use oov to represent
tokenizer.fit_on_texts(sentences) # give each word the token
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences) # Use the token to represent sentences

padded = pad_sequences(sequences, padding='post')
# padding, the maximum length for each sentense is set to 5, so the last sentense uses 5 token to represent

print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)

# Try with words that the tokenizer wasn't fit to
"""
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)
"""















