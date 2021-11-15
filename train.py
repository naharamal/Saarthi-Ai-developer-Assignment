#imports
import pandas as pd
import cleanup_utils as cp
import numpy as np
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model as md
from config import * 



train = pd.read_csv(input +"/train_data.csv")
valid = pd.read_csv(output+"/valid_data.csv")
encodes = {}
Y = {}
Ytests = {}

for cls in classes:
  labels = train[cls].unique()
  encod_labels = {labels[i]:i for i in range(len(labels))}
  _labels = []
  for i in train[cls]:
    _labels.append(encod_labels[i])
  _labels = np.array(_labels)


  test_labels = []
  for i in valid[cls]:
    test_labels.append(encod_labels[i])
  test_labels = np.array(test_labels)

  encodes[cls] = encod_labels
  Y[cls] = _labels
  Ytests[cls] = test_labels

cleanup = [cp.remove_accented_chars,cp.remove_special_characters,cp.remove_punctuation,\
    cp.get_stem,cp.remove_punctuation,cp.remove_stopwords,cp.remove_extra_whitespace_tabs,\
        cp.to_lowercase]
for i in cleanup:
  train['transcription'] = train['transcription'].apply(i)
  valid['transcription'] = valid['transcription'].apply(i)

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train['transcription'])
token_train = tokenizer.texts_to_sequences(train['transcription'])
token_valid = tokenizer.texts_to_sequences(valid['transcription'])

vocab_size = len(tokenizer.word_index) + 1
maxlen = max_length


train['padded_sequences'] = pad_sequences(token_train,  padding='post', maxlen=maxlen).tolist()
train_data = np.array([np.array(i) for i in train['padded_sequences']])

valid['padded_sequences'] = pad_sequences(token_valid,  padding='post', maxlen=maxlen).tolist()
valid_data = np.array([np.array(i) for i in train['padded_sequences']])

action_model = md.model(vocab_size,embedding_dim,6)
location_model = md.model(vocab_size,embedding_dim,4)
object_model = md.model(vocab_size,embedding_dim,14)

models = [action_model,location_model,object_model]

for i in range(3):
    models[i].compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = models[i].fit(train_data, Y["location"], epochs=num_epochs, verbose=2)
    models[i].save_weights(output+f"/checkpoint{i}")

