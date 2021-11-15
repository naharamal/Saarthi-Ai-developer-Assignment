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
from metrics import f1_score_class0

def accuracy(preds,labels):
  return sum(1 for x,y in zip(preds,labels) if x == y) / len(preds)


input ="/content/drive/MyDrive/saarthi/task_data"
output = "/content/drive/MyDrive/saarthi/task_data"


vocab_size = 5000
embedding_dim = 64
max_length = 200
num_epochs =10


train = pd.read_csv(input +"/train_data.csv")
valid = pd.read_csv(output+"/valid_data.csv")

classes = ["action","location","object"]
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

action_model.load_weights(output+f"/checkpoint0")
location_model.load_weights(output+f"/checkpoint1")
object_model.load_weights(output+f"/checkpoint2")

action_preds = action_model.predict(valid_data)
action_preds = [np.argmax(i) for i in action_preds]
metric = tf.keras.metrics.MeanIoU(num_classes=1)
print("action classification accuracy ",accuracy(Ytests["action"], action_preds))

location_preds = location_model.predict(valid_data)
location_preds = [np.argmax(i) for i in location_preds]
print("location classification accuracy ",accuracy(Ytests["location"], location_preds))


object_preds = object_model.predict(valid_data)
object_preds = [np.argmax(i) for i in object_preds]
print("object classification accuracy ",accuracy(Ytests["object"], object_preds))




