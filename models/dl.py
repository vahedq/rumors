
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.model import Model


class NNBaseModel(Model):
  def preprocess_data(self):
    # Max number of words in each tweet.
    self.maxlen = 30
    self.tk = Tokenizer(oov_token='<UNK>')
    self.tk.fit_on_texts(self.X_train)
    self.tk.word_index['<PAD>'] = len(self.tk.word_index) + 1
    self.output_size = len(set(self.y_train))
    self.vocab_size = len(self.tk.word_index) + 1
    self.X_train = self.tk.texts_to_sequences(self.X_train)
    self.X_train = pad_sequences(
      self.X_train, value=self.tk.word_index['<PAD>'],
      padding='post', maxlen=self.maxlen
    )
    self.X_test = self.tk.texts_to_sequences(self.X_test)
    self.X_test = pad_sequences(
      self.X_test, value=self.tk.word_index['<PAD>'],
      padding='post', maxlen=self.maxlen
    )


class SimpleDense(NNBaseModel):
  def train(self):    
    self.model = keras.Sequential()
    self.model.add(keras.layers.Embedding(self.vocab_size, 16))
    self.model.add(keras.layers.GlobalAveragePooling1D())
    self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    self.model.add(keras.layers.Dense(self.output_size, activation=tf.nn.sigmoid))
    print(self.model.summary())
    self.model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['acc'])
    history = self.model.fit(
      self.X_train,
      self.y_train,
      epochs=30,
      batch_size=64,
      verbose=1
    )
  
  def test(self):
    self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
    print(self.y_pred)
