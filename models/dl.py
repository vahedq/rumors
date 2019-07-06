
import numpy as np
import tensorflow as tf

from tensorflow import keras

from models.model import NNBaseModel


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
      epochs=100,
      batch_size=64,
      verbose=1
    )


class BiLSTM(NNBaseModel):
  def train(self):
    self.model = keras.Sequential()
    self.model.add(keras.layers.Embedding(self.vocab_size, 16))
    self.model.add(keras.layers.Bidirectional(
      keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    self.model.add(keras.layers.Dense(16, activation='softmax'))
    self.model.add(keras.layers.Dense(self.output_size, activation='sigmoid'))
    print(self.model.summary())
    self.model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['acc'])
    history = self.model.fit(
      self.X_train,
      self.y_train,
      epochs=100,
      batch_size=64,
      verbose=1
    )