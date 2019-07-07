
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Reshape

from models.model import NNBaseModel


class SimpleDense(NNBaseModel):
  def train(self):    
    self.model = Sequential()
    self.model.add(Embedding(self.vocab_size, 16))
    self.model.add(GlobalAveragePooling1D())
    self.model.add(Dense(16, activation=tf.nn.relu))
    self.model.add(Dense(self.output_size, activation=tf.nn.sigmoid))
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
    batch_size = 64
    units = 100
    embedding_matrix = np.zeros((self.vocab_size, 100))
    for word, index in self.tk.word_index.items():
      embedding_vector = self.word2vec.get(word)
      if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

    self.model = Sequential()
    self.model.add(
      Embedding(self.vocab_size, units, weights=[embedding_matrix], trainable=False)
    )
    self.model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=0.2)))
    self.model.add(Bidirectional(LSTM(units, dropout=0.2)))
    self.model.add(Dense(self.output_size, activation='sigmoid'))
    print(self.model.summary())
    self.model.compile(optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['acc'])
    history = self.model.fit(
      self.X_train,
      self.y_train,
      epochs=100,
      batch_size=batch_size,
      verbose=1
    )