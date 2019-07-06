
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Model(object):
  '''High-level model class.'''
  def __init__(self, X_train, X_test, y_train, y_test, task='detection'):
    self.target_names = ['non-rumor', 'rumor']
    if task == 'sentiment':
      self.target_names = ['endorse', 'deny', 'question', 'neutral']
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.y_pred = []
    self.model = None
    self.preprocess_data()

  def preprocess_data(self):
    pass

  def train(self):
    pass

  def test(self):
    pass

  def print_report(self):
    print('Classification Report:')
    print(classification_report(
      self.y_test, self.y_pred, target_names=self.target_names))
    print('\nConfusion Matrix:')
    cm = confusion_matrix(self.y_test, self.y_pred)
    a =  confusion_matrix(self.y_test, self.y_pred)
    print(pd.DataFrame(a, index=self.target_names, columns=self.target_names))
    print('\n')

  def evaluate(self):
    self.train()
    self.test()
    self.print_report()


class NNBaseModel(Model):
  '''Base class for keras data preprocessing.'''
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
    
  def test(self):
    self.y_pred = self.model.predict(self.X_test).argmax(axis=1)
