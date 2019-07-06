
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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
