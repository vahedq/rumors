__author__      = 'Vahed Qazvinian'

import argparse
import numpy as np
import pandas as pd

from utils import get_train_test_data
from models.tfidf import sdg_model
from models.tfidf import nb_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



target_names = ['non-rumor', 'rumor']
METHODS = {
  'nb': ('Naive Bayes', nb_model),
  'sgd': ('Stochastic Gradient Descent', sdg_model),
}

def test(model, X_test, y_test):
  y_pred = model.predict(X_test)
  print('Classification Report:')
  print(classification_report(
    y_test, y_pred, target_names=target_names))
  print('Confusion Matrix:')
  cm = confusion_matrix(y_test, y_pred)
  a =  confusion_matrix(y_test, y_pred)
  print(pd.DataFrame(a, index=target_names, columns=target_names))


def main(arguments):
  if arguments.task == 'sentiment':
    global target_names
    target_names = ['endorse', 'deny', 'question', 'neutral']
  X_train, X_test, y_train, y_test = get_train_test_data(
    task=arguments.task)
    
  if arguments.method == 'all':
    for method in METHODS:
      print('=========== {0} ==========='.format(METHODS.get(method)[0]))
      model = METHODS.get(method)[1](X_train, y_train)
      test(model, X_test, y_test)
      print('\n')

  elif arguments.method in METHODS.keys():
    method = arguments.method
    print('=========== {0} ==========='.format(METHODS.get(method)[0]))
    model = METHODS.get(method)[1](X_train, y_train)
    test(model, X_test, y_test)
    print('\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--task', default='detection', help='[detection/sentiment]')
  parser.add_argument(
    '--method', default='nb', help='/'.join(METHODS.keys()))
  args = parser.parse_args()
  main(args)