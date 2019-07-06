__author__      = 'Vahed Qazvinian'

import argparse
import numpy as np
import pandas as pd

from utils import get_train_test_data
from models.nb import model as nb_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



target_names = ['non-rumor', 'rumor']

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
  X_train, X_test, y_train, y_test = get_train_test_data(
    task=arguments.task)
  model = nb_model(X_train, y_train)

  if arguments.task == 'sentiment':
    global target_names
    target_names = ['endorse', 'deny', 'question', 'neutral']
  test(model, X_test, y_test)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--task', default='detection', help='[detection/sentiment]')
  args = parser.parse_args()
  main(args)