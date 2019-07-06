__author__      = 'Vahed Qazvinian'

import argparse

from utils import get_train_test_data
from models.tfidf import NBModel
from models.tfidf import SGDModel
from models.dl import SimpleDense



MODELS = {
  'nb': NBModel,
  'sgd': SGDModel,
  'dense': SimpleDense,
}

  

def main(arguments):

  X_train, X_test, y_train, y_test = get_train_test_data(
    task=arguments.task)

  if arguments.method == 'all':
    for model in MODELS:
      model_class = MODELS.get(model)
      print('=========== {0} ==========='.format(model_class.__name__))
      model = model_class(
        X_train, X_test, y_train, y_test, task=arguments.task)
      model.evaluate()

  elif arguments.method in MODELS.keys():
    model_class = MODELS.get(arguments.method)
    print('=========== {0} ==========='.format(model_class.__name__))
    model = model_class(
      X_train, X_test, y_train, y_test, task=arguments.task)
    model.evaluate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--task', default='detection', help='[detection/sentiment]')
  parser.add_argument(
    '--method', default='nb', help='/'.join(MODELS.keys()))
  args = parser.parse_args()
  main(args)