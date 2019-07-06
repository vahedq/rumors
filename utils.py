'''data utils.'''
__author__      = 'Vahed Qazvinian'

import random
import numpy as np

from sklearn.model_selection import train_test_split


datasets = [
  './data/airfrance.txt',
  './data/michelle.txt',
  './data/palin.txt',
]


def create_data_pool(task):
  tweets = []
  X = []
  y = []
  for data in datasets:
    with open(data, 'r', encoding = 'ISO-8859-1') as f:
      tweets.extend(f.readlines())
  # Set a fixed seed for reproducibility.
  random.seed(20)
  random.shuffle(tweets)
  for tweet in tweets:
    parts = tweet.strip().split('\t')
    text = parts[2]
    annotation = parts[3]
    if int(annotation) == 2:
      # Ignore any tweet where annoator is undetermined.
      continue
    if task == 'detection':
      X.append(text)
      label = 1 if int(annotation) > 0 else 0
      y.append(label)
    elif task == 'sentiment' and int(annotation) > 0:
      # Only add tweets that have
      X.append(text)
      # 11 -> if the tweet endorses the rumor
      # 12 -> if the tweet denies the rumor
      # 13 -> if the tweet questions the rumor
      # 14 -> if the tweet is neutral
      # Map it to [0-3] classes
      # Convert to integers 0-3
      y.append(int(annotation) - 11)
  return X, y


def get_train_test_data(task):
  if task not in {'detection', 'sentiment'}:
    return []
  X, y = create_data_pool(task)
  # Already shuffled in create_data_pool() with fixed seed.
  # don't shuffle further for reproducibility.
  return train_test_split(
    np.array(X), np.array(y), train_size=0.8, shuffle=False)