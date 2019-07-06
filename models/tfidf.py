'''Naive-Bayes classifier on bag of words representation.'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


def nb_model(X_train, y_train):
  pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
  ])
  pipeline.fit(X_train, y_train)
  return pipeline


def sdg_model(X_train, y_train):
  pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(
      loss='hinge', penalty='l2',
    )),
  ])
  pipeline.fit(X_train, y_train)
  return pipeline
