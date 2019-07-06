'''Naive-Bayes classifier on bag of words representation.'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

from models.model import Model


class NBModel(Model):
  '''Naive Bayes model'''
  def train(self):
    self.model = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('clf', MultinomialNB()),
    ])
    self.model.fit(self.X_train, self.y_train)

  def test(self):
    self.y_pred = self.model.predict(self.X_test)


class SGDModel(Model):
  '''Stochastic Gradient Descent.'''

  def train(self):
    self.model = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('clf', SGDClassifier(
        loss='hinge', penalty='l2',
      )),
    ])
    self.model.fit(self.X_train, self.y_train)
  
  def test(self):
    self.y_pred = self.model.predict(self.X_test)
