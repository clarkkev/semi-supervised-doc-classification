from utils import *
import numpy as np
import random

from math import sqrt
from operator import itemgetter

from sklearn import metrics

def get_y(examples, target_fun):
  return np.array([target_fun(e) for e in examples])

def shuffle(X, y, examples):
  p = range(len(y))
  random.seed(0)
  random.shuffle(p)
  return X[p], y[p], [examples[p[i]] for i in range(len(p))]

def rmse(y, p):
  return sqrt(sum((y[i] - p[i]) ** 2 for i in range(len(y))) / float(len(y)))

def get_metrics(y, p, metric_list):
  print metrics.confusion_matrix(y, p)
  return [metric_mapping[label](y, p) for label in metric_list]

metric_mapping = {'f1': metrics.f1_score,
          'precision': metrics.precision_score,
          'recall': metrics.recall_score,
          'rmse':  rmse}
default_metrics = ['f1', 'precision', 'recall']

def print_metrics(y, p, metric_list=default_metrics):
  #mean = sum((y[i] - p[i]) ** 2 for i in range(len(y))) / float(len(y))
  N = len(y)
  errors = [(y[i] - p[i]) ** 2 for i in range(N)]
  mean = float(sum(errors)) / N
  stdv = sqrt(sum((errors[i] - mean) ** 2 for i in range(N)) / N)
  
  print '  N: ' + str(N)
  print '  Mean: {0:.4}'.format(mean)
  print '  Sigma: {0:.4}'.format(stdv)
  
  for i in range(len(metric_list)):
    label = metric_list[i]
    f = metric_mapping[metric_list[i]]
    print '  ' + label + ': {0:.4}'.format(f(y, p))

def run_classifier(clf, X, y, metric_list=default_metrics, 
                        splits=None, n_components=None, pd=None,
                        selector=None, pred_fun=None, method='content'):
  if splits == None:
    print "No splits provided, doing 50/50 train/test split"
    n = len(y)
    splits = [(np.arange(n/2), np.arange(n/2, n))]
  
  print method
  metric_scores = []
  for train_index, test_index in splits:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if pd:
      X_train, X_test = pd.decompose(X_train, X_test)

    if selector:
      X_train = selector.fit_transform(X_train, y_train)
      X_test = selector.transform(X_test)

    clf.fit(X_train, y_train)
    p = clf.predict(X_test)
    metric_scores.append(get_metrics(y_test, p, metric_list))
    if pred_fun:
      pred_fun(p, test_index)

  avg_scores = [sum(s[i] for s in metric_scores) / len(metric_scores) \
                for i in range(len(metric_scores[0]))]
  for i in range(len(metric_list)):
    print '  ' + metric_list[i] + ': {0:.4}'.format(avg_scores[i])

class PartialDecomposer():
  def __init__(self, decomposer, features):
    self.decomposer = decomposer
    self.features = sorted(features)
  
  def decompose(self, X_train, X_test):
    def move_column(source, target, c):
      column = source[:, c]
      source = np.delete(source, c, 1)
      if target == None: 
        return source, column
      else:
        return source, np.column_stack([target, column])
    
    to_dec_train = to_dec_test = None
    n = 0
    for f in self.features:
      X_train, to_dec_train = move_column(X_train, to_dec_train, f - n)
      X_test, to_dec_test = move_column(X_test, to_dec_test, f - n)
      n += 1

    dec_train = self.decomposer.fit_transform(to_dec_train)
    dec_test = self.decomposer.transform(to_dec_test)

    return np.column_stack([X_train, dec_train]), \
           np.column_stack([X_test, dec_test])

class FeatureGenerator:
  def __init__(self, examples, features_fun):
    self.features_fun = features_fun
    
    feature_dicts = [features_fun(e) for e in examples]
    self.features = reduce(lambda l1, l2: set(l1) | set(l2),
                           [d.keys() for d in feature_dicts])
    self.feature_mapping = dict(zip(self.features, range(len(self.features))))
    self.inverse_feature_mapping = invert(self.feature_mapping)
    
  def get_X(self, examples):
    feature_dicts = [self.features_fun(e) for e in examples]
    matrix = np.zeros((len(feature_dicts), len(self.feature_mapping)))
    for i in range(len(feature_dicts)):
      d = feature_dicts[i]
      for f in d:
        matrix[i, self.feature_mapping[f]] = d[f]
    return matrix

  def selected_features(self, selection_fun):
    return [self.feature_mapping[f] 
            for f in self.features if selection_fun(f)]

  def print_feature_scores(self, X, y, metric):
    print 80 * '='
    print 'Feature Scores'
    feature_scores = metric(X, y)
    feature_list = [(self.inverse_feature_mapping[i], 
                     feature_scores[1][i])
                    for i in range(len(feature_scores[1]))]
    feature_list = sorted(feature_list, key=itemgetter(1))[:]
    
    n = 0
    for f in feature_list:
      n += 1
      print str(n) + '. ' + str(f[0]) + ' ' + str(f[1])

