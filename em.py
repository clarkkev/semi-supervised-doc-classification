from time import time

from loader import DataGatherer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
import itertools

import numpy as np
from scipy.sparse import *

import math
from util import subset, subsets

def sparse_diagonal(v, n):
  return coo_matrix(([v] * n, (range(n), range(n))))  

def get_word_probs(clf, X_train):
  #print X_train.shape
  #print sparse_diagonal(1, X_train.shape[1]).shape
  return clf.predict_proba(sparse_diagonal(1, X_train.shape[1]))

def em(clf, vectorizor, unlabeled, labeled, test, y_labeled, y_test, mode="hard"):
  print mode
  #print sparse_diagonal(1, 4).todense()
  
  print "Transforming data..."
  X_train = vectorizor.fit_transform(labeled + unlabeled)
  X_labeled = vectorizor.transform(labeled)
  X_test = vectorizor.transform(test)

  #for i in range(len(X_train.data)):
  #  print X_train.data[i]
  #print "SHAPE: ", X_train.shape[1]
   

  ws = ([2] * len(labeled)) + ([1] * len(unlabeled))

  clf2 = MultinomialNB(alpha=0.01)
  if mode=='SFE':
    initial = benchmark(clf2, X_labeled, X_test, y_labeled, y_test)
  initial = benchmark(clf, X_labeled, X_test, y_labeled, y_test)
 
  final = 0
  for iteration in range(1):
    if mode=='hard':
      pred = clf.predict(X_train)
      clf.fit(X_train, pred, ws)
      final = evaluate(clf, X_test, y_test)
      print final

    elif mode=='soft':
      # E-step
      prob = clf.predict_proba(X_train)
      classes = prob.shape[1]

      # M-step
      ys, ws_new = [], []
      for c in range(classes):
        for i in range(prob.shape[0]):
          ys.append(c)
          if prob[i, c] == 0:
            ws_new.append(0)
          else:
            #ws_new.append(ws[i] * int(1e12 / (2 - math.log(prob[i, c]))))
            ws_new.append(ws[i] * int(1e12 * prob[i, c]))
      Xs = vstack([X_train] * classes)
      print "Training"
      clf.fit(Xs, ys, ws_new)
      final = evaluate(clf, X_test, y_test)
      print final
    
    elif mode=='SFE':
      print "Computing word probs..."
      prob = clf.predict_proba(X_train)
      probs = get_word_probs(clf2, X_train)
      classes = probs.shape[1]

      ys, ws_new = [], []
      for c in range(classes):
        for i in range(X_train.shape[0]):
          ys.append(c)
          if prob[i, c] == 0:
            ws_new.append(0)
          else:
            #ws_new.append(ws[i] * int(1e12 / (2 - math.log(prob[i, c]))))
            ws_new.append(ws[i] * int(1000 * prob[i, c]))
 
      stacks = []     
      for c in range(classes):
        #aprint "Building for class ", c
        word_probs = probs[:,c]
        #for i in range(len(word_probs)):
        #  print word_probs[i] * 20
        
        prob_matrix = diags([word_probs], [0])
        #prob_matrix = diags([[1] * X_train.shape[1]], [0])
        #for i in range(prob.shape[0]):
        
        #print X_train.shape, prob_matrix.shape
        
        stacks.append((X_train * prob_matrix).tocoo())
        #print X_train.shape
        #print prob_matrix.shape
        #stacks.append(X_train)
      
      Xs = vstack(stacks)
      #Xs = vstack([X_train] * classes)
      print "Training"
      clf.fit(Xs, ys)
      clf2.fit(Xs, ys, ws_new)
      final = evaluate(clf, X_test, y_test)
      print final

      '''print "Computing word probs..."
      #probs = clf.predict_proba(X_train)
      probs = get_word_probs(clf, X_train)
      classes = probs.shape[1]

      ys, ws_new = [], []
      for c in range(classes):
        for i in range(probs.shape[0]):
          ys.append(c)
          #if probs[i, c] == 0:
          #  ws_new.append(0)
          #else:
            #ws_new.append(ws[i] * int(1e12 / (2 - math.log(prob[i, c]))))
          #  ws_new.append(ws[i] * int(1e12 * probs[i, c]))
 
      stacks = []     
      for c in range(classes):
        print "Building for class ", c
        word_probs = probs[:,c] 
        prob_matrix = diags([word_probs], [0])
        prob_matrix = diags([[1] * X_train.shape[0]], [0])
        #for i in range(prob.shape[0]):
        #stacks.append((X_train * prob_matrix).tocoo())
        print X_train.shape
        print prob_matrix.shape
        stacks.append(X_train)
      
      #Xs = vstack(stacks)
      Xs = vstack([X_train] * classes)
      print "Training"
      clf.fit(Xs, ys)
      final = evaluate(clf, X_test, y_test)
      print final'''
    else:
      print "MODE NOT VALID"

  return initial, final

  

def naive_bayes(train, test, y_train, y_test):
  vectorizor = CountVectorizer(lowercase=False, stop_words='english', charset_error='ignore')
  X_train = vectorizor.fit_transform(train)
  X_test = vectorizor.transform(test)
  benchmark(MultinomialNB(), X_train, X_test, y_train, y_test)

def benchmark(clf, X_train, X_test, y_train, y_test):
  #print("Training: ")
  #print(clf)
  t0 = time()
  clf.fit(X_train, y_train)
  train_time = time() - t0
  #print("train time: %0.3fs" % train_time)

  return evaluate(clf, X_test, y_test)
  #print

def evaluate(clf, X_test, y_test):
  t0 = time()
  pred = clf.predict(X_test)
  test_time = time() - t0
  #print("test time:  %0.3fs" % test_time)

  score = metrics.precision_score(y_test, pred)
  #print("precision:   %0.3f" % score)
  return score

def run_em(dg, size, n):
  #labeled_data, labeled_target = subset(dg.labeled_data, dg.labeled_target, 0, 0.1)
  clf = MultinomialNB(alpha=0.4)
  #clf = RidgeClassifier(tol=1e-2, solver="lsqr")
  #clf=Perceptron(n_iter=50)
  #clf = PassiveAggressiveClassifier(n_iter=50)
  #clf = KNeighborsClassifier(n_neighbors=10)
  #clf = NearestCentroid()

  #clf= SGDClassifier(alpha=.0001, n_iter=50,
  #                                     penalty="elasticnet")

  vectorizor = CountVectorizer(lowercase=True,stop_words='english',max_df=.5,min_df=2,charset_error='ignore')
  
  labeled = subsets(dg.labeled_data, dg.labeled_target, size, n)

  initials = []
  finals = []
  lengths = []
  for labeled_data, labeled_target in labeled:
    print "LEN: " + str(len(labeled_data))
    initial, final = \
      em(clf, vectorizor, dg.unlabeled_data, 
         labeled_data, dg.validate_data,
         labeled_target, dg.validate_target,
         mode='SFE')

    print initial, final
    initials.append(initial)
    finals.append(final)
    lengths.append(len(labeled_data))
  print 80 * "="
  print size, avg(lengths), avg(initials), avg(finals)
  print 80 * "="

def avg(l):
  return sum(l) / len(l)

def main():
  dg = DataGatherer()
  #naive_bayes(dg.labeled_data[:1000], dg.validate_data,
  #            dg.labeled_target[:1000], dg.validate_target)
  
  
  labeled_data, labeled_target = subset(dg.labeled_data, dg.labeled_target, 0, 0.2)

  vectorizor = CountVectorizer(lowercase=True,stop_words='english',max_df=.5,min_df=2,charset_error='ignore')
  run_em(dg, 50 / (18846 * 0.2), 10)
  run_em(dg, 100 / (18846 * 0.2), 10)
  run_em(dg, 200 / (18846 * 0.2), 10)
  run_em(dg, 350 / (18846 * 0.2), 7)
  run_em(dg, 500 / (18846 * 0.2), 5)
  run_em(dg, 750 / (18846 * 0.2), 4)
  run_em(dg, 1000 / (18846 * 0.2), 3)
  run_em(dg, 1500 / (18846 * 0.2), 2)
  run_em(dg, 2000 / (18846 * 0.2), 1)
  run_em(dg, 2500 / (18846 * 0.2), 1)
  run_em(dg, 3000 / (18846 * 0.2), 1)
  
  #initial, final =
  #  em(clf, vectorizor, dg.unlabeled_data, 
  #     labeled_data, dg.validate_data,
  #     labeled_target, dg.validate_target)

if __name__ == '__main__':
  main()
