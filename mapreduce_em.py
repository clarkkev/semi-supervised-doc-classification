import random
import time
import sys
sys.path.append("..")
import loader
import util

import numpy as np
import scipy
from multiprocessing import Pool
from operator import itemgetter
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

MAX_ITER = 20
NUM_TARGETS = 20

# Tell pickle how to handle instance methods
# so we can multiprocess them
def _pickle_method(method):
  func_name = method.im_func.__name__
  obj = method.im_self
  cls = method.im_class
  return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
  for cls in cls.mro():
    try:
      func = cls.__dict__[func_name]
    except KeyError:
      pass
    else:
      break
  return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def vectorize_and_run(classifier, vectorizer, labeled_data,
                      targets, unlabeled_data, processes, labeled_weight):
  labeled_data = partition_and_vectorize(labeled_data, vectorizer, processes)
  unlabeled_data = partition_and_vectorize(unlabeled_data, vectorizer, processes)
  test_data = scipy.sparse.vstack(partition_and_vectorize(test_data,
                                                         vectorizer,
                                                          processes))
  targets = list(chunks(targets, len(targets) / processes))
  return em(classifier, labeled_data, targets, unlabeled_data, \
              processes, labeled_weight)

def partition_and_run(classifier, labeled_data, targets, unlabeled_data,
                        processes, labeled_weight):
  labeled_data = labeled_data.tocsr()
  labeled_data = list(matrix_chunks(labeled_data, processes))
  unlabeled_data = unlabeled_data.tocsr()
  unlabeled_data = list(matrix_chunks(unlabeled_data, processes))
  targets = list(chunks(targets, len(targets) / processes))

  return em(classifier, labeled_data, targets, unlabeled_data, \
              processes, labeled_weight)

def chunks(l, n):
  for i in xrange(0, len(l), n):
    yield l[i:i+n]

def matrix_chunks(l, n):
  n = l.shape[0] / n
  for i in xrange(0, l.shape[0], n):
    yield l[i:i+n, :]

def tocsr(x):
  return x.tocsr()

def get_labeled_counts( (data_chunk, targets) ):
  counts = np.zeros([NUM_TARGETS, data_chunk.shape[1]])
  for i in xrange(data_chunk.shape[0]):
    counts[targets[i]] += data_chunk.getrow(i).todense().getA()[0]
  return counts

def get_unlabeled_counts( (data_chunk, targets) ):
  return (data_chunk.T * targets).T

def build_classifer_labeled(classifier, data_chunks, 
                            target_chunks, pool):
  counts = pool.map(get_labeled_counts, zip(data_chunks, target_chunks))
  x = np.vstack(counts)
  classifier.fit(x, np.array(range(NUM_TARGETS) * len(counts)))

def build_classifer_mixed(classifier, labeled_chunks, label_chunks,
                        unlabeled_chunks, projected_label_chunks, pool,
                          labeled_weights):
  counts = pool.map(get_labeled_counts, zip(labeled_chunks, label_chunks))
  counts += pool.map(get_unlabeled_counts, zip(unlabeled_chunks, 
                                                projected_label_chunks))
  weights = [labeled_weights]*NUM_TARGETS*(len(counts)/2) + \
      [1]*NUM_TARGETS*(len(counts)/2)
  classifier.fit(np.vstack(counts), np.array(range(NUM_TARGETS) * len(counts)),
                weights)

#  weights = np.ones(len(counts) * NUM_TARGETS)
#  np.fill(weights, len(counts)/2, 2)

# Assumes data is partitioned
def em(classifier, labeled_data, targets, unlabeled_data, \
         processes, labeled_weights):
  pool = Pool(processes = 2)

  build_classifer_labeled(classifier, labeled_data, targets, pool)

  projected_targets = None
  for i in xrange(MAX_ITER):
    # E step
    projected_targets = pool.map(classifier.predict_proba, unlabeled_data)

    # Check for convergence
#    if(projected_targets != None and 
#       all([np.allclose(a,b) for a,b in \
#              zip(new_projected_targets,projected_targets)])):
#      print("Converged")
#      break
#    projected_targets = new_projected_targets
#    new_projected_targets = None

    # M step
    build_classifer_mixed(classifier, labeled_data, targets,
                          unlabeled_data, projected_targets,
                          pool, labeled_weights)
  return classifier

def preprocess(data, vectorizer, processes):
  pool = Pool(processes)
  partitioned_data = list(chunks(data, len(data) / processes))
  mapped_data = pool.map(vectorizer.transform, partitioned_data)
  return scipy.sparse.vstack(mapped_data)

def partition_and_vectorize(data, vectorizer, processes):
  pool = Pool(processes)
  partitioned_data = list(chunks(data, len(data) / processes))
  mapped_data = pool.map(vectorizer.transform, partitioned_data)
  return pool.map(tocsr, mapped_data)

def test_em_times(num_labeled_docs, dg, processes_to_test, vectorizer,
               clf, labeled_weights, trials):
  for processes in processes_to_test:
    mr_times = []
    em_times = []
    offset = 0
    for i in range(trials):
      indices = random.sample(xrange(dg.X_labeled_csr.shape[0]),
                             num_labeled_docs)

def test_em(labeled_docs, processes, dg, vectorizer, clf,
            labeled_weight, trials):

  labeled = util.subsets(dg.labeled_data, dg.labeled_target,
                         labeled_docs/dg.num_classes, trials, percentage=False)

  accuracies_sup, accuracies_semi = [], []
  times_sup, times_semi = [], []
  for trial, (labeled_data, labeled_target) in enumerate(labeled):
    print 60 * "="
    print "ON TRIAL: " + str(trial + 1) + " OUT OF " + str(trials)
    print 60 * "="

#    X_labeled = scipy.sparse.vstack(labeled_data)
    X_labeled = vectorizer.transform(labeled_data)
    print(X_labeled.shape)

    prevTime = time.time()
    clf.fit(X_labeled, labeled_target)
    accuracies_sup.append(clf.score(dg.X_validate, dg.validate_target))
    times_sup.append(time.time() - prevTime)

    # Should really cache the partitions but oh well
    prevTime = time.time()
    partition_and_run(clf, X_labeled, labeled_target, dg.X_unlabeled,
                      processes, labeled_weight)
    accuracies_semi.append(clf.score(dg.X_validate,dg.validate_target))
    times_semi.append(time.time() - prevTime)

  print 60 * "="
  avg = lambda l: sum(l) / len(l)
  print "Average supervised accuracy: {:.2%}".format(avg(accuracies_sup))
  print "Average Semi-supervised accuracy: {:.2%}".format(avg(accuracies_semi))
  print 60 * "="

def test_times(labeled_docs, processes, dg, vectorizer, clf,
               labeled_weights, trials):
  pass

def main():
  dg = loader.NewsgroupGatherer()

  vectorizer = CountVectorizer(lowercase=True, stop_words='english',
                               max_df=.5, min_df=2, charset_error='ignore')
  vectorizer.fit(dg.unlabeled_data)
  print("Done fitting vectorizer\n")
  dg.vectorize(vectorizer)

  nb = MultinomialNB(alpha = 0.4)
  processes = 2
  labeled_weight = 2
  trials = 1
  labeled_test_size = 100

  test_em(labeled_test_size, processes, dg, vectorizer, nb, labeled_weight, trials)


if __name__ == '__main__':
  main()
