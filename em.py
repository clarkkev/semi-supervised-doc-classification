from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import BallTree, KNeighborsClassifier


from sklearn.pipeline import Pipeline

from scipy import sparse
import numpy as np

from collections import Counter

from heapq import nsmallest

from operator import itemgetter
import loader
import util

def dist(x1, x2):
  return 1 + np.sum((x1 - x2) ** 2)

class KNN:
  def __init__(self):
    self.pts = []
    self.k = 3
    
  def add_pt(self, x, y):
    self.pts.append((x, y))  

  def classify(self, x):
    closest = self.closest_k(x, self.k)
    class_counts = Counter()
    for d, (x2, y) in closest:
      class_counts[y] += 1
    return max((count/ float(self.k), y) for y, count in class_counts.iteritems())


  def closest_k(self, x, k):
    ds = [(dist(x, x2), (x2, y)) for x2, y in self.pts]
    return nsmallest(k, ds, key=itemgetter(0))

  def get_clf(self):
    xs, ys = zip(*self.pts)
    X = np.vstack(xs)
    clf = KNeighborsClassifier(3)
    clf.fit(X, ys)
    return clf


def knn(X_unlabeled, X_labeled, X_test, y_labeled, y_test):
  def get_xs(X):
    X = X.toarray()
    return [X[i,:] for i in range(X.shape[0])]
  
  Xs_labeled = get_xs(X_labeled)
  Xs_unlabeled = get_xs(X_unlabeled)
  Xs_test = get_xs(X_test)

  '''print "Building point set"
  C = KNN()
  ll = util.LoopLogger(100, len(Xs_labeled), True)
  for x, y in zip(Xs_labeled, y_labeled):
    if not C.pts:
      C.add_pt(x, y)
    else:
      score, pred = C.classify(x)
      if pred != y:
        C.add_pt(x, y)
  print len(C.pts)
  print len(y_labeled)'''


  '''print "Adding unsupervised examples"
  ll = util.LoopLogger(100, len(Xs_unlabeled), True)
  for x in Xs_unlabeled:
    ll.step()
    p = C.classify(x)
    score, pred = C.classify(x)
    if score > 0.6:
      C.add_pt(x, pred)
  print len(C.pts)
  print len(y_labeled) + len(Xs_unlabeled)'''
  
  print "Training on labeled data..."
  clf = KNeighborsClassifier(n_neighbors=3)
  clf.fit(X_labeled, y_labeled)
  
  supervised_accuracy = get_accuracy(clf, X_test, y_test)
  print "Supervised accuracy: {:.2%}".format(supervised_accuracy)


  print "Adding unlabeled points..."
  probas = clf.predict_proba(X_unlabeled)
  X_indices = []
  ys = []
  for i in range(probas.shape[0]):
    instance_probas = probas[i,:]
    best = np.argmax(instance_probas)
    best_score = instance_probas[best]
      
    if best_score > 0.9:
      X_indices.append(i)
      ys.append(best)
  
  ys = y_labeled + ys
  Xs = np.vstack((X_labeled.toarray(), X_unlabeled[X_indices,:].toarray()))

  print "Condensing..."
  '''C = KNN()
  ll = util.LoopLogger(100, len(Xs_labeled), True)
  for x, y in zip(Xs, ys):
    if not C.pts:
      C.add_pt(x, y)
    else:
      score, pred = C.classify(x)
      if pred != y:
        C.add_pt(x, y)
  print len(C.pts)'''
  #clf = C.get_clf()

  print "Training semi-supervised"
  clf.fit(Xs, ys)
  
  semi_supervised_accuracy = get_accuracy(clf, X_test.toarray(), y_test)
  print "Semi-supervised accuracy: {:.2%}".format(semi_supervised_accuracy)
  
  return supervised_accuracy, semi_supervised_accuracy
  #print "Condensing..."
  #C2 = KNN()
  #for x, y in C.pts:
  #  if not C2.pts:
  #    C2.add_pt(x, y)
  #  else:
  #    score, pred = C2.classify(x)
  #    if pred != y:
  #      C2.add_pt(x, y)
  #print len(C2.pts)

  #C3 = KNN()
  #for x, y in zip(Xs_labeled, y_labeled):
  #  C3.add_pt(x, y)

  #print get_accuracy(C.get_clf(), X_test, y_test)
  #print get_accuracy(C2.get_clf(), X_test, y_test)
  #print get_accuracy(C3.get_clf(), X_test, y_test)


def em(clf, X_unlabeled, X_labeled, X_test, y_labeled, y_test,
       iterations=10, labeled_weight=2, mode="hard"):
  print "Running semi-supervised EM, mode = " + mode
  print "Building word counts..."

  X_train = sparse.vstack([X_labeled, X_unlabeled])
  num_classes = max(max(y_test), max(y_labeled))

  if mode == 'SFE':
    print "Computing p(c|w) for each word..."
    total_word_counts = np.zeros(X_labeled.shape[1])
    class_given_word = np.zeros((20, X_labeled.shape[1]))
    labeled_csr = X_labeled.tocsr()
    for i, y in enumerate(y_labeled):
      row = np.asarray(labeled_csr.getrow(i).todense())[0]
      total_word_counts += row
      class_given_word[y] += row
    class_word_counts = np.copy(class_given_word)

    smoothing = 0.001
    total_word_counts = np.maximum(total_word_counts, 1)
    for c in range(num_classes):
      class_given_word[c] = (class_given_word[c] + smoothing) \
        / (total_word_counts + (num_classes + 1) * smoothing)

    clf2 = MultinomialNB(alpha=0.0001)
    clf2.fit(X_labeled, y_labeled)

  print "Initializing with supervised prediction..."
  clf.fit(X_labeled, y_labeled)
  supervised_accuracy = get_accuracy(clf, X_test, y_test)
  print "Supervised accuracy: {:.2%}".format(supervised_accuracy)

  # TODO: stopping criteria based on log-likelihood convergence
  for iteration in range(iterations):
    print "On iteration: " + str(iteration + 1) + " out of " + str(iterations)
    if mode == 'hard':
      ws = ([labeled_weight] * X_labeled.shape[0]) + ([1] * X_unlabeled.shape[0])
      predictions = clf.predict(X_unlabeled)
      clf.fit(X_train, np.append(y_labeled, predictions))#, ws)

    elif mode == 'soft':
      probas = clf.predict_proba(X_unlabeled)

      #ys = range(num_classes)
      #Xs = np.zeros((num_classes, X_labeled.shape[1]))

      Xs = sparse.vstack([X_labeled] + ([X_unlabeled] * num_classes))
      ys = y_labeled[:]
      ws = [int(labeled_weight * 10000)] * len(y_labeled)
      for c in range(num_classes):
        for i in range(probas.shape[0]):
          ys.append(c)
          ws.append(int(10000 * probas[i, c]))
      clf.fit(Xs, ys, ws)

    elif mode == 'SFE':
      num_words = X_train.shape[1]
      word_matrix = sparse.coo_matrix(([1] * num_words,
       (range(num_words), range(num_words))))

      probas = clf2.predict_proba(word_matrix)
      probas *= num_classes

      X_matrices = [X_labeled]
      for c in range(num_classes):
        word_probas_2 = probas[:,c]
        word_probas = class_given_word[c]

        #for i in range(len(word_probas)):
        #  print word_probas[i], word_probas_2[i]
        #  print total_word_counts[i], class_word_counts[c][i]
        #  print
        #return

        if iteration == 0:
          X_matrices.append((X_unlabeled * \
            sparse.diags([[1.0/20] * num_words], [0])).tocoo())
            #sparse.diags([word_probas], [0])).tocoo())
        else:
          X_matrices.append((X_unlabeled * \
            sparse.diags([word_probas_2], [0])).tocoo())

      Xs = sparse.vstack(X_matrices)
      ys = y_labeled + (range(num_classes) * X_unlabeled.shape[0])
      #ws = ([num_classes * labeled_weight] * len(y_labeled)) + ([1] * num_classes * X_unlabeled.shape[0])
      ws = ([labeled_weight] * len(y_labeled)) + ([1] * num_classes * X_unlabeled.shape[0])
      clf.fit(Xs, ys, ws)
      clf2.fit(Xs, ys, ws)

    else:
      print "MODE NOT RECOGNIZED"
      return

    semi_supervised_accuracy = get_accuracy(clf, X_test, y_test)
    print "Semi-supervised accuracy: {:.2%}".format(semi_supervised_accuracy)

  return supervised_accuracy, semi_supervised_accuracy

def get_accuracy(clf, X_test, y_test):
  return metrics.precision_score(y_test, clf.predict(X_test))

def em_runner(clf, mode):
  def run(X_unlabeled, X_labeled, X_test, y_labeled, y_test):
    return em(clf, X_unlabeled, X_labeled, X_test, y_labeled, y_test, mode=mode)
  return run


def test_method(dg, labeled_size, trials, run_fun):
  labeled = util.subsets_matrix(dg.X_labeled, dg.labeled_target,
                                labeled_size/dg.num_classes, trials, percentage=False)

  accuracies_sup, accuracies_semi = [], []
  for trial, (X_labeled, y_labeled) in enumerate(labeled):
    print 60 * "="
    print "ON TRIAL: " + str(trial + 1) + " OUT OF " + str(trials)
    print 60 * "="

    sup, semi = run_fun(dg.X_unlabeled, X_labeled, dg.X_validate,
                        y_labeled, dg.validate_target)

    accuracies_sup.append(sup)
    accuracies_semi.append(semi)

  print 60 * "="
  avg = lambda l: sum(l) / len(l)
  print "Average supervised accuracy: {:.2%}".format(avg(accuracies_sup))
  print "Average Semi-supervised accuracy: {:.2%}".format(avg(accuracies_semi))
  print 60 * "="

def main():
  clf = MultinomialNB(alpha=0.4)
  #vectorizer = CountVectorizer(lowercase=True, stop_words='english',
  #  max_df=.5, min_df=2, charset_error='ignore')
  hasher = HashingVectorizer(lowercase=True, stop_words='english',
    n_features=10000, norm=None, binary=False, non_negative=True, charset_error='ignore')
  vectorizer = Pipeline((('hasher', hasher),
                         ('tf_idf', TfidfTransformer())))

  #dg = loader.DMOZGatherer()
  dg = loader.NewsgroupGatherer()
  dg.vectorize(vectorizer)

  labeled_sizes = [80, 160, 300, 600, 1000, 2000, 3000]
  for size in labeled_sizes:
    print
    print "---TESTING FOR " + str(size) + " LABELED EXAMPLES---"
    #test_method(dg, size, min(int(dg.size * 0.2 * 0.9 / size), 10), em_runner(clf, 'hard'))
    test_method(dg, size, min(int(dg.size * 0.2 * 0.9 / size), 5), knn)
    print

if __name__ == '__main__':
  main()
