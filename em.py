from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from scipy import sparse
import numpy as np

import loader
import util

def em(clf, vectorizor, unlabeled, labeled, test, y_labeled, y_test,
       iterations=10, labeled_weight=2, mode="hard"):
  print "Running semi-supervised EM, mode = " + mode
  print "Building word counts..."
  vectorizor.fit(labeled + unlabeled)
  X_labeled = vectorizor.transform(labeled)
  X_unlabeled = vectorizor.transform(unlabeled)
  X_train = sparse.vstack([X_labeled, X_unlabeled])
  X_test = vectorizor.transform(test)
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

  print "Initializing with supervised prediction..."
  clf.fit(X_labeled, y_labeled)
  clf2.fit(X_labeled, y_labeled)
  supervised_accuracy = get_accuracy(clf, X_test, y_test)
  print "Supervised accuracy: {:.2%}".format(supervised_accuracy)

  # TODO: stopping criteria based on log-likelihood convergence
  for iteration in range(iterations):
    print "On iteration: " + str(iteration + 1) + " out of " + str(iterations)
    if mode == 'hard':
      ws = ([labeled_weight] * len(labeled)) + ([1] * len(unlabeled))
      predictions = clf.predict(X_unlabeled)
      clf.fit(X_train, np.append(y_labeled, predictions), ws)

    elif mode == 'soft':
      probas = clf.predict_proba(X_unlabeled)

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

def test_em(dg, labeled_size):
  trials = min((18000 / 5) / labeled_size, 5)
  
  clf = MultinomialNB(alpha=0.4)
  vectorizor = CountVectorizer(lowercase=True, stop_words='english',
    max_df=.5, min_df=2, charset_error='ignore')

  labeled = util.subsets(dg.labeled_data, dg.labeled_target,
                         labeled_size/20, trials, percentage=False)
  
  accuracies_sup, accuracies_semi = [], []
  for trial, (labeled_data, labeled_target) in enumerate(labeled):
    print 60 * "="
    print "ON TRIAL: " + str(trial + 1) + " OUT OF " + str(trials)
    print 60 * "="
    
    sup, semi = em(clf, vectorizor, dg.unlabeled_data,
                   labeled_data, dg.validate_data,
                   labeled_target, dg.validate_target,
                   mode='soft')

    accuracies_sup.append(sup)
    accuracies_semi.append(semi)

  print 60 * "="
  avg = lambda l: sum(l) / len(l)
  print "Average supervised accuracy: {:.2%}".format(avg(accuracies_sup))
  print "Average Semi-supervised accuracy: {:.2%}".format(avg(accuracies_semi))
  print 60 * "="

def main():
  labeled_sizes = [40, 80, 160, 300, 600, 1000, 2000, 3000]
  for size in labeled_sizes:
    test_em(loader.DataGatherer(), size)

if __name__ == '__main__':
  main()
