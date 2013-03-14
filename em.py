from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from scipy import sparse
import numpy as np

import loader
import util

def add_row(m, r, c, w, priors):
  m[c] += r * w
  priors[c] += w


def em(clf, X_unlabeled, X_labeled, X_test, y_labeled, y_test,
       iterations=10, labeled_weight=2, mode="hard"):
  print "Running semi-supervised EM, mode = " + mode
  print "Building word counts..."

  #vectorizor.fit(labeled + unlabeled)
  #X_labeled = vectorizor.transform(labeled)
  #X_unlabeled = vectorizor.transform(unlabeled)
  #X_train = sparse.vstack([X_labeled, X_unlabeled])
  #X_test = vectorizor.transform(test)

  #X_train = sparse.vstack([X_labeled, X_unlabeled])
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

  #clf2 = MultinomialNB(alpha=0.0001)

  print "Initializing with supervised prediction..."
  clf.fit(X_labeled, y_labeled)
  #clf2.fit(X_labeled, y_labeled)
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

      ys = range(num_classes)
      Xs = np.zeros((num_classes, X_labeled.shape[1]))
      

      #Xs = sparse.vstack([X_labeled] + ([X_unlabeled] * num_classes))
      #ys = y_labeled[:]
      #ws = [int(labeled_weight * 10000)] * len(y_labeled)
      #for c in range(num_classes):
      #  for i in range(probas.shape[0]):
      #    ys.append(c)
      #    ws.append(int(10000 * probas[i, c]))
      #clf.fit(Xs, ys, ws)

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

# for labeled
# go class to list of 1 x n matrices
# run get_subset
# then stack matrices to get labeled matrix

def test_em(dg, labeled_size, clf, vectorizer, num_classes, trials):
  #X_labeled = dg.X_labeled_csr
  #X_labeled_list = [X_labeled.getrow(i) for i in range(X_labeled.shape[0])]
  #y_labeled_list = list(dg.labeled_target)
  #labeled = util.subsets(X_labeled_list, y_labeled_list,
  #                       labeled_size/num_classes, trials, percentage=False)


  #labeled = util.subsets(dg.labeled_data, dg.labeled_target,
  #                       labeled_size/num_classes, trials, percentage=False)
  #labeled = util.subsets_matrix(dg.X_labeled, dg.labeled_target,
  #                              labeled_size/num_classes, trials, percentage=False)
  labeled = util.subsets_matrix(dg.X_labeled, dg.labeled_target,
                                0.3, trials, percentage=True)
  #labeled = [(dg.X_labeled, dg.labeled_target)]
  print dg.X_labeled.shape

  accuracies_sup, accuracies_semi = [], []
  for trial, (labeled_data, labeled_target) in enumerate(labeled):
    print 60 * "="
    print "ON TRIAL: " + str(trial + 1) + " OUT OF " + str(trials)
    print 60 * "="

    #X_labeled = sparse.vstack(labeled_data)
    #X_labeled = vectorizer.transform(labeled_data)
    X_labeled = labeled_data

    sup, semi = em(clf, dg.X_unlabeled,
                   X_labeled, dg.X_validate,
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
  #clf = LogisticRegression()
  clf = MultinomialNB(alpha=0.4)
  vectorizer = CountVectorizer(lowercase=True, stop_words='english',
    max_df=.5, min_df=2, charset_error='ignore')
  #vectorizor = HashingVectorizer(lowercase=True, stop_words='english',
  #  n_features=1000, norm=None, non_negative=True, charset_error='ignore')

  dg = loader.DMOZGatherer()
  #dg = loader.NewsgroupGatherer()
  #dg.vectorize(vectorizer)
  print dg.X_unlabeled.shape

  labeled_sizes = [1000]#300, 80, 160, 300, 600, 1000, 2000, 3000]
  for size in labeled_sizes:
    print
    print "---TESTING FOR " + str(size) + " LABELED EXAMPLES---"
    test_em(dg, size, clf, vectorizer, dg.num_classes, min(int(dg.size * 0.2 * 0.9 / size), 3))
    print

  #test_em(dg, 200)

if __name__ == '__main__':

  main()
