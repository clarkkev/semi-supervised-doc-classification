from sklearn.datasets import fetch_20newsgroups
from util import subset, subset_matrix, shuffle
import pickle
import random

class DataGatherer():
  def __init__(self):
    print "Loading data..."

    validate_size = 1.0 / 2
    labeled_size = 1.0 / 3

    self.labeled_data, self.labeled_target = \
      subset(self.train_data, self.train_target, 0, labeled_size)
    self.unlabeled_data, self.unlabeled_target = \
      subset(self.train_data, self.train_target, labeled_size, 1)
    self.validate_data, self.validate_target = \
      subset(self.alltest_data, self.alltest_target, 0, validate_size)
    self.test_data, self.test_target = \
      subset(self.alltest_data, self.alltest_target, validate_size, 1)

    self.X_labeled = self.X_unlabeled = self.X_validate = self.X_test = None

    self.size = len(self.train_data) + len(self.alltest_data)

    print len(self.labeled_data)
    print len(self.unlabeled_data)
    print len(self.validate_data)
    print len(self.test_data)
    print "Done loading data"

  def vectorize(self, vectorizer):
    print "Vectorizing..."
    self.X_unlabeled = vectorizer.fit_transform(self.unlabeled_data)
    self.X_labeled = vectorizer.transform(self.labeled_data)
    self.X_validate = vectorizer.transform(self.validate_data)
    self.X_labeled_csr = self.X_labeled.tocsr()

    #self.X_test = vectorizer.transform(self.test_data)

class DMOZGatherer(DataGatherer):
  def __init__(self):
    print "Loading data..."
    def load(fname):
      with open(fname) as f:
        return pickle.load(f)

    self.labeled_data, self.labeled_target = \
      load('data/dmoz_labeled')
    self.unlabeled_data, self.unlabeled_target = \
      load('data/dmoz_unlabeled')
    self.validate_data, self.validate_target = \
      load('data/dmoz_validate')
    self.test_data, self.test_target = \
      load('data/dmoz_test')

    self.X_labeled = self.labeled_data
    self.X_unlabeled = self.unlabeled_data
    self.X_validate = self.validate_data
    self.X_test = self.test_data

    self.size = len(self.labeled_target) + len(self.unlabeled_target) + \
      len(self.validate_target) + len(self.test_target)

    print len(self.labeled_target)
    print len(self.unlabeled_target)
    print len(self.validate_target)
    print len(self.test_target)
    
    self.num_classes = 30
    
    print "Done loading data"
    
class ReviewGatherer(DataGatherer):
  def __init__(self):
    alldata, alltargets = [], []
    with open('./data/review_data') as f:
      alldata = pickle.load(f)
    with open('./data/review_targets') as f:
      alltargets = pickle.load(f)

    p = range(len(alldata))
    random.seed(0)
    random.shuffle(p)
    shuffle = lambda l: [l[p[i]] for i in range(len(p))]
    alldata = shuffle(alldata)
    alltargets = shuffle(alltargets)

    train_size = 0.6
    self.train_data, self.train_target = \
      subset(alldata, alltargets, 0, 0.6)
    self.alltest_data, self.alltest_target = \
      subset(alldata, alltargets, 0.6, 1)

    self.num_classes = 3

    DataGatherer.__init__(self)

class NewsgroupGatherer(DataGatherer):
  def __init__(self):
    data_train = fetch_20newsgroups(subset='train', categories=None,
                                    shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=None,
                                   shuffle=True, random_state=42)
    self.train_data = data_train.data
    self.train_target = data_train.target
    self.alltest_data = data_test.data
    self.alltest_target = data_test.target

    self.categories = data_train.target_names
    self.num_classes = 20

    DataGatherer.__init__(self)

if __name__ == '__main__':
  print "Opening data files..."
  X, y = [], []
  with open('./data/dmoz_data') as f:
    X = pickle.load(f)
  with open('./data/dmoz_targets') as f:
    y = pickle.load(f)

  print y[:20]
  
  print "Shuffling..."
  p = range(len(y))
  random.seed(0)
  random.shuffle(p)
  shuffle = lambda l: [l[p[i]] for i in range(len(p))]
  y = shuffle(y)
  X = X[p]

  print "Loading data..."

  labeled_data, labeled_target = \
    subset_matrix(X, y, 0, 0.2)
  unlabeled_data, unlabeled_target = \
    subset_matrix(X, y, 0.2, 0.6)
  validate_data, validate_target = \
    subset_matrix(X, y, 0.6, 0.8)
  test_data, test_target = \
    subset_matrix(X, y, 0.8, 1)

  X_labeled = X_unlabeled = X_validate = X_test = None
  def dump (data, target, fname): 
    with open(fname, 'w') as f:
      pickle.dump((data, target), f)
  
  dump(labeled_data, labeled_target, 'data/dmoz_labeled')
  dump(unlabeled_data, unlabeled_target, 'data/dmoz_unlabeled')
  dump(validate_data, validate_target, 'data/dmoz_validate')
  dump(test_data, test_target, 'data/dmoz_test')

  print len(labeled_target)
  print len(unlabeled_target)
  print len(validate_target)
  print len(test_target)
  print "Done loading data"
