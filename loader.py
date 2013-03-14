from sklearn.datasets import fetch_20newsgroups
from util import subset, shuffle
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

    print len(self.labeled_data)
    print len(self.unlabeled_data)
    print len(self.validate_data)
    print len(self.test_data)
    print "Done loading data"

  def vectorize(self, vectorizer):
    print "Vectorizing..."
    self.X_unlabeled = vectorizer.fit_transform(self.unlabeled_data)
    #self.X_labeled = vectorizer.transform(self.labeled_data)
    self.X_validate = vectorizer.transform(self.validate_data)
    #self.X_test = vectorizer.transform(self.test_data)

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
    
    DataGatherer.__init__(self)
