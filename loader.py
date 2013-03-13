from sklearn.datasets import fetch_20newsgroups
from util import subset

class DataGatherer():
  def __init__(self):
    print "Loading data..."

    data_train = fetch_20newsgroups(subset='train', categories=None,
                                    shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=None,
                                   shuffle=True, random_state=42)
    self.categories = data_train.target_names

    validate_size = 1.0 / 2
    labeled_size = 1.0 / 3
    
    self.labeled_data, self.labeled_target = \
      subset(data_train.data, data_train.target, 0, labeled_size)
    self.unlabeled_data, self.unlabeled_target = \
      subset(data_train.data, data_train.target, labeled_size, 1)
    self.validate_data, self.validate_target = \
      subset(data_test.data, data_test.target, 0, validate_size)
    self.test_data, self.test_target = \
      subset(data_test.data, data_test.target, validate_size, 1)

    #print len(self.labeled_data)
    #print len(self.unlabeled_data)
    #print len(self.validate_data)
    #print len(self.test_data)
    print "Done loading data"
