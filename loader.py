from sklearn.datasets import fetch_20newsgroups

class DataGatherer():
  def __init__(self):
    print "Loading data..."

    data_train = fetch_20newsgroups(subset='train', categories=None,
                                    shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=None,
                                   shuffle=True, random_state=42)
    categories = data_train.target_names

    validate_size = len(data_test.data) / 2
    labeled_size = len(data_train.data) / 3
    
    self.train_data = data_train.data
    self.labeled_data = data_train.data[:labeled_size]
    self.labeled_target = data_train.target[:labeled_size]
    self.unlabeled_data = data_train.data[labeled_size:]
    self.unlabeled_target = data_train.target[labeled_size:]
    self.validate_data = data_test.data[:validate_size]
    self.validate_target = data_test.target[:validate_size]
    self.test_data = data_test.data[validate_size:]
    self.test_target = data_test.target[validate_size:]

    print "Done loading data"
