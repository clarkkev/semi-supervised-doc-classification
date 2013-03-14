import pickle
import random
from time import time
from operator import itemgetter
from scipy import sparse


def subset_matrix(data, target, start, end, percentage=True):
  index_list = range(data.shape[0])
  index_subset, target_subset = subset(index_list, target, start, end, percentage)
  #print "getting rows..."
  rows = [data.getrow(i) for i in sorted(index_subset)]
  #print "done"
  return sparse.vstack(rows).tocoo(), target_subset


def subsets_matrix(data, target, size, n, percentage=True):
  #if percentage:
  #  return variable_subsets_matrix(data, target)
  return [subset_matrix(data.tocsr(), target, i * size, (i + 1) * size, percentage)
          for i in range(n)]

'''def variable_subsets_matrix(data, target, points):
  index_list = range(data.shape[0])
  for i, y in enumerate(target):
    indices_by_class.setdefault(y, []).append(i)
  
  subsets_list = []
  for i in range(len(points - 1)):
    subsets_list.append(subset(index_list, target, points[i], points[i + 1], True, indices_by_class))

  matrix_subsets_list = []
  for index_subset, target_subset in subsets_list:
    rows = [data.getrow(i) for i in index_subset]
    matrix_subsets_list.append((sparse.vstack(rows), target_subset))

  return matrix_subsets_list'''


def subsets(data, target, size, n, percentage=True):
  indices_by_class = {}
  for i, y in enumerate(target):
    indices_by_class.setdefault(y, []).append(i)

  return [subset(data, target, i * size, (i + 1) * size, percentage)
          for i in range(n), indices_by_class]

def subset(data, target, start, end, percentage=True, indices_by_class=None):
  if not indices_by_class:
    indices_by_class = {}
    for i, y in enumerate(target):
      indices_by_class.setdefault(y, []).append(i)

  new_data, new_target = [], []
  for y in indices_by_class:
    n = len(indices_by_class[y]) if percentage else 1
    for i in indices_by_class[y][int(start * n):int(end * n)]:
      new_data.append(data[i])
      new_target.append(y)

  return new_data, new_target

def shuffle(X, y, examples):
  p = range(len(y))
  random.seed(0)
  random.shuffle(p)
  return X[p], y[p], [examples[p[i]] for i in range(len(p))]

class LoopLogger():
  def __init__(self, step_size, size=0, print_time=False):
    self.step_size = step_size
    self.size = size
    self.n = 0
    self.print_time = print_time

  def step(self):
    if self.n == 0:
      self.start_time = time()

    self.n += 1
    if self.n % self.step_size == 0:
      if self.size == 0:
        print 'On item ' + str(self.n)
      else:
        print 'On item ' + str(self.n) + ' out of ' + str(self.size)
        if self.print_time and (self.n % (self.step_size * 10)) == 0:
          time_elapsed = time() - self.start_time
          print "Time elapsed: " + str(time_elapsed)
          time_per_step = time_elapsed / self.n
          print "Time remaining: " + str((self.size - self.n) * time_per_step)
