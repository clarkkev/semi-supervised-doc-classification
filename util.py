import pickle
from time import time
from operator import itemgetter

def subsets(data, target, size, n, percentage=True):
  return [subset(data, target, i * size, (i + 1) * size, percentage)
          for i in range(n)]

def subset(data, target, start, end, percentage=True):
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
