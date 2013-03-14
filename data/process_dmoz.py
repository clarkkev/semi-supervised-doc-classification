from collections import Counter
import pickle
import itertools

from sklearn.feature_extraction import DictVectorizer

import sys
sys.path.insert(0, '..')
from util import LoopLogger

NUM_CLASSES = 30 
NUM_EXAMPLES = 150000

print "Building tree hierarchy..."
class_counts = Counter()
with open('./track2/train/track2-DMOZ-train.txt') as f:
  for i, line in enumerate(f):
    class_counts[line.split(" ", 1)[0]] += 1
    if i > NUM_EXAMPLES: break

tree = {}
uptree = {}
all_children = set()
with open('./track2/track2-DMOZ-hierarchy.txt') as f:
  for line in f:
    parent, child = line.strip().split(" ")
    all_children.add(child)
    tree.setdefault(parent, []).append(child)
    uptree[child] = parent

def size(root, classes):
  s = 0
  if root in tree:
    for child in tree[root]:
      if child not in classes:
        s += size(child, classes)
  return class_counts[root] + s

def max_argmax(f, args):
  return max((f(a), a) for a in args)

 
classes = set(tree.keys()) - all_children
 
def all_nodes(root):
  s = set([root])
  if root in tree:
    for child in tree[root]:
      s |= all_nodes(child)
  return s

#for c in classes:
#  for c2 in classes:
#    if c == c2: continue
#    print len(all_nodes(c) & all_nodes(c2))
 
#depth2 = classes.copy()
#for root in classes:
#  depth2 |= set(tree[root])
#print len(depth2)

print "Picking classes..."
while len(classes) < NUM_CLASSES:
  largest_size, largest_class = max_argmax(lambda c: size(c, classes), classes) 
  #largest_child_size, largest_child = max_argmax(lambda c: size(c, classes), [child for child in tree[largest_class] if child not in classes]) 


  # The best line of python I have ever written
  all_children = list(itertools.chain(*[[child for child in tree[c] if child not in classes] 
                                         for c in classes if c in tree]))
  largest_child_size, largest_child = max_argmax(lambda c: size(c, classes), all_children)

  print 60 * "="
  for root in sorted(list(classes)):
    print root, size(root, classes)
  print
  print largest_class, largest_size
  print largest_child, largest_child_size
  print 60 * "="
  
  classes.add(largest_child)

def get_class(node):
  if node in classes:
    return node
  return get_class(uptree[node])

class_mapping = {}
for i, c in enumerate(sorted(list(classes))):
  class_mapping[c] = i


print "Getting labeled data..."
examples = []
targets = []
ll = LoopLogger(1000, NUM_EXAMPLES, True)


occurances = Counter()
totals = Counter()
with open('./track2/train/track2-DMOZ-train.txt') as f:
  for i, line in enumerate(f):
    ll.step()
    split = line.split(" ", 1)
    targets.append(class_mapping[get_class(split[0])])

    integerize = lambda (s1, s2): (int(s1), int(s2))
    vector = dict([integerize(pair.split(":")) for pair in split[1].split(" ")])
    for token in vector:
      occurances[token] += 1
      totals[token] += vector[token]

    examples.append(vector)

    if i > NUM_EXAMPLES: break

bad_tokens = set()
bad = 0
for token in occurances:
  if occurances[token] <= 2:
    bad_tokens.add(token) 
  if occurances[token] > 5000:
    print token, occurances[token]

print len(occurances), len(bad_tokens)
for vector in examples:
  for token in vector.keys():
    if token in bad_tokens:
      del vector[token]
  #print i,  occurances[i], totals[i]

print "Vectorizing..."
DV = DictVectorizer()
X = DV.fit_transform(examples)
del examples

print "Dumping results..."

extra = ""
with open('./dmoz_data' + extra, 'w') as f:
  pickle.dump(X, f)

with open('./dmoz_targets' + extra, 'w') as f:
  pickle.dump(targets, f)
