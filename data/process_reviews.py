import json
import pickle

relabel = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
data = []
targets = []
with open('./reviews') as f:
  reviews  = json.loads(f.read())
  for p, r in reviews.iteritems():
    targets.append(relabel[int(p[-1])])
    data.append(r)

with open('./review_data', 'w') as f:
  pickle.dump(data, f)

with open('./review_targets', 'w') as f:
  pickle.dump(targets, f)
