from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.grid_search import IterGrid
from loader import DataGatherer

def find_best_vectorizor(vectorizer, grid):
  dg = DataGatherer()
  y_test = dg.validate_target
  y_train = dg.labeled_target

  nb = MultinomialNB()
  header_printed = False
  best_params = None
  best_score = -1
  for param in IterGrid(grid):
    if not header_printed:
      print(str(",".join(param.keys())) + ",Score")
    header_printed = True
    vectorizer.set_params(**param)
    X_train = vectorizer.fit_transform(dg.labeled_data)    
    X_test = vectorizer.transform(dg.validate_data)
    nb.fit(X_train, y_train)
    score = nb.score(X_test, y_test)
    if score > best_score:
      best_score = score
      best_params = param
    print(str(",".join(map(str, param.values()))) + "," + str(score))
  print("")
  print("Best params: " + str(best_params))
  print("Best score: " + str(best_score))

def find_best_tfidf_vectorizor():
  print("Searching for best tfidf vector")
  vectorizer = TfidfVectorizer()
  grid = {'lowercase' : [False, True],
          'stop_words' : ['english',None],
          'max_df' : [.9,.8,.7,.5],
          'min_df' : [0,2,5,10],
          'sublinear_tf' : [False, True]}
  find_best_vectorizor(vectorizer, grid)

def find_best_count_vectorizor():
  print("Searching for best count vector")
  vectorizer = CountVectorizer()
  grid = {'lowercase' : [False, True],
          'stop_words' : ['english',None],
          'max_df' : [.9,.8,.7,.5],
          'min_df' : [0,2,5,10]}
  find_best_vectorizor(vectorizer, grid)

def alpha_search(vectorizer):
  dg = DataGatherer()
  y_test = dg.validate_target
  y_train = dg.labeled_target
  X_train = vectorizer.fit_transform(dg.labeled_data)    
  X_test = vectorizer.transform(dg.validate_data)
  nb = MultinomialNB()
  alpha = 0.0
  best_score = -1
  best_alpha = None
  while(alpha < 5):
    alpha += 0.5
    nb.set_params(alpha = alpha)
    nb.fit(X_train, y_train)
    score = nb.score(X_test, y_test)
    if score > best_score:
      best_score = score
      best_alpha = alpha
    print(str(alpha) + "," + str(score))
  print("")
  print("Best alph: " + str(best_alpha))
  print("Best score: " + str(best_score))

def find_best_alpha():
#  vectorizer = TfidfVectorizer(lowercase=True,sublinear_tf=True,
#                               stop_words='english',max_df=.9,min_df=2)
  vectorizer = CountVectorizer(lowercase=True,stop_words='english',max_df=.5,min_df=2)

  alpha_search(vectorizer)

def run_tests():
  dg = DataGatherer()
  print("Extracting features from the training dataset")
  vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
  X_train = vectorizer.fit_transform(dg.labeled_data)
  y_train = dg.labeled_target

  X_test = vectorizer.transform(dg.validate_data)
  y_test = dg.validate_target

  for i in range(0,100):
    i = i / 100.0
    nb = MultinomialNB(alpha = i)
    nb.fit(X_train, y_train)
    print(str(i) + " " + str(nb.score(X_test, y_test)))

if __name__ == "__main__":
  find_best_alpha()
