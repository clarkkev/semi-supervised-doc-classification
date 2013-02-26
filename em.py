from loader import DataGatherer
from sklearn.feature_extraction.text import CountVectorizer

def vectorizor_transformed(data):
  vectorizor = CountVectorizer(lowercase=False, stop_words='english', charset_error='ignore')
  data = vectorizor.fit_transform(data)
  return vectorizor, data

def main():
  gatherer = DataGatherer()
  

if __name__ == '__main__':
  main()
