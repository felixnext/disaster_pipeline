import sys
# general + data loading
import pandas as pd
from sqlalchemy import create_engine

# pre-processing
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# machine learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# storage
import pickle
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
# generate stop words
stops = set(stopwords.words('english'))

def load_data(database_filepath):
  engine = create_engine('sqlite:///{}'.format(database_filepath))
  df = pd.read_sql_table('texts', engine)
  X = df[['message', 'original', 'genre']]
  Y = df.iloc[:, 4:]
  return X['message'], Y, Y.columns

def tokenize(text):
  # remove punctuation
  text = re.sub("[\.\\:;!?'\"-]", " ", text.lower())
  tokens = word_tokenize(text)

  # remove stopwords
  tokens = [tok for tok in tokens if tok not in stops]

  # part of speech
  tags = pos_tag(tokens)

  # TODO: further processing (e.g. NER)
  return tags

def build_model():
  pipeline = [
      ('vectorize', CountVectorizer(tokenizer=tokenize)),
      ('tfidf', TfidfTransformer()),
      ('cls', MultiOutputClassifier(SVC(), n_jobs=1) )
  ]
  pipeline = Pipeline(pipeline)
  return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
  y_pred = model.predict(X_test)
  score = f1_score(Y_test, y_pred, average='micro')
  print("MODEL PERFORMED WITH: {:.6f}".format(score))

def save_model(model, model_filepath):
  pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
