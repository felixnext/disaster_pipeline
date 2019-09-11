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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# storage
import pickle

# Load custom functions from the model folder
import sys, os
sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__))))
import glove
import ml_helper as utils

def load_data(database_filepath):
  '''Loads the relevant data for the model training.'''
  # download glove data
  if not glove.download('twitter'):
    print("ERROR: Download of GloVe Data failed. Please download and extract manually to the ./models directory from http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip")
  # laad remaining data
  engine = create_engine('sqlite:///{}'.format(database_filepath))
  df = pd.read_sql_table('texts', engine)
  X = df[['message', 'original', 'genre']]
  Y = df.iloc[:, 4:]
  return X['message'], Y, Y.columns

def build_model():
  '''Creates the pipeline with pre-processing and classifier.'''
  pipeline = Pipeline([
    ('features', FeatureUnion([
      ('term_emb', Pipeline([
        ('vectorize', CountVectorizer(tokenizer=utils.tokenize_clean, max_df=0.5)),
        ('tfidf', TfidfTransformer(use_idf=False)),
      ])),
      ('glove', Pipeline([
        ('glove_emb', glove.GloVeTransformer('twitter', 25, 'centroid', tokenizer=utils.tokenize_clean, max_feat=5))
      ]))
    ])),
    ('cls', MultiOutputClassifier(LogisticRegression(max_iter=50, C=2.0), n_jobs=-1) )
  ])
  return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
  '''Evaluate the model.'''
  utils.score_and_doc(model, "deploy_final", X_test, Y_test, extended=True)

def save_model(model, model_filepath):
  '''Store the model as serialized data.'''
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
