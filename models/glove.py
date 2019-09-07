'''Module to load and use GloVe Models.

Code Inspiration from:
https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
'''

import numpy as np
import urllib.request
from zipfile import ZipFile
from sklearn.base import BaseEstimator, TransformerMixin

def download(name):
  '''Downloads the relevant dataset and extracts it.

  Args:
    name (str): Name of the model to download (options are: [twitter, wikipedia])

  Returns:
    True if successful, otherwise False
  '''
  url = None
  if name == 'twitter':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip'
  elif name == 'wikipedia':
    url = 'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'
  if url is not None:
    try:
      urllib.request.urlretrieve(url, '../models/{}.zip'.format(name))
    except:
      print("download failed")
      return False
    try:
      # Create a ZipFile Object and load sample.zip in it
      with ZipFile('../models/{}.zip'.format(name), 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall('../models')
      return True
    except:
      print("extraction failed")
      return False
  return False

class GloveEmbeddings:
  '''Class to load embeddings model and generate it for words or sentences.'''
  def __init__(self, name, dim=25):
    # load data
    self.emb = self.load_vectors(name, dim)
    self.emb_size = dim
    # calculate items for randomization
    all_embs = np.stack(self.emb.values())
    self.emb_mean,self.emb_std = all_embs.mean(), all_embs.std()

  def get_coefs(self, word, *arr):
    '''Helper Function to transform the given vector into a float array.'''
    return word, np.asarray(arr, dtype='float32')

  def load_vectors(self, name, dim):
    '''Load the given vector data.'''
    # TODO: additional error checks
    file = '../models/glove.{}.27B.{}d.txt'.format(name, dim)
    # load the embeddings
    embeddings_index = dict(self.get_coefs(*o.strip().split()) for o in open(file, encoding='utf-8'))
    return embeddings_index

  def word_vector(self,word):
    '''Tries to retrieve the embedding for the given word, otherwise returns random vector.'''
    # generate randomness otherwise
    vec = self.emb.get(word)
    return vec if vec is not None else np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))

  def sent_vector(self, sent, use_rand=True):
    '''Generates a single embedding vector.

    Args:
      sent (list): List of tokenized words to use
      use_rand (bool): Defines if unkown words should be filled with random vectors (otherwise only use known vectors)

    Returns:
      Single normalized Vector to be used as embedding
    '''
    vec = None
    vec_count = 0
    for word in sent:
      wvec = self.emb.get(word)
      if wvec is None and use_rand:
        wvec = np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))
      if wvec is not None:
        if vec is None:
          vec = wvec
        else:
          vec += wvec
        vec_count += 1
    # normalize the vector
    if vec is not None:
      vec = vec / vec_count
    # if no word is found return random vector
    return vec if vec is not None else np.random.normal(self.emb_mean, self.emb_std, (self.emb_size))

  def sent_matrix(self, sent, max_feat, pad):
    '''Generates a Matrix of single embeddings for the item.

    Args:
      sent (list): List of tokenized words
      max_feat (int): Number of maximal features to extract
      pad (bool): Defines if the resulting matrix should be zero-padded to max_feat

    Returns:
      2-D Matrix with dimensions [max_feat, embedding_size]
    '''
    nb_words = min(max_feat, len(sent))
    embedding_matrix = np.random.normal(self.emb_mean, self.emb_std, (nb_words, self.emb_size))
    # iterate through all words
    for i, word in enumerate(sent):
      if i >= max_feat: continue
      vec = self.emb.get(word)
      if vec is not None: embedding_matrix[i] = vec
    # pad the matrix to max features
    if pad and nb_words < max_feat:
      pass
    return embedding_matrix

class GloVeTransformer(BaseEstimator, TransformerMixin):
  '''Transformer for the GloVe Model.'''

  def __init__(self, name, dim, type, tokenizer, max_feat=None):
    '''Create the Transformer.

    Args:
      name (str): Name of the model
      dim (int): Number of dimensions to use
      type (str): Type of the transformation (options are: ['word', 'sent', 'sent-matrix'])
      tokenizer (fct): Function to tokenize the input data
    '''
    self.glove = GloveEmbeddings(name, dim)
    self.type = type
    self.tokenizer = tokenizer
    self.max_feat = max_feat

  def fit(self, x, y=None):
    return self

  def vectors(self, text):
    # retrieve the vectors
    tokens = self.tokenizer(text)
    if self.type == 'word':
      return [self.glove.word_vector(tok) for tok in tokens]
    elif self.type == 'sent':
      return self.glove.sent_vector(tokens)
    elif self.type == 'sent-matrix':
      return self.glove.sent_matrix(tokens, self.max_feat)
    return np.nan

  def transform(self, X):
    X_tagged = pd.Series(X).apply(self.vectors, expand=True)
    return pd.DataFrame(X_tagged)
