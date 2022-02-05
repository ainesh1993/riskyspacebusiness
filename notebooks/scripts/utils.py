# basic data science packages
import numpy as np

# gensim packages for text preprocessing and LDA
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# nltk packages for text preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

stemmer = PorterStemmer()

def preprocess_string_pro(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result = result + ' ' + stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))
    return result.strip()

def preprocess_string_aug(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text, max_len=np.Inf):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result = result + ' ' + stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))
    return result.strip()

def preprocess_list(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
    return result