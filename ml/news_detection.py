import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Failed')
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import re
import pickle
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path
from math import trunc


BASE_DIR = Path(__file__).resolve().parent.parent

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    corpus = []
    review = re.sub('[^a-zA-Z]',' ',text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english') and len(word) > 2]
    review = ' '.join(review)
    corpus.append(review)
    return corpus

def tokenization(text):
    tokenize = pickle.load( open(str(BASE_DIR) + '/ml/tokenize.pkl', 'rb'))
    sequence = tokenize.texts_to_sequences(text)
    return sequence

def predict(text):
    info = text
    sequences = tokenization(lemmatization(info))
    sequences = pad_sequences(sequences, padding = 'post', truncating = 'post', maxlen = 900)
    sequences = np.array(sequences)
    len(sequences)
    model = tf.keras.models.load_model(str(BASE_DIR) + '/ml/news_model.h5')
    prediction = model.predict(sequences.reshape(-1,900))[0][0]
    prediction = float("{:.3f}".format(prediction))

    if prediction > 0.6:
        result = "Real"
        return result,prediction
    else:
        result = "Fake"
        return result,1-prediction