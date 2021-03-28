# NewsClues
## _Get your news verified_

![image_0](https://github.com/pradyyadav/Images/blob/main/newsclus.png?raw=true)

Classifies the news given as input as real or fake and uses Word2Vec model for creating word embeddings and these embeddings are then used in the embedding layer as weights.

## Machine Learning Techniques
---
- Natural Language Processing
- Word Embeddings using Word2Vec
- Long Short Term Memory (LSTM) (RNN)

## Model Summary
---
![summary](https://github.com/pradyyadav/Images/blob/main/lstmnews.png?raw=true)

## Word2Vec
---
![w2v_image](https://github.com/pradyyadav/Images/blob/main/w2v.png?raw=true)

The purpose of *Word2vec* is to group the vectors of similar words together in vectorspace. That is, it detects similarities mathematically. *Word2Vec* creates vectors that are distributed numerical representations of word features, features such as the context of individual words.

## LSTM
---
![lstm](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images), but also entire sequences of data (such as speech or video). A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. 

Learn more about LSTM in this [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

## Instructions to run locally
---
- Clone this repository

```
$ git clone https://github.com/pradyyadav/NewsClues 
```


- Check Django Installation

``` 
$ python -m django --version 
```


- Install Django if not installed

``` 
$ python -m pip install Django
```


- Install all the dependencies

```
$ pip install -r requirements.txt
```


## Languages and Frameworks
---
- Python
- Django (Framework for Python)

## Libraries
---
### Machine Learning Libraries
- Numpy
- Pandas
- Scikit-Learn
- WordCloud
### Deep Learning Libraries
- Tensorflow
### Language Processing Libraries
- NLTK
- Gensim (For Word2Vec)


