import transformers
import datasets
import shap
from shap.plots import *
import csv
import nltk
import re, string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import datetime
import torch
import random
from PIL import Image


# convert date from string to datetime object
def convert_date(tweets):
    for i in range(len(tweets)):
        tweets['date'][i] = datetime.datetime.strptime(tweets['date'][i], '%Y-%m-%d %H:%M:%S')
    return tweets
        
# select n random tweets from start date to end date
def choose_tweets_from_date(tweets, start, end, n):
    start = datetime.datetime.strptime(start, '%Y-%m-%d')
    end = datetime.datetime.strptime(end, '%Y-%m-%d')
    choices = []
    for i in range(len(tweets)):
        if start <= tweets['date'][i] <= end:
            el = tweets['content'][i]
            choices.append(el)
        
    chosen=random.sample(choices, n)

    return chosen

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\'','', text)
    
    return text

def load_and_preprocess_tweets():
    tweets = pd.read_csv('WFiIS-MIO-main/datasets/realdonaldtrump.csv')
    tweets = tweets.drop(['link', 'retweets', 'favorites', 'mentions', 'hashtags'], axis=1)

    tweets['content'] = tweets['content'].map(lambda x: re.sub(r'@\w+\s', ' ', x))
    tweets['content'] = tweets['content'].map(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
    tweets['content'] = tweets['content'].map(lambda x: re.sub(r'\W+', ' ', x))
    tweets['content'] = tweets['content'].replace(r'\W+', ' ', regex=True)
    tweets['content'] = tweets['content'].apply(lambda x: clean_text(x))
    tweets['token'] = tweets['content'].apply(lambda x: word_tokenize(x))
    tweets['text'] = tweets['token'].apply(lambda x: ' '.join([word for word in x if len(word)>2]))
    convert_date(tweets)

    return tweets

tweets = load_and_preprocess_tweets()

print (tweets)