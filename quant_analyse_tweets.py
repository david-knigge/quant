import sys
import os
import re
import string
import requests
import csv
import json
import numpy as np
from datetime import datetime
from textblob import TextBlob
from datetime import datetime

def clean(tweet):

    split = tweet.split('||')

    match = re.search(r'\d{4}-\d{2}-\d{2}', tweet)
    if match == None:
        tweetdate = 'invalid'
    else:
        try:
            tweetdate = datetime.strptime(match.group(), '%Y-%m-%d').date()
            tweetdate = str(tweetdate).replace("-", "")
        except ValueError:
            tweetdate = 'invalid'

    tweetcontent = split[1]
    tweetcontent = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweetcontent, flags=re.MULTILINE)
    tweetcontent = ''.join(filter(lambda x: x in string.printable, tweetcontent))  # filter non-ascii characers
    tweetcontent = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweetcontent).split())

    return tweetdate, tweetcontent

def process_tweets():
    startTime = datetime.now()
    tweets = []
    with open('datasets/tweets_raw.txt', 'r',encoding="utf8") as raw:
        for tweet in raw:
            tweetdate, tweetcontent = clean(tweet)
            if tweetdate != 'invalid':
                tweets.append((tweetdate, tweetcontent))
    twittermatrix = np.array(['date', 'content', 'sentiment'])
    for tweet in tweets:
        try:
            sentiment = calculate_sentiment(tweet[1])
            if ((sentiment < 1) and (sentiment > -1)) and (tweet[1] != ''):
                twittermatrix = np.vstack([twittermatrix, [tweet[0], tweet[1], sentiment]])
        except Exception as e:
            pass
    with open("datasets/tweets_processed.csv", 'wb') as processed:
        np.savetxt(processed, twittermatrix, fmt='%s', delimiter=',')
    print(datetime.now() - startTime)

def calculate_sentiment(tweet):
    tweet = TextBlob(tweet)
    return tweet.sentiment.polarity


process_tweets()
#processedfile = open('datasets/tweets_processed.txt', 'r')
#print(processedfile.readline(2))
