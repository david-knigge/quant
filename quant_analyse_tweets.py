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
    tweetcontent = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", tweetcontent).split())

    return tweetdate, tweetcontent

def process_tweets():
    tweets = []
    with open('datasets/tweets_raw.txt', 'r',encoding="utf8") as raw:
        for tweet in raw:
            tweetdate, tweetcontent = clean(tweet)
            if tweetdate != 'invalid':
                tweets.append((tweetdate, tweetcontent))
    twittermatrix = np.array(['date', 'content', 'sentiment'])
    startTime = datetime.now()
    for tweet in tweets[:1000000]:
        try:
            sentiment = calculate_sentiment(tweet[1])
            if ((sentiment < 1) and (sentiment > -1)) and (tweet[1] != ''):
                twittermatrix = np.vstack([twittermatrix, [tweet[0], tweet[1], sentiment]])
        except Exception as e:
            pass
    print(datetime.now() - startTime)
    print("done cleaning tweets")
    with open("datasets/tweets_clean.csv", 'wb') as processed:
        np.savetxt(processed, tweets, fmt='%s', delimiter=',')
    print(datetime.now() - startTime)

def calculate_sentiment(tweet):
    tweet = TextBlob(tweet)
    return tweet.sentiment.polarity

def sentiment_tweets():
    startTime = datetime.now()
    latest_time = datetime.now()
    count = 0

    twittermatrix = []
    with open("datasets/tweets_clean.csv", 'r') as tweets:
        for tweet in tweets:
            tweet = tweet.split(',')

            count += 1
            if count % 100 == 0:
                print(count)
                print(datetime.now() - latest_time)
                latest_time = datetime.now()

            try:
                sentiment = calculate_sentiment(tweet[1])
                if ((sentiment < 1) and (sentiment > -1)) and (tweet[1] != ''):
                    twittermatrix.append((tweet[0], tweet[1], sentiment))
            except Exception as e:
                print("ERROR: "+str(e))
    with open("datasets/tweets_sentiment.csv", 'wb') as processed:
        np.savetxt(processed, twittermatrix, fmt='%s', delimiter=',')
    print(datetime.now() - startTime)

def group_tweets():
    startTime = datetime.now()
    latest_time = datetime.now()

    count = 0

    twittermatrix = []
    with open("datasets/tweets_sentiment.csv", 'r') as tweets:
        day = tweets[0].split(',')[0]

        for tweet in tweets:
            tweet = tweet.split(',')
            if count == 0:
                day = tweet[0]

            count += 1
            if count % 100 == 0:
                print(count)
                print(datetime.now() - latest_time)
                latest_time = datetime.now()



#process_tweets()
#sentiment_tweets()
#processedfile = open('datasets/tweets_processed.txt', 'r')
#print(processedfile.readline(2))
