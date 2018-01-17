import sys
import os
import re
import string
import requests
import csv
import json
import numpy as np
import numbers
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from datetime import datetime

def clean(tweet):

    split = tweet.split('||')

    tweetdate = split[2]

    match = re.search(r'\d{4}-\d{2}-\d{2}', tweetdate)
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

def clean_tweets():
    tweets = []
    with open('datasets/tweets_raw.txt', 'r',encoding="utf8") as raw:
        counter = 0
        for tweet in raw:
            counter += 1
            tweetdate, tweetcontent = clean(tweet)
            if tweetdate != 'invalid':
                tweets.append((tweetdate, tweetcontent))
    return tweets

def calculate_sentiment(tweet):
    tweet = TextBlob(tweet)
    return tweet.sentiment.polarity

def sentiment_tweets(tweets):
    print(len(tweets))
    tweets_with_sentiment = {}

    count = 0

    for tweet in tweets:
        content = tweet[1]
        date = tweet[0]

        count += 1
        if count % 100 == 0:
            print(count)

        try:
            sentiment = calculate_sentiment(content)
            if ((sentiment < 1) and (sentiment > -1)) and (tweet[1] != ''):
                if date in tweets_with_sentiment.keys():
                    tweets_with_sentiment[date].append(sentiment)
                else:
                    tweets_with_sentiment[date] = [sentiment]
        except Exception as e:
            print("ERROR: " +str(e))

    return tweets_with_sentiment

def dict_day(tweets):
    dict_day = {}
    for key in tweets.keys():
        mean = np.mean(tweets[key])
        dict_day[key] = mean
    rows = zip(dict_day.keys(), dict_day.values())
    with open('datasets/tweets_with_sentiment.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

tweet_dict = sentiment_tweets(clean_tweets())
print(dict_day(tweet_dict))
#process_tweets()
#sentiment_tweets()
#processedfile = open('datasets/tweets_processed.txt', 'r')
#print(processedfile.readline(2))
