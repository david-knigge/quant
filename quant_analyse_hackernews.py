from requests import get
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
from datetime import datetime
import pandas as pd
from pandas import DataFrame
import json
import html2text

def data(date):
    news = []
    with open('hackernews/'+date, 'rb') as f:
        file_content = [s.strip() for s in (f.read()).splitlines()]

    bitcoin_hackernews = []
    terms = ['bitcoin', 'btc', 'Bitcoin', 'BTC', 'BITCOIN']
    for f in file_content:
        for term in terms:
            if term in f:
                bitcoin_hackernews.append(f)

    news.append(bitcoin_hackernews)

    return bitcoin_hackernews

def clean(news):
    for hack_list in bitcoin_hackernews_17:
        for hl in hack_list:
            dict_hl = json.loads(hl)
            text = dict_hl["text"]
            time = dict_hl["time"]

    return time, text

def format_html(html_text):
    return html2text.html2text(html_text)

def calculate_sentiment(text):
    sent = TextBlob(text)
    return sent.sentiment.polarity

def make_dataframe(news):
    texts, times, sents = [], [], []
    for hl in news:
        dict_hl = json.loads(hl)
        text, time = dict_hl.get("text", ""), dict_hl.get("time", 0)
        if text:
            texts.append(format_html(text))
            times.append(time)
            sents.append(calculate_sentiment(text))

    dataframe = pd.DataFrame({'Time': times, 'Text': texts, 'Sentiment': sents})

    return dataframe

list_17 = ['HN_2017-01', 'HN_2017-02', 'HN_2017-03', 'HN_2017-04', 'HN_2017-05',
 'HN_2017-06', 'HN_2017-07']

for date in list_17:
    news_17 = data(date)
    dict_nt = make_dataframe(news_17)
    print(dict_nt)
