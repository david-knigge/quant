import numpy as np
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta, date
from itertools import chain
from scipy.interpolate import interp1d

# Plot a graph
def plot_graph(x, y, title):
    plt.plot(x, y, 'ro')
    plt.title(title)
    plt.show()

# API for the bitcoin interest
pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Bitcoin"]
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

i_df = pytrends.interest_over_time()
interests = i_df.drop(labels='isPartial', axis=1)
interests_df = pd.DataFrame(interests)
interests_week = list(interests_df['Bitcoin'])

plot_graph(np.array(range(len(interests_week))), np.array(interests_week), 'Interest per week')

# Create linear points between two points
def linear_function(A, B):
    diff = (B - A)/6
    linearlist = []
    for i in range(1,7):
        linearlist.append(A+diff*i)
    return linearlist

# Create the interests per day list
newlist = []
for i in range(len(interests_week)-1):
    point1, point2 = interests_week[i], interests_week[i+1]
    linearlist = [point1]  + linear_function(point1, point2)
    newlist.append(linearlist)

interests_day = list(chain.from_iterable(newlist))
interests_day.append(interests_week[-1])
interests_day = [interests_week[0]]+interests_day

plot_graph(np.array(range(len(interests_day))), np.array(interests_day), 'Interest per day')

# Create date list from 2013-2018
def daterange(date1, date2):
    dates = []
    for n in range(int ((date2 - date1).days)+1):
        dates.append(date1 + timedelta(n))
    return dates

dates = daterange(date(2013, 1, 13), date(2018, 1, 9))
dates_day = [(pd.to_datetime(str(d))).strftime('%Y.%m.%d') for d in dates]

# Final dataframe
dataframe_day = pd.DataFrame({'Interest':interests_day, 'Date': dates_day})
dataframe_day

# Not used interpolate function:
f = interp1d(range(len(interests_week)), interests_week)
y_new = f(range(len(interests_week)))

plt.plot(range(len(interests_week)), interests_week, 'o', range(len(interests_week)), y_new, '-')
plot = plt.figure()
plt.show(plot)
