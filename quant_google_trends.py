import datetime as dt
from datetime import timedelta, date
from itertools import chain
from scipy.interpolate import interp1d
from pytrends.request import TrendReq

class QuantGoogleTrends:

    # constructs a dataset
    def __init__(self, timeframe):
        self.trendmatrix = self.gettrendmatrix()


    def trendmatrix(self):
        # API for the bitcoin interest
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ["Bitcoin"]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')

        i_df = pytrends.interest_over_time()
        interests = i_df.drop(labels='isPartial', axis=1)
        interests_df = pd.DataFrame(interests)
        interests_week = list(interests_df['Bitcoin'])

        # Create the interests per day list
        newlist = []
        for i in range(len(interests_week)-1):
            point1, point2 = interests_week[i], interests_week[i+1]
            linearlist = [point1]  + linear_function(point1, point2)
            newlist.append(linearlist)

        interests_day = list(chain.from_iterable(newlist))
        interests_day.append(interests_week[-1])
        interests_day = [interests_week[0]]+interests_day


        dates = daterange(date(2013, 1, 13), date(2018, 1, 8))
        dates_day = [(pd.to_datetime(str(d))).strftime('%Y.%m.%d') for d in dates]

        # Final dataframe
        dataframe_day = pd.DataFrame({'Interest':interests_day, 'Date': dates_day})
        return dataframe_day

    # takes a timeframe of unix timestamps
    def gettrends(self, timeframe):
        fr = datetime.datetime.fromtimestamp(
                int(timeframe[0])
            ).strftime('%Y.%m.%d')

        to = datetime.datetime.fromtimestamp(
                int(timeframe[1])
            ).strftime('%Y.%m.%d')

        fromind = df.index[df['Date'] == fr].tolist()
        toind = df.index[df['Date'] == to].tolist()

        return self.trendmatrix.copy(deep=true).iloc([fr, to + 1])

    # Plot a graph
    def plot_graph(self, x, y, title):
        plt.plot(x, y, 'ro')
        plt.title(title)
        plt.show()
