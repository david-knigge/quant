import datetime as dt
from datetime import timedelta, date
from itertools import chain
from scipy.interpolate import interp1d
from pytrends.request import TrendReq

class QuantGoogleTrends:

    # Plot a graph
    def plot_graph(self, x, y, title):
        plt.plot(x, y, 'ro')
        plt.title(title)
        plt.show()
