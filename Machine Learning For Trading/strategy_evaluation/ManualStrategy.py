import datetime as dt
import pandas as pd
from util import get_data
from .indicators import compute_macd_histogram, compute_relative_strength_index, compute_stochastic_oscillator

def testPolicy(self, symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
    # Fetch price data for the given symbol and date range
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices = prices[symbol]

    # Calculate indicators
    rsi = compute_relative_strength_index(prices)
    macd_hist = compute_macd_histogram(prices)
    stochastic = compute_stochastic_oscillator(prices)

    # Initialize trades DataFrame
    trades = prices.copy()
    trades[:] = 0

    position = 0  # Current position: 0 (out), 1 (long), -1 (short)

    for i in range(1, len(prices)):
        signal = 0

        # Generate signals based on indicators
        if (rsi[i] < 30) and (macd_hist[i-1] < 0 and macd_hist[i] > 0) and (stochastic[i] < 20):
            signal = 1  # Buy signal
        elif (rsi[i] > 70) and (macd_hist[i-1] > 0 and macd_hist[i] < 0) and (stochastic[i] > 80):
            signal = -1  # Sell signal

        # Execute trades based on signals
        if signal == 1 and position <= 0:
            trades.iloc[i] = 1000 - position * 1000  # Buy enough to get to 1000 shares
            position = 1
        elif signal == -1 and position >= 0:
            trades.iloc[i] = -1000 - position * 1000  # Sell enough to get to -1000 shares
            position = -1

    return trades