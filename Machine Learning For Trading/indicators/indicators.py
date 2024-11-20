import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_data

def author():
    return 'yshah89'

def study_group():
    return 'yshah89'

def simple_moving_average(df, window=20):
    return df.rolling(window=window).mean()

def calculate_bollinger_bands(df, window=20):
    moving_avg = simple_moving_average(df, window)
    rolling_std_dev = df.rolling(window=window).std()
    band_diff = 2 * rolling_std_dev
    return moving_avg + band_diff, moving_avg - band_diff

def exponential_moving_average(df, window=20):
    return df.ewm(span=window, adjust=False).mean()

def compute_relative_strength_index(df, window=14):
    delta = df.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd_histogram(df, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = exponential_moving_average(df, fast_period)
    ema_slow = exponential_moving_average(df, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal_period)
    macd_hist = macd_line - signal_line
    return macd_hist

def compute_stochastic_oscillator(df, window=14):
    lowest_low = df.rolling(window=window).min()
    highest_high = df.rolling(window=window).max()
    k_percent = 100 * (df - lowest_low) / (highest_high - lowest_low)
    return k_percent

def plot_indicator_with_df_on_axis(axis, df, indicator, indicator_name):
    axis.plot(df.index, df / df.iloc[0], label='Normalized Price')
    axis.plot(indicator.index, indicator, label=indicator_name)
    axis.set_title(f'{indicator_name}')
    axis.legend()
    axis.set_xlabel('Date')
    axis.set_ylabel('Value')

def plot_df_with_sma_and_bollinger_bands(df):
    sma_20 = simple_moving_average(df)
    upper_band, lower_band = calculate_bollinger_bands(df)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df, label='Price')
    plt.plot(sma_20.index, sma_20, label='SMA(20)')
    plt.plot(upper_band.index, upper_band, label='Upper Band')
    plt.plot(lower_band.index, lower_band, label='Lower Band')
    plt.title('Price with SMA and Bollinger Bands')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('images/df_sma_bollinger_bands.png')
    plt.close()

def plot_macd(df, macd_histogram):
    fig, (axis_df, axis_macd) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ema_fast = exponential_moving_average(df, 12)
    ema_slow = exponential_moving_average(df, 26)
    axis_df.plot(df.index, df, label='Price')
    axis_df.plot(ema_fast.index, ema_fast, label='12 EMA')
    axis_df.plot(ema_slow.index, ema_slow, label='26 EMA')
    axis_df.set_title('Price and EMAs')
    axis_df.legend()
    axis_df.set_ylabel('Price')

    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, 9)

    axis_macd.bar(macd_histogram.index, macd_histogram, label='MACD Hist')
    axis_macd.plot(macd_line.index, macd_line, label='MACD Line')
    axis_macd.plot(signal_line.index, signal_line, label='Signal Line')
    axis_macd.set_title('MACD')
    axis_macd.legend()
    axis_macd.set_xlabel('Date')
    axis_macd.set_ylabel('MACD')
    plt.tight_layout()
    plt.savefig('images/macd.png')
    plt.close()

def plot_stochastic_oscillator(df, k_percent):
    fig, (axis_df, axis_stoch) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axis_df.plot(df.index, df, label='Price')
    axis_df.set_title('Price')
    axis_df.legend()
    axis_df.set_ylabel('Price')
    d_percent = k_percent.rolling(window=3).mean()
    axis_stoch.plot(k_percent.index, k_percent, label='%K')
    axis_stoch.plot(d_percent.index, d_percent, label='%D')
    axis_stoch.axhline(y=80, color='r', linestyle='--')
    axis_stoch.axhline(y=20, color='g', linestyle='--')
    axis_stoch.set_title('Stochastic Oscillator')
    axis_stoch.legend()
    axis_stoch.set_xlabel('Date')
    axis_stoch.set_ylabel('Value')
    axis_stoch.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('images/stochastic_oscillator.png')
    plt.close()

def run_indicators(symbol, start_date, end_date):
    df_data = get_data([symbol], pd.date_range(start_date, end_date))
    df = df_data[symbol]

    plot_df_with_sma_and_bollinger_bands(df)
    ema_values = exponential_moving_average(df)
    rsi_values = compute_relative_strength_index(df)
    macd_histogram = compute_macd_histogram(df)
    stochastic_values = compute_stochastic_oscillator(df)

    plt.figure(figsize=(12, 6))
    plot_indicator_with_df_on_axis(plt.gca(), df, ema_values, 'Exponential Moving Average (EMA)')
    plt.savefig('images/ema.png')
    plt.close()

    fig, (axis_price, axis_rsi) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axis_price.plot(df.index, df, label='Price')
    axis_price.set_title('Price')
    axis_price.legend()
    axis_price.set_ylabel('Price')
    axis_rsi.plot(rsi_values.index, rsi_values, label='RSI')
    axis_rsi.axhline(y=70, color='r', linestyle='--')
    axis_rsi.axhline(y=30, color='g', linestyle='--')
    axis_rsi.set_title('Relative Strength Index (RSI)')
    axis_rsi.legend()
    axis_rsi.set_xlabel('Date')
    axis_rsi.set_ylabel('RSI')
    axis_rsi.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig('images/rsi.png')
    plt.close()

    plot_macd(df, macd_histogram)
    plot_stochastic_oscillator(df, stochastic_values)

if __name__ == "__main__":
    symbol_of_interest = "JPM"
    start_date_str = "2008-01-01"
    end_date_str = "2009-12-31"
    run_indicators(symbol_of_interest, start_date_str, end_date_str)