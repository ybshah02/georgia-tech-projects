import pandas as pd
import numpy as np
import datetime as dt
from util import get_data

def author():
    return 'yshah89'

def study_group():
    return 'yshah89'

def compute_tp(dr_value):
    if dr_value > 0:
        return 1000
    elif dr_value < 0:
        return -1000
    else:
        return 0

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    date_range = pd.date_range(sd, ed)
    data_frame = get_data([symbol], date_range)
    price_series = data_frame[symbol]
    daily_ret = price_series.pct_change()
    trade_df = pd.DataFrame(0, index=price_series.index, columns=['Shares'])

    current_pos = 0
    idx = 1
    while idx < len(price_series):
        target_pos = compute_tp(daily_ret.iloc[idx])
        trade_val = target_pos - current_pos
        current_pos = target_pos
        trade_df.iloc[idx] = trade_val
        idx += 1

    return trade_df