""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Yash Shah 		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: yshah89 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 904069476 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  	

import scipy.optimize as spo
  		  	   		 	   		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		 	   		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data, plot_data  	

def author():
    return 'yshah89'

def study_group():
    return 'yshah89'

# daily portfolio value calculations
# normed = prices / prices[0]
# alloced = normed * allocs
# pos_vals = aloced * start_val
# port_val = pos_vals.sum(axis=1)
# daily_rets = daily_rets[1:]
# cum_ret = port_val[-1] / port_val[0] - 1
# avg_daily_ret = daily_rets.mean()
# std_daily_ret = daily_rets.std()
# s = mean(daily_rets -  daily_rf) / std(daily_rets - daily_rf)	  

def get_port_val(prices, allocs, start_val):
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)

    return port_val	 	     		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	   		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		 	   		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		 	   		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		 	   		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		 	   		  		  		    	 		 		   		 		  
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	   		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	   		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	   		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	   		  		  		    	 		 		   		 		  
    statistics.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		 	   		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	   		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	   		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	   		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	   		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		 	   		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		 	   		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		 	   		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		 	   		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  

    def assess(allocs):
        port_val = get_port_val(prices, allocs, 1)
        daily_ret = (port_val / port_val.shift(1) - 1)
        sharpe_ratio = np.sqrt(252) * daily_ret.mean() / daily_ret.std() # using 252 for number of trading days, and 0.0 for risk free return rate
        return -sharpe_ratio	

    n = len(syms)	
    initial_guess = np.array([1.0 / n] * n)
    bounds = tuple((0,1) for _ in syms)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = spo.minimize(assess, initial_guess, bounds=bounds, constraints=constraints)

    allocs = res.x
    port_val = get_port_val(prices, allocs, 1)
    daily_ret = (port_val / port_val.shift(1) - 1)
  		  	   		 	   		  		  		    	 		 		   		 		 		  	   		 	   		  		  		    	 		 		   		 		  
    cr, adr, sddr, sr = [  		  	   		 	   		  		  		    	 		 		   		 		  
        port_val.iloc[-1] / port_val.iloc[0] - 1,  		  	   		 	   		  		  		    	 		 		   		 		  
        daily_ret.mean(),  		  	   		 	   		  		  		    	 		 		   		 		  
        daily_ret.std(),  		  	   		 	   		  		  		    	 		 		   		 		  
        np.sqrt(252) * daily_ret.mean() / daily_ret.std(),  		  	   		 	   		  		  		    	 		 		   		 		  
    ]  		   		 	   		  		  		    	 		 		   		 		    	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		 	   		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		 	   		  		  		    	 		 		   		 		 		 		   		 		  
        df_temp = pd.concat(  		  	   		 	   		  		  		    	 		 		   		 		  
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1  		  	   		 	   		  		  		    	 		 		   		 		  
        )
        df_temp = df_temp / df_temp.iloc[0]

        plt.figure(figsize=(10, 6))
        plt.plot(df_temp.index, df_temp["Portfolio"], label="Portfolio", color="dodgerblue")
        plt.plot(df_temp.index, df_temp["SPY"], label="SPY", color="limegreen")
        plt.title('Daily Portfolio Value and SPY')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)	  
        plt.xlim(sd, ed)
        plt.savefig('Figure1.png')
        plt.close()	    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2009, 6, 1)  		  	   		 	   		  		  		    	 		 		   		 		  
    symbols = ["IBM", "X", "GLD", "JPM"]  	  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		 	   		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		 	   		  		  		    	 		 		   		 		  
    )  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		 	   		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		 	   		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
