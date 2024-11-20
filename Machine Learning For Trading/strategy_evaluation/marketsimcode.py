
import datetime as dt  		  	   		 	   		  		  		    	 		 		   		 		  
import os  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		     		 	   		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	   		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		 	   		  		  		    	 		 		   		 		  

def author():
    return 'yshah89'

def study_group():
    return 'yshah89'   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		 	   		  		  		    	 		 		   		 		  
    trades,  		  	   		 	   		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		 	   		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		 	   		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		 	   		  		  		    	 		 		   		 		  
):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		 	   		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		 	   		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		 	   		  		  		    	 		 		   		 		  
    """

    start_date = trades.index[0]
    end_date = trades.index[-1]
    
    symbol = 'JPM' 
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]  # remove SPY
    prices['Cash'] = 1.0  # add cash column
    
    positions = pd.DataFrame(index=prices.index, columns=prices.columns)
    positions = positions.fillna(0)
    positions['Cash'] = start_val
    
    for date, row in trades.iterrows():
        price = prices.loc[date, symbol]
        trade_shares = row['Shares']
        trade_value = price * trade_shares
        
        positions.loc[date:, symbol] += trade_shares
        positions.loc[date:, 'Cash'] -= trade_value
        
        positions.loc[date:, 'Cash'] -= commission
        positions.loc[date:, 'Cash'] -= abs(trade_value) * impact
    
    portvals = (positions * prices).sum(axis=1)
    
    return pd.DataFrame(portvals, columns=['Portfolio Value'])
