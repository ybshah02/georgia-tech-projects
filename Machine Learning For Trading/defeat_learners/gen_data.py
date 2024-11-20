""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Yash Shah (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: yshah89 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 904069476 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  

def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "yshah89" 

def study_group():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT usernames of the study group  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "yshah89" 	 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		 	   		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		  	   		 	   		  		  		    	 		 		   		 		  
    x = np.random.normal(0, 1, size=(500, 8)) # features with 500 rows, 8 cols
    beta = np.random.uniform(-10, 10, x.shape[1])		 
    y = x.dot(beta) + np.random.normal(0, 0.01, size=x.shape[0]) # add little noise to simulate real world data   			  		  		    	 		 		   		 		  
    return x, y	  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		 	   		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		  	   		 	   		  		  		    	 		 		   		 		  
    x = np.random.rand(1000, 3) # features with 1000 rows 3 cols
    y = np.zeros(x.shape[0])	
    y += np.where(x[:, 0] > 0.5, 1, -1)
    y += np.where(x[:, 1] > 0.5, -1, 1)	 	   		  		  		    	 		 		   		 		  
    return x, y  		  	   		 	   		  		  		    	 		 		   		 		  
	  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    print("they call me Tim.")  		  	   		 	   		  		  		    	 		 		   		 		  
