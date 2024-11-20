""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import math  		  	   		 	   		  		  		    	 		 		   		 		  
import sys  	
import time	  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import matplotlib.pyplot as plt
import DTLearner as dtl	 
import RTLearner as rtl
import BagLearner as bl 	  

def experiment1(train_x, train_y, test_x, test_y):
        leaf_sizes = range(1, 51)
        train_rmse = []
        test_rmse = []

        for leaf_size in leaf_sizes: 		  
            learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)	   	
            learner.add_evidence(train_x, train_y)

            pred_y_train = learner.query(train_x) 		  	   		 	   		  		  		    	 		 		   		 		  
            rmse_train = math.sqrt(((train_y - pred_y_train) ** 2).sum() / train_y.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		    	   		 	   		  		  		    	 		 		   		 		  
            train_rmse.append(rmse_train)	 		  	   		 	   		  		  		    	 		 		   		 		   		  	   		 	   		  		  		    	 		 		   		 		  
                                                                                                		  	   		 	   		  		  		    	 		 		   		 		  
            pred_y_test = learner.query(test_x)		  	   		 	   		  		  		    	 		 		   		 		  
            rmse_test = math.sqrt(((test_y - pred_y_test) ** 2).sum() / test_y.shape[0])  	
            test_rmse.append(rmse_test)	 

        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, train_rmse, label='Training RMSE')
        plt.plot(leaf_sizes, test_rmse, label='Testing RMSE')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        plt.title('DTLearner RMSE vs Leaf Size')
        plt.legend()
        plt.savefig('images/experiment1.png')
        plt.close()  

def experiment2(train_x, train_y, test_x, test_y):
        leaf_sizes = range(1, 51)
        dtl_train_rmse = []
        dtl_test_rmse = []
        bag_train_rmse = []
        bag_test_rmse = []

        num_bags = 20

        for leaf_size in leaf_sizes: 		  
            learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)	   	
            learner.add_evidence(train_x, train_y)


            dtl_learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
            dtl_learner.add_evidence(train_x, train_y)
            dtl_train_pred = dtl_learner.query(train_x)
            dtl_test_pred = dtl_learner.query(test_x)
            dtl_train_rmse.append(math.sqrt(((train_y - dtl_train_pred) ** 2).mean()))
            dtl_test_rmse.append(math.sqrt(((test_y - dtl_test_pred) ** 2).mean()))

            bag_learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size":leaf_size}, bags=num_bags, boost=False, verbose=False)
            bag_learner.add_evidence(train_x, train_y)
            bag_train_pred = bag_learner.query(train_x)
            bag_test_pred = bag_learner.query(test_x)
            bag_train_rmse.append(math.sqrt(((train_y - bag_train_pred) ** 2).mean()))
            bag_test_rmse.append(math.sqrt(((test_y - bag_test_pred) ** 2).mean()))

        plt.figure(figsize=(12, 6))
        plt.plot(leaf_sizes, dtl_train_rmse, label='DTL Train RMSE')
        plt.plot(leaf_sizes, dtl_test_rmse, label='DTL Test RMSE')
        plt.plot(leaf_sizes, bag_train_rmse, label='Bag Train RMSE')
        plt.plot(leaf_sizes, bag_test_rmse, label='Bag Test RMSE')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        plt.title(f'DTLearner vs BagLearner (bags={num_bags}) Performance')
        plt.legend()
        plt.savefig('images/experiment2.png')
        plt.close()  

def calculate_r_squared(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (rss / tss)

def experiment3(train_x, train_y, test_x, test_y):
    leaf_sizes = range(1, 51)
    dtl_train_t = []
    dtl_r_squared = []
    rtl_train_t = []
    rtl_r_squared = []

    for leaf_size in leaf_sizes:
        dtl_learner = dtl.DTLearner(leaf_size=leaf_size, verbose=False)
        start_time = time.time()
        dtl_learner.add_evidence(train_x, train_y)
        dtl_train_t.append(time.time() - start_time)
        
        dtl_predictions = dtl_learner.query(test_x)
        dtl_r_squared.append(calculate_r_squared(test_y, dtl_predictions))

        rtl_learner = rtl.RTLearner(leaf_size=leaf_size, verbose=False)
        start_time = time.time()
        rtl_learner.add_evidence(train_x, train_y)
        rtl_train_t.append(time.time() - start_time)
        
        rtl_predictions = rtl_learner.query(test_x)
        rtl_r_squared.append(calculate_r_squared(test_y, rtl_predictions))

    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, dtl_train_t, label='DTLearner')
    plt.plot(leaf_sizes, rtl_train_t, label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Leaf Size')
    plt.legend()
    plt.savefig('images/experiment3_training_time.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, dtl_r_squared, label='DTLearner')
    plt.plot(leaf_sizes, rtl_r_squared, label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('R-squared')
    plt.title('R-squared vs Leaf Size')
    plt.legend()
    plt.savefig('images/experiment3_r_squared.png')
    plt.close()
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__": 

    if len(sys.argv) != 2:  		  	   		 	   		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	   		  		  		    	 		 		   		 		  
        sys.exit(1)  		

    data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    data = data[1:, 1:]	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		 	   		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		 	   		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	   		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		 	   		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		 	   		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		 	   		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]

    experiment1(train_x, train_y, test_x, test_y)
    experiment2(train_x, train_y, test_x, test_y)	
    experiment3(train_x, train_y, test_x, test_y)		   		 	   		  		  		    	 		 		   		 		   		 	   		  		  		    	 		 		   		 		  
