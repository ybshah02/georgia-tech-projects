""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import warnings  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
    
    def author(self):  		  	   		 	   		  		  		    	 		 		   		 		  	  	   		 	   		  		  		    	 		 		   		 		  
        return "yshah89" 
    
    def study_group(self):
        return "yshah89"

    def add_evidence(self, X, Y):
        data = np.column_stack((X, Y))
        self.tree = self.build_tree(data)

    def get_best_feature_i(self, data):
        feat_count = data.shape[1] - 1
        target = data[:, -1]
        corrs = np.zeros(feat_count)
        
        for i in range(feat_count):
            feat = data[:, i]
            if np.std(feat) == 0:
                corrs[i] = 0
            else:
                temp = np.vstack((feat, target))
                corr = np.abs(np.corrcoef(temp)[0, 1])
                corrs[i] = 0 if np.isnan(corr) else corr
        
        best_i = np.argmax(corrs)
        return best_i
    
    def build_tree(self, data):

        nan = np.nan
        # JR Quinlan algorithm
        if data.shape[0] <= self.leaf_size or np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[-1, -1], nan, nan]])
        else:
            # determine best feature
            i = self.get_best_feature_i(data)
            col = data[:, i]

            # split value for columns
            split_val = np.median(col)
            if split_val == np.max(col):
                return np.array([[-1, np.mean(col), nan, nan]])
            
            # recursively build tree on left and right sides
            left_tree = self.build_tree(data[col <= split_val])
            right_tree = self.build_tree(data[col > split_val])

            root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        pred = np.zeros(points.shape[0])
        for i, sample in enumerate(points):
            idx = 0
            while True:
                curr = self.tree[idx]
                if curr[0] == -1:
                    pred[i] = curr[1]
                    break
                feat = int(curr[0])
                split = curr[1]
                idx += int(curr[2]) if sample[feat] <= split else int(curr[3])
        return pred
