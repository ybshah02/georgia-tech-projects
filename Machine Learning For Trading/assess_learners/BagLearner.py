
import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose=False):
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []

        for i in range(self.bags):
            self.learners.append(self.learner(**kwargs))
    
    def author(self):  		  	   		 	   		  		  		    	 		 		   		 		  		  	   		 	   		  		  		    	 		 		   		 		  
        return "yshah89"

    def study_group(self):
        return "yshah89" 		

    def add_evidence(self, X, Y):
        for i in range(self.bags):
            X_shape = X.shape[0]
            indices = np.random.choice(X_shape, X_shape, replace=True)
            X_bag = X[indices]
            Y_bag = Y[indices]
            
            self.learners[i].add_evidence(X_bag, Y_bag)

    def query(self, test):
        pred = np.array([learner.query(test) for learner in self.learners])
        return np.mean(pred, axis=0)