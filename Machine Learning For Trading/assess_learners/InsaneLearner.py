import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=self.verbose) for i in range(20)]
    def author(self):  		  	   		 	   		  		  		    	 		 		   		 		  		  	   		 	   		  		  		    	 		 		   		 		  
        return "yshah89" 
    def add_evidence(self, X, Y):
        for learner in self.learners:
            learner.add_evidence(X, Y)
    def query(self, test):
        return np.mean([learner.query(test) for learner in self.learners], axis=0)