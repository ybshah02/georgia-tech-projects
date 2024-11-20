""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
import random as rand  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
class QLearner(object):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		 	   		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		 	   		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    def __init__(  		  	   		 	   		  		  		    	 		 		   		 		  
        self,  		  	   		 	   		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		 	   		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		 	   		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		 	   		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		 	   		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		 	   		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		 	   		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		 	   		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		 	   		  		  		    	 		 		   		 		  
    ):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        
        self.bigQ = np.zeros((num_states, num_actions))

        self.curr_s = 0
        self.curr_a = 0 

        self.transitions = np.empty((num_states, num_actions), dtype=int)
        self.transitions.fill(-1)
        
        self.rewards = np.zeros((num_states, num_actions))
        
        self.visited = []

    def author(self):
        return 'yshah89'	

    def study_group(self):
        return 'yshah89' 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		 	   		  		  		    	 		 		   		 		  
        :type s: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        self.curr_s = s
    
        if rand.random() <= self.rar:
            self.curr_a = rand.randrange(self.num_actions)
            return self.curr_a

        q_vals = self.bigQ[s]
        actions = []
        
        for i in range(len(q_vals)): # find best actions
            if q_vals[i] == q_vals.max():
                actions.append(i)
        
        self.curr_a = actions[rand.randint(0, len(actions)-1)] # choose randomly from best actions
        return self.curr_a

    def do_q_update(self, s, a, reward, next_state):
        old_q = self.bigQ[s, a]

        # https://www.analyticsvidhya.com/blog/2021/02/understanding-the-bellman-optimality-equation-in-reinforcement-learning/#:~:text=The%20Bellman%20Optimality%20Equation%20relates,to%20the%20optimal%20Q%2Dfunction.
        learned_val = reward + self.gamma * np.max(self.bigQ[next_state, :])
        diff = learned_val - old_q

        self.bigQ[s, a] = old_q + self.alpha * diff

    def do_dyna_updates(self):
        # http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%209.pdfLinks to an external site.

        for i in range(self.dyna):
            idx = rand.randint(0, len(self.visited) - 1)
            s, a = self.visited[idx]
            
            self.do_q_update(s, a, self.rewards[s, a], self.transitions[s, a] )  # learn from visited
   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		 	   		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		 	   		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		 	   		  		  		    	 		 		   		 		  
        :type r: float  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
        """  		  	
        s = self.curr_s
        a = self.curr_a

        self.do_q_update(s, a, r, s_prime)

        self.transitions[s, a] = s_prime
        self.rewards[s, a] = r

        sa_pair = (s, a)
        if sa_pair not in self.visited: # track visited actions
            self.visited.append(sa_pair)

        if self.dyna > 0:
            self.do_dyna_updates()

        self.rar *= self.radr

        return self.querysetstate(s_prime)	# pick next action  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  	  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		 	   		  		  		    	 		 		   		 		  
