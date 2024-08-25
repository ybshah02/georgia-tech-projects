""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np  
import matplotlib.pyplot as plt	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return 904069476 		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    result = False  		  	   		 	   		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	   		  		  		    	 		 		   		 		  
        result = True  		  	   		 	   		  		  		    	 		 		   		 		  
    return result  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def monte_carlo_sim(win_prob, bankroll=False):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns a list of all the earnings from the martingale approach to roulette betting		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	  
    :param bankroll: The existance of a bankroll
    :type bankroll: boolean  		 	   		  		  		    	 		 		   		 		  
    :return: The earning of each of the 1000 spins.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: list  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		
    episode_winnings = 0
    max_num_spins = 1000
    winnings = np.zeros(max_num_spins)
    curr_spin = 0

    while episode_winnings < 80 and curr_spin < max_num_spins:
        won = False
        bet_amt = 1

        while not won and curr_spin < max_num_spins:
            won = get_spin_result(win_prob)
            if won:
                episode_winnings += bet_amt
            else:
                episode_winnings -= bet_amt
                bet_amt *= 2 # double or nothing baby!

                if bankroll:
                    curr_bankroll = episode_winnings + 256 # 256 signifies the initial bankroll
                    if curr_bankroll == 0:
                        winnings[curr_spin:] = episode_winnings # fill forward if bankroll is 0
                        return winnings
                    elif bet_amt > curr_bankroll: 
                        bet_amt = curr_bankroll # set bet amt to bankroll if bet amt is more than what's affordable
            
            winnings[curr_spin] = episode_winnings
            curr_spin += 1 # increment spin
    
    winnings[curr_spin:] = episode_winnings # fill forward
    return winnings

def build_plot():
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Build the plotted graph that will outline results of the martingale approach		  	   		 	   		  		  		    	 		 		   		 		  	  	   		 	   		  		  		    	 		 		   		 		  
    """  
    plt.figure()
    plt.xlim(0, 300)
    plt.xlabel('Number of Spins')
    
    plt.ylim(-256, 100)
    plt.ylabel('Winnings ($)')

def save_fig(fname):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Save a plotted graph under the images folder		  	   		 	   		  		  		    	 		 		   		 		  		  	   		 	   		  		  		    	 		 		   		 		  
    """  
    plt.savefig(f'./images/{fname}')
    
def run_experiment1_figure1(winnings):
    """
    Creates Figure 1: Winnings for 10 Episodes
    
    :param winnings: List of winnings for 10 episodes
    :type winnings: numpy list
    """
    build_plot()
    for i, winning in enumerate(winnings):
        plt.plot(winning, label=f'Episode {i+1}')

    plt.title('Figure 1: Winnings for 10 Episodes') 
    plt.legend()
    save_fig('Figure_1.png')

def run_experiment_figure2_or_figure4(winnings, figure):
    """
    Creates Figure 2 or Figure 4: Mean Winnings of 1000 Episodes
    
    :param winnings: Array of winnings for 1000 episodes
    :type winnings: numpy list
    :param figure: Specifies which figure to create ('fig2' or 'fig4')
    :type figure: str
    """
    build_plot()
    mean = np.mean(winnings, axis=0)
    std = np.std(winnings, axis=0)
    plt.plot(mean, label='Mean')
    plt.plot(mean+std, label='Mean + Std')
    plt.plot(mean-std, label='Mean - Std')
    plt.legend()

    if figure == 'fig2':
        plt.title('Figure 2: Mean Winnings of 1000 Episodes')
        save_fig('Figure_2.png')
    elif figure == 'fig4':
        plt.title('Figure 2: Mean Winnings of 1000 Episodes with Bankroll of $256')
        save_fig('Figure_4.png')

def run_experiment_figure3_or_figure5(winnings, figure):
    """
    Creates Figure 3 or Figure 5: Median Winnings of 1000 Episodes
    
    :param winnings: Array of winnings for 1000 episodes
    :type winnings: numpy list
    :param figure: Specifies which figure to create ('fig3' or 'fig5')
    :type figure: str
    """

    build_plot()
    median = np.median(winnings, axis=0)
    std = np.std(winnings, axis=0)
    plt.plot(median, label='Median')
    plt.plot(median+std, label='Median + Std')
    plt.plot(median-std, label='Median - Std')
    plt.legend()

    if figure == 'fig3':
        plt.title('Figure 3: Median Winnings of 1000 Episodes')
        save_fig('Figure_3.png')
    elif figure == 'fig5':
        plt.title('Figure 5: Median Winnings of 1000 Episodes with Bankroll of $256')
        save_fig('Figure_5.png')
    

def test_code():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    win_prob = 18/38 # set appropriately to the probability of a win - 18 black/red out of 38 possible landing spots 		  	   		 	   		  		  		    	 		 		   		 		  
    np.random.seed(gtid())  # do this only once  		  	   		 	   		  		  		    	 		 		   		 		    	   		 	   		  		  		    	 		 		   		 		  

    winnings = np.array([monte_carlo_sim(win_prob) for _ in range(10)])
    run_experiment1_figure1(winnings)

    winnings = np.array([monte_carlo_sim(win_prob) for _ in range(1000)])
    run_experiment_figure2_or_figure4(winnings, figure ='fig2')
    run_experiment_figure3_or_figure5(winnings, figure='fig3')

    winnings = np.array([monte_carlo_sim(win_prob, bankroll=True) for _ in range(1000)])		  
    run_experiment_figure2_or_figure4(winnings, figure='fig4')
    run_experiment_figure3_or_figure5(winnings, figure='fig5')	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	   		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	   		  		  		    	 		 		   		 		  
