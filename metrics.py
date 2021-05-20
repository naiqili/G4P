import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import exp, abs, log
from scipy.special import gamma, factorial
from utils import *

def cumulative_return(rt_v):
    return exp(np.sum(rt_v))

def plot_cumulative_return_history(concat_results, strategy_lst, figsize=(10,5)):
    plt.figure(figsize=figsize)    
    plt.title('cumulative return')
    for strategy in strategy_lst:
        strategy_name = strategy.split('(')[0]
        rt_v, x_vec = concat_results[strategy_name]
        exp_rt = rt_v.copy()
        for i in range(1, len(rt_v)):
            exp_rt[i] += exp_rt[i-1]
        exp_rt = exp(exp_rt)
        plt.plot(exp_rt, label=strategy_name)
    plt.legend()
    plt.show()
    
def daily_return(rt_v):
    return exp(np.mean(rt_v))

def cumulative_return_fee(rt_v, x_vec, c): # TODO
    return 0

def plot_cumulative_return_history(concat_results, strategy_lst, c, figsize=(10,5)): # TODO
    pass
    
def daily_return_fee(rt_v, x_vec, c): # TODO
    return 0

def max_redraw(rt_v): # TODO: to standard
    res = max([rt_v[i] - min(rt_v[i+1:]) for i in range(len(rt_v)-1)])
    return res

def sharpe_ratio(rt_v, rf): # TODO
    return 0

def volatility(rt_v): # TODO
    return 0

def turnover(rt_v): # TODO
    return 0