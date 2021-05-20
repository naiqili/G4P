import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import exp, abs, log
from scipy.special import gamma, factorial
import os
import scipy.stats as stats
import statsmodels.api as sm

lb = 10
maxiter = 1000
subn = 100

def get_res(phase):
    data_path1 = './tmp/original_res_phase%02d_iter%d_subn%d_lb%d.npz'%(phase, maxiter, subn, lb)
    data1 = np.load(data_path1)
    return data1['predY0'], data1['std_varY0'], data1['sample_Y0']

def ucb_strategy(lam):
    rt_v = []
    ninvest = 10

    for i in range(len(predY0)):
        p = predY0[i]
        sell_idx = np.argsort(p-lam*std_varY0[i])[:ninvest//2]
        buy_idx = np.argsort(-(p+lam*std_varY0[i]))[:ninvest//2]
        rr = 1/ninvest
        tmp = rr*(sum(exp(sample_Y0[i, buy_idx]))+sum(exp(-sample_Y0[i, sell_idx])))
        rt_v.append(log(tmp))
    return rt_v


for phs in range(12):
    print('Phase %d' % phs)
    predY0, std_varY0, sample_Y0 = get_res(phs)

    print('Naive strategy')
    rt_v = []
    ninvest = 10
    for i in range(len(predY0)):
        p = predY0[i]
        sell_idx = np.argsort(p)[:ninvest // 2]
        buy_idx = np.argsort(-p)[:ninvest // 2]
        rr = 1 / ninvest
        tmp = rr * (sum(exp(sample_Y0[i, buy_idx])) + sum(exp(-sample_Y0[i, sell_idx])))
        rt_v.append(log(tmp))
    print('Total log return: %.8f' % sum(rt_v))
    print('Total return: %.8f' % exp(sum(rt_v)))
    print('Mean log return: %.8f' % np.mean(rt_v))
    print('Mean return: %.8f' % exp(np.mean(rt_v)))
    exp_rt = rt_v.copy()
    for i in range(1, len(rt_v)):
        exp_rt[i] += exp_rt[i - 1]
    exp_rt = exp(exp_rt)
    plt.figure(figsize=(10, 5))
    plt.plot(exp_rt, label='cumulated return')
    plt.title('Naive')
    plt.grid()
    plt.legend()
    # plt.show()

    print('')
    print('UCB strategy')
    best_logrt = -np.inf
    best_lam = 0
    testv = [1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.03, 0.01]
    for lam in testv + [-v for v in testv] + [0]:
        rt_v = ucb_strategy(lam)
        print('lambda: %.3f   Total log return: %.8f' % (lam, sum(rt_v)))
        if sum(rt_v) > best_logrt:
            best_logrt = sum(rt_v)
            best_lam = lam
    rt_v = ucb_strategy(best_lam)
    print('best lambda: %.3f   Total log return: %.8f' % (best_lam, sum(rt_v)))
    print('Total log return: %.8f' % sum(rt_v))
    print('Total return: %.8f' % exp(sum(rt_v)))
    print('Mean log return: %.8f' % np.mean(rt_v))
    print('Mean return: %.8f' % exp(np.mean(rt_v)))
    exp_rt = rt_v.copy()
    for i in range(1, len(rt_v)):
        exp_rt[i] += exp_rt[i - 1]
    exp_rt = exp(exp_rt)
    plt.figure(figsize=(10, 5))
    plt.plot(exp_rt, label='cumulated return')
    plt.title('UCB')
    plt.grid()
    plt.legend()
    # plt.show()