import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import exp, abs, log
from scipy.special import gamma, factorial
import cvxpy
# from utils import *

def ucb_strategy(lam, predY0, std_varY0, sample_Y0, ninvest = 10):
    rt_v, x_vec = [], []

    for i in range(len(predY0)):
        x_opt = np.zeros(len(predY0[0]))
        p = predY0[i]
#         sell_idx = np.argsort(p-lam*std_varY0[i])[:ninvest//2]
        buy_idx = np.argsort(-(p+lam*std_varY0[i]))[:ninvest]
        rr = 1/ninvest
        x_opt[buy_idx] = rr
#         tmp = rr*(sum(exp(sample_Y0[i, buy_idx]))+sum(exp(-sample_Y0[i, sell_idx])))
        tmp = rr*(sum(exp(sample_Y0[i, buy_idx])))
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
        
    return rt_v, x_vec

def passive_strategy(predY0, std_var_Y0, sample_Y0, cov, eps, delta, cash=1.0):
    def passive_solve(x_len, x_ori, r_hat, cov, eps, delta, cash=1):
        x = cvxpy.Variable(x_len)  # optimization vector variable
        x0 = cvxpy.Parameter(x_len)
        r = cvxpy.Parameter(x_len)  # placeholder for vector c
        A = cvxpy.Parameter((x_len, x_len))  # placeholder for sqrtm(Sigma)
        obj = cvxpy.Minimize(0.5*cvxpy.norm(x - x0, 2))  #define objective function
        cons = [x.T @ r >= eps, cvxpy.norm(A@x, 2) <= delta, cvxpy.sum(x) == cash, x >= 0]      # define set of constraints
        prob = cvxpy.Problem(obj, cons)  #setup the problem
        A.value = sqrtm(cov)
        r.value = r_hat
        x0.value = x_ori

        prob.solve(solver='SCS')  # solve the problem

        x_opt = x.value  # the optimal variable
        return x_opt
    
    rt_v, x_vec = [], []
    x_ori = np.ones(len(predY0[0])) / len(predY0[0])
    for i in range(len(predY0)):
        r = exp(predY0[i])
        cov_dia = std_var_Y0[i] ** 2 * r * r
        cov_current = np.zeros_like(cov)
#        cov_current = cov.copy()
        for k in range(len(r)):
            cov_current[k, k] = cov_dia[k]
        x_opt = passive_solve(len(r), x_ori, r, cov_current, eps, delta, cash)
        if x_opt is None:
            print('No opt solution')
            x_opt = x_ori
        
        tmp = sum(x_opt * exp(sample_Y0[i])) / cash
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
        
        x_ori = x_opt
    return rt_v, x_vec

def passive_strategy2(predY0, std_var_Y0, sample_Y0, cov, eps, gamma0, cash=1.0):
    def passive_solve2(x_len, x_ori, r_hat, cov, eps, gamma, cash=1):
        x = cvxpy.Variable(x_len)  # optimization vector variable
        x0 = cvxpy.Parameter(x_len)
        r = cvxpy.Parameter(x_len)  # placeholder for vector c
        gamma = cvxpy.Parameter(nonneg=True)
        risk = cvxpy.quad_form(x, cov)
        obj = cvxpy.Minimize(0.5*cvxpy.norm(x - x0, 2) + gamma*risk)  #define objective function
        x0.value = x_ori
        prob = cvxpy.Problem(obj, 
                       [cvxpy.sum(x) == 1, x.T @ r >= eps,
                        x >= 0, x <= 0.3])
        r.value = r_hat
        gamma.value = gamma0
        prob.solve(solver='SCS')  # solve the problem
        x_opt = x.value  # the optimal variable
        return x_opt
    
    rt_v, x_vec = [], []
    x_ori = np.ones(len(predY0[0])) / len(predY0[0])
    for i in range(len(predY0)):
        r = exp(predY0[i])
        cov_dia = std_var_Y0[i] ** 2 * r * r
        cov_current = np.zeros_like(cov)
#            cov_current = cov.copy()
        for k in range(len(r)):
            cov_current[k, k] = cov_dia[k]
        x_opt = passive_solve2(len(r), x_ori, r, cov_current, eps, gamma0, cash)
        if x_opt is None:
            print('No opt solution')
            x_opt = x_ori
        
#         print(x_opt)
#         print(sum(abs(x_opt-x_ori)))
        
        tmp = sum(x_opt * exp(sample_Y0[i])) / cash
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
        
        x_ori = x_opt
    return rt_v, x_vec

def opt_strategy(predY0, std_var_Y0, sample_Y0, cov, gamma, cash=1.0):
    def solve2(x_len, r_hat, cov, gamma0=5.0):
        x = cvxpy.Variable(x_len)  # optimization vector variable
        r = cvxpy.Parameter(x_len)  # placeholder for vector c
        gamma = cvxpy.Parameter(nonneg=True)
        dmax = cvxpy.Parameter()  # placeholder for parameter dmax
        risk = cvxpy.quad_form(x, cov)
        obj = cvxpy.Maximize(cvxpy.sum(x@r) - gamma*risk)  #define objective function
        prob = cvxpy.Problem(obj, 
                       [cvxpy.sum(x) == 1, 
                        x >= 0, x <= 0.3])
        r.value = r_hat
        gamma.value = gamma0
#         print(np.min(cov))
        prob.solve()  # solve the problem
        x_opt = x.value  # the optimal variable
        return x_opt
    
    rt_v, x_vec = [], []
    for i in range(len(predY0)):
        r = exp(predY0[i])
        cov_dia = std_var_Y0[i]**2 * r * r
#         cov_current = cov.copy()
        cov_current = np.zeros_like(cov)
        for k in range(len(r)):
            cov_current[k, k] = cov_dia[k]
        x_opt = solve2(len(r), r, cov_current, gamma)
        
#         print(x_opt)
        
        tmp = sum(x_opt * exp(sample_Y0[i])) / cash
        rt_v.append(log(tmp))
        x_vec.append(x_opt)
    return rt_v, x_vec