import numpy as np
import cvxpy
from scipy.linalg import sqrtm  # for finding the squared root of Sigma

# def get_res(paras, phase, maxiter, subn, lb):
#     mattype=paras['mattype']
#     dg=paras['dg']
#     if mattype == -1: # pearson        
#         data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_pearson_dg%d_inv%s_ggd%s.npz'%  \
#         (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
#     elif mattype == -2: # spearman
#         data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_spearman_dg%d_inv%s_ggd%s.npz'%  \
#         (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
#     elif mattype == -3: # kendalltau
#         data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_kendall_dg%d_inv%s_ggd%s.npz'%  \
#         (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
#     else: # HATS's graph    
#         data_path1 = path='./tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_matcnt%d_inv%s_ggd%s.npz'%  \
#         (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['matcnt'], paras['inverse'], paras['ggd'])
# #     print(data_path1)
#     data1 = np.load(data_path1)
#     return data1['predY0'], data1['std_varY0'], data1['sample_Y0']

def get_res(paras, phase, maxiter, subn, lb):
    mattype=paras['mattype']
    dg=paras['dg']
    if mattype == -1: # pearson        
        data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_pearson_dg%.2f_inv%s_ggd%s.npz'%  \
        (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
    elif mattype == -2: # spearman
        data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_spearman_dg%.2f_inv%s_ggd%s.npz'%  \
        (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
    elif mattype == -3: # kendalltau
        data_path1 = './tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_kendall_dg%.2f_inv%s_ggd%s.npz'%  \
        (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['dg'], paras['inverse'], paras['ggd'])
    else: # HATS's graph    
        data_path1 = path='./tmp2/%s_phase%02d_iter%d_subn%d_lb%d_M%d_matcnt%d_inv%s_ggd%s.npz'%  \
        (paras['dataset'], phase, paras['maxiter'], paras['subn'], paras['lb'], paras['M'], paras['matcnt'], paras['inverse'], paras['ggd'])
#     print(data_path1)
    data1 = np.load(data_path1)
    return data1['predY0'], data1['std_varY0'], data1['sample_Y0']


# def solve(x_len, r_hat, cov, delta, cash=1):
#     x = cvxpy.Variable(x_len)  # optimization vector variable
#     r = cvxpy.Parameter(x_len)  # placeholder for vector c
#     A = cvxpy.Parameter((x_len, x_len))  # placeholder for sqrtm(Sigma)
#     dmax = cvxpy.Parameter()  # placeholder for parameter dmax

#     obj = cvxpy.Maximize(cvxpy.sum(x@r))  #define objective function
#     cons = [cvxpy.norm(A@x, 2) <= dmax, cvxpy.sum(x) == cash]      # define set of constraints
#     for i in range(x_len):
#         cons = cons + [x >= 0, x <= 0.3]

#     prob = cvxpy.Problem(obj, cons)  #setup the problem

#     # instantiate the problem for a specific value of parameters
#     A.value = sqrtm(cov)
#     r.value = r_hat
#     dmax.value = delta

#     prob.solve()  # solve the problem

#     x_opt = x.value  # the optimal variable
#     return x_opt

# def passive_solve(x_len, x_ori, r_hat, cov, eps, delta, cash=1):
#     x = cvxpy.Variable(x_len)  # optimization vector variable
#     x0 = cvxpy.Parameter(x_len)
#     r = cvxpy.Parameter(x_len)  # placeholder for vector c
#     A = cvxpy.Parameter((x_len, x_len))  # placeholder for sqrtm(Sigma)

#     obj = cvxpy.Minimize(0.5*cvxpy.norm(x - x0, 2))  #define objective function
#     cons = [x.T @ r >= eps, cvxpy.norm(A@x, 2) <= delta, cvxpy.sum(x) == cash, x >= 0]      # define set of constraints

#     prob = cvxpy.Problem(obj, cons)  #setup the problem

#     # instantiate the problem for a specific value of parameters
#     A.value = sqrtm(cov)
#     r.value = r_hat
#     x0.value = x_ori

#     prob.solve(solver='SCS')  # solve the problem

#     x_opt = x.value  # the optimal variable
#     return x_opt

# def solve2(x_len, r_hat, cov, gamma0=5.0):
#     x = cvxpy.Variable(x_len)  # optimization vector variable
#     r = cvxpy.Parameter(x_len)  # placeholder for vector c
# #     Sigma = cvxpy.Parameter((x_len, x_len))  # placeholder for (Sigma)
#     gamma = cvxpy.Parameter(nonneg=True)
#     dmax = cvxpy.Parameter()  # placeholder for parameter dmax

#     risk = cvxpy.quad_form(x, cov)
#     obj = cvxpy.Maximize(cvxpy.sum(x@r) - gamma*risk)  #define objective function
        
#     prob = cvxpy.Problem(obj, 
#                    [cvxpy.sum(x) == 1, 
#                     x >= 0, x <= 0.3])

#     # instantiate the problem for a specific value of parameters
# #     Sigma.value = cov
#     r.value = r_hat
#     gamma.value = gamma0

#     prob.solve()  # solve the problem

#     x_opt = x.value  # the optimal variable
#     return x_opt

# def solve_u(x_len, x_ori, r_hat, cov, gamma0=5.0, c=0):
#     u = cvxpy.Variable(x_len)
#     x0 = cvxpy.Parameter(x_len)
#     x = cvxpy.Variable(x_len)  # optimization vector variable
#     r = cvxpy.Parameter(x_len)  # placeholder for vector c
#     #     Sigma = cvxpy.Parameter((x_len, x_len))  # placeholder for (Sigma)
#     gamma = cvxpy.Parameter(nonneg=True)

#     risk = cvxpy.quad_form(x, cov)
#     obj = cvxpy.Maximize(x.T @ r - gamma * risk)  # define objective function
#     # cons = [cvxpy.sum(u) + c_limit * cvxpy.norm(u, 1) == 0]
#     cons = [x >= 0, x <= 0.3, x==x0+u, cvxpy.sum(u) + c * cvxpy.norm(u, 1) <= 0]

#     prob = cvxpy.Problem(obj, cons)

#     # instantiate the problem for a specific value of parameters
#     #     Sigma.value = cov
#     x0.value = x_ori
#     r.value = r_hat
#     gamma.value = gamma0

#     prob.solve()  # solve the problem

#     u_opt = u.value  # the optimal variable
#     return u_opt


# def solve_sharpratio(x_len, x_ori, r_hat, r_f, cov):
#     t = cvxpy.Variable(nonneg=True)
#     gamma = cvxpy.Variable(nonneg=True)
#     u_tlide = cvxpy.Variable(x_len)
#     A = cvxpy.Parameter((x_len, x_len))
#     x0 = cvxpy.Parameter(x_len)
#     rhat = cvxpy.Parameter(x_len)
#     rf = cvxpy.Parameter()
#     x_tlide = cvxpy.Variable(x_len)
#     obj = cvxpy.Minimize(t)
# #     cons = [cvxpy.norm(A@x_tlide, 2) <= t, cvxpy.sum(u_tlide) == 0, cvxpy.sum(rhat@x_tlide) - gamma * rf == 1, x_tlide == gamma * x0 + u_tlide]
#     cons = [cvxpy.norm(A@x_tlide, 2) <= t, cvxpy.sum(u_tlide) == 0, rhat.T@x_tlide - gamma * rf == 1, x_tlide == gamma * x0 + u_tlide]
#     for i in range(x_len):
# #         cons = cons + [x_tlide[i] >= 0, x_tlide[i] <= 0.3 * gamma]
#         cons = cons + [x_tlide[i] >= 0]

#     prob = cvxpy.Problem(obj, cons)

    A.value = sqrtm(cov)
    x0.value = x_ori
    rhat.value = r_hat
    rf.value = r_f

    prob.solve()
    u_tlide_opt = u_tlide.value
    gamma_opt = gamma.value
    u_opt = u_tlide_opt / gamma_opt
    
    print(r_hat[:10])
    print(A.value[:10, :10])
    
#     print(u_tlide_opt)
#     print(gamma_opt)
#     print(u_opt)
    return u_opt