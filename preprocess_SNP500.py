import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import exp, abs, log
from scipy.special import gamma, factorial
import os
import scipy.stats as stats
import statsmodels.api as sm
import time
import datetime

path = './stock_data/data/price/snp500/processed/'
drop_duplicate_path = './stock_data/data/price/snp500/dropped/'
filename = os.listdir(path)
'''
for fid in range(len(filename)):
    print(fid)
    _df = pd.read_csv(drop_duplicate_path + filename[fid])
    for i in range(1, len(_df)):
        _df.iloc[i, 1] = _df.iloc[i - 1, 4]
        if len(_df.iloc[i, 0]) > 10:
            _df.iloc[i, 0] = _df.iloc[i, 0][:10]
    _df.to_csv('./stock_data/data/price/snp500/shifted/' + filename[fid], index=None)

'''
'''
_df = pd.read_csv(path + filename[0], index_col='date', parse_dates=True)
_df = _df.reset_index().drop_duplicates(subset='date', keep='first')
_df['date'] = _df['date'].dt.date
_df = _df.set_index('date')
_df.to_csv(drop_duplicate_path + filename[0])
'''
_df = pd.read_csv(drop_duplicate_path + filename[0], index_col='date', parse_dates=True)
df_close = _df.loc[~_df.index.duplicated(), ['close']].rename(columns={'close': (filename[0].split('_')[0])})

# df_close = pd.DataFrame()
for fid in range(1, len(filename)):

    # print(fid, filename[fid])
    '''
    _df = pd.read_csv(path + filename[fid], index_col='date', parse_dates=True)
    # 如果你想保留第一个aa，那么keep就是first
    _df = _df.reset_index().drop_duplicates(subset='date', keep='first')
    _df['date'] = _df['date'].dt.date
    _df = _df.set_index('date')
    _df.to_csv(drop_duplicate_path + filename[fid])
    '''
    _df = pd.read_csv(drop_duplicate_path + filename[fid], index_col='date', parse_dates=True)

#     print(_df.index.duplicated().sum())
    df_close = pd.concat([df_close, _df.loc[~_df.index.duplicated(), ['close']].rename(
        columns={'close': (filename[fid].split('_')[0])})], join='outer', axis=1, sort=True)
df_close = df_close.fillna(method='bfill')

df_close = log(df_close)

for i in list(range(1650))[::-1]:
    df_close.iloc[i] -= df_close.iloc[i-1]
drop_filenames = os.listdir(drop_duplicate_path)
print(len(drop_filenames))
import pickle

with open('./stock_data/data/relation/ordered_ticker.pkl', 'rb') as f:
    all_stock = pickle.load(f)
df_close = df_close[all_stock]
with open('./stock_data/data/relation/adj_mat.pkl', 'rb') as f:
    all_mat = pickle.load(f)

end_dates = ['2014-12-26', '2015-05-21', '2015-10-13', '2016-03-08', '2016-07-29', '2016-12-20',
             '2017-05-16', '2017-10-05', '2018-04-09', '2018-08-29', '2019-01-24', '2019-05-16']

for phs in range(0, len(end_dates)):
    print(phs)
    ntrain = 300
    nval = 0
    ntest = 100
    win = 5
    nstock = len(all_stock)

    cp = df_close[:end_dates[phs]].tail(ntrain+nval+ntest)
    cp_train = cp.iloc[:ntrain, :]
    cov_train = np.cov(np.exp(cp_train.to_numpy().T))
    cp_train_std = np.std(cp_train.to_numpy(), axis=0)
    choice = np.argsort(cp_train_std)
    
    cp_val = cp.iloc[ntrain-win:ntrain+nval, :]
    cp_test = cp.iloc[ntrain+nval-win:, :]

    cp_trainx = np.zeros((ntrain - win, win * nstock))
    cp_trainy = np.zeros((ntrain - win, nstock))

    for i in range(win, ntrain):
        cp_trainy[i - win] = cp_train.to_numpy()[i]
        for s in range(nstock):
            cp_trainx[i - win, s * win:(s + 1) * win] = cp_train.to_numpy()[i - win:i, s]
    
    cp_valx = np.zeros((nval, win * nstock))
    cp_valy = np.zeros((nval, nstock))

    for i in range(win, nval + win):
        cp_valy[i - win] = cp_val.to_numpy()[i]
        for s in range(nstock):
            cp_valx[i - win, s * win:(s + 1) * win] = cp_val.to_numpy()[i - win:i, s]
    

    cp_testx = np.zeros((ntest, win * nstock))
    cp_testy = np.zeros((ntest, nstock))

    for i in range(win, ntest + win):
        cp_testy[i - win] = cp_test.to_numpy()[i]
        for s in range(nstock):
            cp_testx[i - win, s * win:(s + 1) * win] = cp_test.to_numpy()[i - win:i, s]

    np.savez('./stock_data/stock_phase%02d_lb%d' % (phs, win), rt_trainx=cp_trainx, rt_trainy=cp_trainy,  \
             rt_valx=cp_valx, rt_valy=cp_valy,  \
             rt_testx=cp_testx, rt_testy=cp_testy, choice=choice, cov_train=cov_train)
