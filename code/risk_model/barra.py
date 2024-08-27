import pandas as pd
import sys

from optparse import OptionParser
sys.path.append('/mnt/clouddisk1/share/open_lib')
import os
import numpy as np
import shared_tools.back_test as bt
from pathlib import Path
import datetime as dt
from shared_tools.io import AZ_IO
from shared_utils.config.config_global import Env

import multiprocessing

from shared_tools.io import AZ_IO, AZ_load_hdf
import shared_tools.back_test as bt
from shared_tools.send_email import send_email

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from tqdm import tqdm
class Path:
    def __init__(self, mod):
        Env_obj = Env(mod=mod)
        self.root_path = Env_obj.BinFiles
        self.aadj_r_path = Env_obj.BinFiles.EM_Funda / "DERIVED_14/aadj_twap_r_1015.pkl"
        self.mktcap_path = Env_obj.BinFiles.EM_Funda.LICO_YS_STOCKVALUE / "AmarketCap.pkl"

        self.forb_suspend_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "SuspendedStock.pkl"
        self.forb_stpt_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "StAndPtStock.pkl"
        self.forb_limited_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "LimitedBuySellStock.pkl"

        self.sector_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1.pkl"
        # index data
        self.HS300 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "HS300_a.pkl"
        self.T800 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T800_a.pkl"
        self.T1800 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T1800_a.pkl"
        self.ZZ1000 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ1000_a.pkl"
        self.ZZ500 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ500_a.pkl"
        self.T3K = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T3K_a.pkl"
        self.all_stock = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "AllStock_a.pkl"

        self.wHS300 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "HS300_w.pkl"
        self.wT800 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T800_w.pkl"
        self.wT1800 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T1800_w.pkl"
        self.wZZ1000 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ1000_w.pkl"
        self.wZZ500 = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ500_w.pkl"
        self.wT3K = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T3K_w.pkl"

        # factor data
        self.leverage_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'leverage_factor.pkl'
        self.btop_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'btop_factor.pkl'
        self.liquid_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'liquid_factor.pkl'
        self.momentum_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'momentum_factor.pkl'
        self.size_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'size_factor.pkl'
        self.residual_vol_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'resvol_factor.pkl'
        self.beta_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'beta_factor.pkl'
        self.earn_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'earnyield_factor.pkl'
        self.nlsize_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'nlsize_factor.pkl'
        self.growth_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra' / 'growth_factor.pkl'

        self.portfolio = '/mnt/clouddisk1/share/dat_hjy/ZZ500_yushuiC3'


def single_day_regession(ii, stk_rtns, leverage_factor,
                         btop_factor, liquid_factor,
                         momentum_factor, size_factor,
                         residual_vol_factor, beta_factor,
                         earn_factor, nlsize_factor,
                         growth_factor, m_ls):
    tmp_df = pd.concat([stk_rtns.loc[stk_rtns.index[ii]],
                        leverage_factor.loc[stk_rtns.index[ii]],
                        btop_factor.loc[stk_rtns.index[ii]],
                        liquid_factor.loc[stk_rtns.index[ii]],
                        momentum_factor.loc[stk_rtns.index[ii]],
                        size_factor.loc[stk_rtns.index[ii]],
                        residual_vol_factor.loc[stk_rtns.index[ii]],
                        beta_factor.loc[stk_rtns.index[ii]],
                        earn_factor.loc[stk_rtns.index[ii]],
                        nlsize_factor.loc[stk_rtns.index[ii]],
                        growth_factor.loc[stk_rtns.index[ii]],
                        ], axis=1)
    tmp_df = tmp_df.dropna(how='any', axis=0)
    tmp_df.columns=['stk_rtns', 'leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']

    y = tmp_df['stk_rtns']
    x = tmp_df[['leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']]

    if x.shape[0] > 0:
        reg = LinearRegression().fit(x, y)
        r2 = reg.score(x, y)  # 计算R2
        result_df = pd.DataFrame({'date': [stk_rtns.index[ii]], 'r2': [r2]})
    else:
        print(stk_rtns.index[ii])
        result_df = pd.DataFrame({'date': [stk_rtns.index[ii]], 'r2': [np.nan]})

    m_ls.append(result_df)


def multi_process_regression(stk_rtns, leverage_factor,
                             btop_factor, liquid_factor,
                             momentum_factor, size_factor,
                             residual_vol_factor, beta_factor,
                             earn_factor, nlsize_factor,
                             growth_factor):
    result_ls = []
    for ii in tqdm(range(len(stk_rtns)-1)):
        single_day_regession(ii, stk_rtns, leverage_factor,
                             btop_factor, liquid_factor,
                             momentum_factor, size_factor,
                             residual_vol_factor, beta_factor,
                             earn_factor, nlsize_factor,
                             growth_factor, result_ls)
    result_df = pd.concat(list(result_ls), axis=0, sort=False)
    return result_df


def main():
    allstock_a = io_obj.load_data(Path_obj.all_stock)

    data_list = []

    for file_name in os.listdir(Path_obj.portfolio):
        if file_name.find('C3B3') != -1:
            tmp_file = pd.read_csv(f"{Path_obj.portfolio}/{file_name}", sep='|', index_col=0)
            tmp_file.index = pd.to_datetime(tmp_file.index)
            data_list.append(tmp_file)
    portfolio = pd.concat(data_list, axis=0, sort=True)

    portfolio = portfolio.sort_index()
    portfolio = portfolio.shift(-1)
    portfolio = portfolio.reindex(allstock_a.index[allstock_a.index >= '2018-01-01'])
    portfolio = portfolio.div(portfolio.sum(1), axis=0)
    portfolio['000905.SH'] = -1

    stk_rtns = io_obj.load_data(Path_obj.aadj_r_path)

    leverage_factor = io_obj.load_data(Path_obj.leverage_factor)
    btop_factor = io_obj.load_data(Path_obj.btop_factor)
    liquid_factor = io_obj.load_data(Path_obj.liquid_factor)
    momentum_factor = io_obj.load_data(Path_obj.momentum_factor)
    size_factor = io_obj.load_data(Path_obj.size_factor)
    residual_vol_factor = io_obj.load_data(Path_obj.residual_vol_factor)
    beta_factor = io_obj.load_data(Path_obj.beta_factor)
    earn_factor = io_obj.load_data(Path_obj.earn_factor)
    nlsize_factor = io_obj.load_data(Path_obj.nlsize_factor)
    growth_factor = io_obj.load_data(Path_obj.growth_factor)

    hs300_a = io_obj.load_data(Path_obj.HS300)
    zz500_a = io_obj.load_data(Path_obj.ZZ500)
    zz1000_a = io_obj.load_data(Path_obj.ZZ1000)
    T3k_a = io_obj.load_data(Path_obj.T3K)

    zz500_w = io_obj.load_data(Path_obj.wZZ500)

    stk_rtns = stk_rtns.loc[stk_rtns.index>='2018-01-01']
    stk_rtns = stk_rtns.shift(-2)

    univ_select = allstock_a.loc[allstock_a.index >= '2018-01-01']

    stk_rtns = stk_rtns.reindex_like(univ_select) * univ_select
    leverage_factor = leverage_factor.reindex_like(univ_select) * univ_select
    btop_factor = btop_factor.reindex_like(univ_select) * univ_select
    liquid_factor = liquid_factor.reindex_like(univ_select) * univ_select
    momentum_factor = momentum_factor.reindex_like(univ_select) * univ_select
    size_factor = size_factor.reindex_like(univ_select) * univ_select
    residual_vol_factor = residual_vol_factor.reindex_like(univ_select) * univ_select
    beta_factor = beta_factor.reindex_like(univ_select) * univ_select
    earn_factor = earn_factor.reindex_like(univ_select) * univ_select
    nlsize_factor = nlsize_factor.reindex_like(univ_select) * univ_select
    growth_factor = growth_factor.reindex_like(univ_select) * univ_select

    # 计算收益关于Barra的回归
    reg_df = multi_process_regression(stk_rtns, leverage_factor,
                             btop_factor, liquid_factor,
                             momentum_factor, size_factor,
                             residual_vol_factor, beta_factor,
                             earn_factor, nlsize_factor,
                             growth_factor)

    reg_df.to_pickle('./result/barra_R2.pkl')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest='ENV', default="BKT")
    parser.add_option('-d', dest='Days_back', type='int', default=None)
    (options, args) = parser.parse_args()
    # 创建io对象
    io_obj = AZ_IO(mod='pkl')
    days = options.Days_back
    env = options.ENV

    if env == 'BKT':
        if days is None:
            days = 10
        # 创建Path对象
        Path_obj = Path(mod='bkt')
    elif env == 'PRO':
        if days is None:
            days = 3
        # 创建Path对象
        Path_obj = Path(mod='pro')

    script_nm = f'EQT_Barra_{env}'
    run_start = dt.datetime.now()
    main()
    run_end = dt.datetime.now()
    total_run_time = (run_end - run_start).total_seconds()
    print('\n----------------------\n')
    print(script_nm, " Run time:", (total_run_time / 60).__round__(2), "mins")
