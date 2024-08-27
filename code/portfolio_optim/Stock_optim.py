from mosek.fusion import *

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
sys.path.append('/mnt/clouddisk1/share/open_lib')

import os
import shared_tools.back_test as bt
from pathlib import Path
import datetime as dt
from shared_tools.io import AZ_IO
from shared_utils.config.config_global import Env
import shared_tools.back_test as bt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import multiprocessing


class Path:
    def __init__(self, mod):
        Env_obj = Env(mod=mod)
        self.root_path = Env_obj.BinFiles
        self.stk_rtns_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r.pkl"
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

        # Barra data
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

        base_dir = Env_obj.BinFilesEQT.parent
        self.factor = base_dir/"dat_zhanxy" / "MN_Game_Strength_Factor.pkl"
        self.spe_rtn = base_dir/"work_zhanxy" / "result/tmp/spe_rtn.pkl"
        self.sys_rtn = base_dir/"work_zhanxy" / "result/tmp/sys_rtn.pkl"



def get_barra_rtn(barra_rtns_ls, res_ls): ###这里的两个rtn已经shift（-1）
    allstock_a = io_obj.load_data(Path_obj.all_stock)
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

    stk_rtns = stk_rtns.loc[stk_rtns.index>='2018-01-01']
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

    stk_rtns = stk_rtns.shift(-1).dropna(how='all', axis=0)
    dates = stk_rtns.index
    stocks = stk_rtns.columns
    for date in tqdm(dates[:]):
        tmp = pd.concat([stk_rtns.loc[date], leverage_factor.loc[date], btop_factor.loc[date], liquid_factor.loc[date], momentum_factor.loc[date], 
                         size_factor.loc[date], residual_vol_factor.loc[date], beta_factor.loc[date], 
                         earn_factor.loc[date], nlsize_factor.loc[date], growth_factor.loc[date]], axis=1)
        tmp.columns=['stk_rtns', 'leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']
        tmp = tmp.dropna(how='any', axis=0)    ### index 是 stock，column 是因子
        y = tmp['stk_rtns']
        x = tmp[['leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']]
        if x.shape[0] != 0:
            reg = LinearRegression().fit(x, y)
            barra_rtns = pd.Series(reg.coef_)
            barra_rtns = barra_rtns.to_frame().T
            barra_rtns.index = [date]
            barra_rtns.columns = ['leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']
            barra_rtns_ls.append(barra_rtns)

            res = y - reg.predict(x)
            res = res.to_frame().reindex(index = stocks).T
            res.index = [date]
            res_ls.append(res)
        else:
            print(date)


def calculate_risk(sys_rtn, spe_rtn, risk_factor):   ### factor index应该为stock，columns为因子
    spe_rtn = spe_rtn.loc[:, spe_rtn.notna().sum() > 10]
    risk_factor = risk_factor.reindex(index = spe_rtn.columns).dropna(how='any', axis = 0)
    spe_rtn = spe_rtn.reindex(columns = risk_factor.index)
    cov_F = sys_rtn.cov()
    cov_S = np.diag(spe_rtn.var())
    risk = risk_factor @ cov_F @ risk_factor.T + cov_S  
    risk.index = spe_rtn.columns
    risk.columns = spe_rtn.columns
    return risk

def get_current_factor_value(date, leverage, btop, liquid, momentum, size, residual, beta, earn, nlsize, growth):
    factor_value = pd.concat([leverage.loc[date], btop.loc[date], liquid.loc[date], momentum.loc[date], size.loc[date], residual.loc[date], \
                        beta.loc[date], earn.loc[date], nlsize.loc[date], growth.loc[date]], axis=1)
    factor_value.columns=['leverage', 'btop', 'liquidity','momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize', 'growth']
    return factor_value ### index为stock，columns为因子


def optim(factor, beta_exposure, Cov_Matrix):
    # 读取数据
    n = Cov_Matrix.shape[0]
    factor = factor.reindex(index=Cov_Matrix.index)
    beta_exposure = beta_exposure.reindex(index=Cov_Matrix.index)
    
    # 构建模型
    with Model("Portfolio Optimization") as M:
        x = M.variable('x', n, Domain.greaterThan(0.0))
        b = factor.values
        beta = beta_exposure.values
        Q = Cov_Matrix.values

        # Cholesky 分解
        Q_cholesky = np.linalg.cholesky(Q)
        
        # 引入辅助变量 t
        t = M.variable("t", 1, Domain.unbounded())
        
        # 构建二阶锥约束: || Q_cholesky * x ||_2 <= t
        y = Expr.mul(Q_cholesky, x)
        M.constraint(Expr.vstack(t, y), Domain.inQCone())
        
        # 其他约束
        M.constraint(t, Domain.lessThan(0.5))
        M.constraint(Expr.sum(x), Domain.equalsTo(1))
        M.constraint(Expr.dot(beta, x), Domain.lessThan(0.2))
        M.constraint(Expr.dot(beta, x), Domain.greaterThan(-0.2))
        M.constraint(x, Domain.lessThan(0.05))
        # 目标函数
        expected_return = Expr.dot(b, x)
        risk_term = Expr.mul(0.5, t)
        objective = Expr.sub(expected_return, risk_term)
        M.objective(ObjectiveSense.Maximize, objective)
        
        # 求解
        M.setSolverParam('mioMaxTime', 60.0)
        M.solve()
        
        # 获取结果
        weights = pd.Series(x.level(), index=factor.index)
    
    return weights


def calcu_daily_position(date, pos_ls, stk_rtns_columns):
    allstock_a = io_obj.load_data(Path_obj.all_stock)
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

    univ_select = allstock_a.loc[allstock_a.index >= '2018-01-01']
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

    sys_rtn = io_obj.load_data(Path_obj.sys_rtn)
    spe_rtn = io_obj.load_data(Path_obj.spe_rtn)

    factor_Portfolio = pd.read_pickle('/mnt/clouddisk1/share/work_zhanxy/result/factor/GameStrength/MN_Game_Strength_Factor.pkl')
    # factor_Portfolio = pd.read_pickle('/mnt/clouddisk1/share/work_zhanxy/result/factor/Reverse/MN_Reverse_Factor.pkl')

    sys_rtn = sys_rtn.dropna(how ='all', axis = 1)[:date].iloc[-250:] ### 计算权重时无法得到当天的barra统计量以及残差，取不到date
    spe_rtn = spe_rtn.dropna(how ='all', axis = 1)[:date].iloc[-250:] ### 类似于shift（-1）
    factor_Portfolio = factor_Portfolio.loc[date]
    try:
        Current_factor_value = get_current_factor_value(date, leverage_factor, btop_factor, liquid_factor, momentum_factor, \
                                                        size_factor, residual_vol_factor , beta_factor, earn_factor, nlsize_factor, growth_factor)
        
        risk = calculate_risk(sys_rtn, spe_rtn, Current_factor_value)
        weights = optim(factor_Portfolio, beta_factor.loc[date], risk).to_frame().T
        weights = weights.reindex(columns = stk_rtns_columns)
        # weights.index = [date]
        # weights.index = pd.to_datetime(weights.index)
        pos_ls.append(weights)
        return pos_ls
    except:
        return pos_ls




if __name__ == '__main__':
    # 创建io对象
    data_type = 'pkl'
    io_obj = AZ_IO(mod=data_type)
    io_obj_csv = AZ_IO(mod='csv')
    Path_obj = Path(mod='bkt')

    ### 计算系统风险和独特风险
    res_ls = []
    beta_rtn_ls = []
    get_barra_rtn(beta_rtn_ls, res_ls)
    
    sys_rtn = pd.concat(beta_rtn_ls, axis=0)
    sys_rtn.to_pickle('./result/tmp/sys_rtn.pkl')

    spe_rtn = pd.concat(res_ls,axis=0)
    spe_rtn = spe_rtn.dropna(how='all', axis = 1)
    spe_rtn.to_pickle('./result/tmp/spe_rtn.pkl')


    ### 优化
    # dates = dates[:-20]
    
    ############################# 记得来改

    # 单线程for loop调试
    # result_ls = []
    # for date in tqdm(dates):
    #     calcu_daily_position(date, result_ls, stk_rtns)
    #     print(result_ls)

    ### 多进程
    pool = multiprocessing.Pool(10)
    result_ls = multiprocessing.Manager().list()

    allstock_a = io_obj.load_data(Path_obj.all_stock)
    run_start = dt.datetime.now()
    stk_rtns = io_obj.load_data(Path_obj.aadj_r_path)
    stk_rtns = stk_rtns.dropna(how='all', axis = 0)
    allstock_a = allstock_a.loc[allstock_a.index>'2020-01-01']
    dates = list(stk_rtns[stk_rtns.index > '2024-08-05'].index.strftime('%Y%m%d'))

    progress_bar = tqdm(total=len(dates))

    def update_progress(*a):
        progress_bar.update()

    for date in dates:
        pool.apply_async(calcu_daily_position, args = (date, result_ls, stk_rtns.columns,), callback=update_progress, error_callback=print)
    pool.close()
    pool.join()

    print('开始拼接')

    # position = pd.concat(list(result_ls), axis=0)
    position = np.concatenate(result_ls, axis=0)
    position = pd.DataFrame(position,index=pd.to_datetime(dates, format='%Y%m%d'),columns=stk_rtns.columns)
    # position.index = pd.to_datetime(dates, format='%Y%m%d')
    position = position.reindex_like(allstock_a) * allstock_a
    print('开始储存')
    position.to_pickle('/mnt/clouddisk1/share/work_zhanxy/result/tmp/PW/position_daily.pkl')

    run_end = dt.datetime.now()
    total_run_time = (run_end - run_start).total_seconds()
    print('\n----------------------\n')     
    print("Run time:", (total_run_time / 60).__round__(2), "mins")