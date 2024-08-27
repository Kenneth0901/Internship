__author__ = 'zhanxy'


###  分组法回测与回归法回测
###  分组法先中性化，回归法可以先不中性化，一步到位
###  中性化：1.提取残差 2.分组标准化
###  分组法：中性化之后，分成10组来测，看收益率单调性，ic
###  回归法：带截距就wls，不带截距直接ols，因为ols性质，顺便就中性化了，看因子收益率t值，均值

import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing
from optparse import OptionParser

import warnings

from sympy import limit
from tomlkit import value

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/mnt/clouddisk1/share/open_lib/')
from shared_tools.io import AZ_IO
import shared_tools.back_test as bt
from shared_tools.send_email import send_email_df, send_email
from shared_utils.config.config_global import Env


########### paths from DAT_EQT ##############
class Path:
    def __init__(self, mod):
        Env_obj = Env(mod=mod)
        self.root_path = Env_obj.BinFiles
        self.stk_rtns_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_twap_r_1000.pkl"
        self.stk_rtns_close_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r.pkl"
        self.stk_rtns_night_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r_CLOSE_to_OPEN.pkl"
        self.stk_rtns_intraday_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r_OPEN_to_CLOSE.pkl"

        self.stk_aadj_p_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p.pkl"

        self.stk_aadj_p_open_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p_OPEN.pkl"

        self.stk_rtns_p1_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_twap_1445_on_1015.pkl"

        self.stk_twap_1000_price = Env_obj.BinFiles.EM_Funda / "DERIVED_J1/Stk_Twap_pkl/temp/Twap_1000.pkl"

        self.mktcap_path = Env_obj.BinFiles.EM_Funda.LICO_YS_STOCKVALUE / "AmarketCap.pkl"

        self.forb_suspend_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "SuspendedStock.pkl"
        self.forb_stpt_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "StAndPtStock.pkl"
        self.forb_limited_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "LimitedBuySellStock.pkl"

        self.univ_300_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "HS300_a.pkl"
        self.univ_500_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ500_a.pkl"
        self.univ_1000_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ1000_a.pkl"
        self.univ_1800_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T1800_a.pkl"
        self.univ_3000_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "T3K_a.pkl"
        self.univ_all_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "AllStock_a.pkl"

        self.factor_path = Env_obj.BinFiles.factor_pool
        ############ other paths ####################
        self.tradedates_path = Env_obj.BinFiles.EM_Funda / "DERIVED_CONSTANTS/TradeDates.pkl"


def Zscore(df, cap=None):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    target = df.sub(df_mean, axis=0).div(df_std, axis=0)
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target

    

def to_position(value, forbid_days, method='factor'):
    if method == 'position':
        daily_updated_position = value.shift(1) * forbid_days ### 每日更新后权重，如尾盘买则今日close前持仓为前一天的值，close后持仓为今天的值
        daily_updated_position.replace(np.nan, 0) ### 保证第一天持仓为0
        daily_updated_position = daily_updated_position * forbid_days
        daily_updated_position = daily_updated_position.ffill() ###出现forbid保持仓位
        return daily_updated_position
    
    elif method == 'factor':
        value = Zscore(value)
        pos_long = value[value > 0]
        pos_long = pos_long.div(pos_long.sum(1), axis=0)
        pos_long = pos_long.fillna(0)

        pos_short = value[value < 0]
        pos_short = pos_short.div(pos_short.abs().sum(1), axis=0)
        pos_short = pos_short.fillna(0)

        pos = pos_long + pos_short
        pos = pos.shift(1) * forbid_days ### 每日更新后权重，如尾盘买则今日close前持仓为前一天的值，close后持仓为今天的值
        pos.replace(np.nan, 0) ### 保证第一天持仓为0
        pos = pos * forbid_days
        pos = pos.ffill() ###出现forbid保持仓位
        return pos
        

def calculate_pnl(pos: pd.DataFrame, stk_rtns: pd.DataFrame, period: int = 1, t: float = 1.5e-3) -> pd.DataFrame:
    pos = pos.reindex_like(stk_rtns).dropna(how='all',axis=0)
    stk_rtns = stk_rtns.shift(-1) ###对齐收益率
    stk_rtns = stk_rtns.reindex_like(pos)
    if period == 1:
        pos = pos.shift(1)
        cost = pos.diff(periods=1, axis = 0).abs().sum(1) * t
        pnl = (pos * stk_rtns ).sum(1)
        pnl_cost = pnl - cost
        return pnl, pnl_cost ### 不计算/计算交易成本
    else:
        dates = list(pos.index)
        pos_list = []
        cost_list = []
        for i in range(period-1):
            pos_tmp = pos.iloc[i::period] ### 从第i天开始 每隔period取一个
            pos_tmp = pos_tmp.reindex_like(pos).ffill(limit = period-1)
            cost_tmp = pos_tmp.diff(periods=1, axis=0).abs().sum(1) * t
            pos_list.append(pos_tmp.values)
            cost_list.append(cost_tmp.values)
        pos_list = np.concatenate(pos_list, axis=0).reshape(period-1, -1, pos.shape[1])
        pnl_list = np.nansum(pos_list * stk_rtns.values, axis=2) ### 每天各股pnl求和
        pnl_cost_list = pnl_list - cost_list
        pnl, pnl_cost = pnl_list.mean(0), pnl_cost_list.mean(0)
        pnl, pnl_cost = pd.Series(pnl, index=dates), pd.Series(pnl_cost, index=dates)
        return pnl, pnl_cost


def figure(value:pd.DataFrame, forbid_days:pd.DataFrame, stk_rtns:pd.DataFrame, value_optim:pd.DataFrame=None ,period:int = 1)->None:
    
    pos_long = value[value > 0]
    pos_long = pos_long.div(pos_long.sum(1), axis=0)
    pos_long = pos_long.fillna(0)

    pos_raw = to_position(pos_long, forbid_days,'position')
    pnl_raw, pnl_raw_cost = calculate_pnl(pos_raw, stk_rtns, period)

    plt.figure(figsize=[32, 16])

    p1 = plt.subplot(211)
    p1.plot(pnl_raw.fillna(0).cumsum(), label='raw')
    p1.plot(pnl_raw_cost.fillna(0).cumsum(), label='raw_with_cost')
    p1.set_title(f'Sum, period = {period}', fontsize=20, color='black')
    p1.grid(linestyle='--')

    p2 = plt.subplot(212)
    p2.plot((1+pnl_raw.fillna(0)).cumprod(), label='raw')
    p2.plot((1+pnl_raw_cost.fillna(0)).cumprod(), label='raw_with_cost')
    p2.set_title(f'Product, period = {period}', fontsize=20, color='black')
    p2.grid(linestyle='--')
    
    if value_optim:
        pos = to_position(value_optim, forbid_days,'position')
        pnl, pnl_cost = calculate_pnl(pos, stk_rtns, period)

        p1.plot(pnl.fillna(0).cumsum(), label='optim')
        p1.plot(pnl_cost.fillna(0).cumsum(), label='optim_with_cost')
        p2.plot((1+pnl.fillna(0)).cumprod(), label='optim')
        p2.plot((1+pnl_cost.fillna(0)).cumprod(), label='optim_with_cost')

    p1.legend(fontsize=20)
    p2.legend(fontsize=20)
    plt.savefig(f'/mnt/clouddisk1/share/work_zhanxy/result/figure/2fig{period}.png')
    plt.close()


def main():
    stk_rtns = io_obj.load_data(Path_obj.stk_rtns_path)
    stk_rtns_close = io_obj.load_data(Path_obj.stk_rtns_close_path)
    stk_price = io_obj.load_data(Path_obj.stk_aadj_p_path)
    stk_price_open_open = io_obj.load_data(Path_obj.stk_aadj_p_open_path)
    stk_price_open = io_obj.load_data(Path_obj.stk_twap_1000_price)
    stk_price_open.index = pd.to_datetime(stk_price_open.index.strftime('%Y-%m-%d'))
    stk_rtn_intraday = io_obj.load_data(Path_obj.stk_rtns_intraday_path)
    stk_rtn_night = io_obj.load_data(Path_obj.stk_rtns_night_path)
    forb_suspend = io_obj.load_data(Path_obj.forb_suspend_path)
    forb_stpt = io_obj.load_data(Path_obj.forb_stpt_path)
    forb_limited = io_obj.load_data(Path_obj.forb_limited_path)
    forbid_days = forb_suspend * forb_stpt * forb_limited
    forbid_days = forbid_days.replace(0, np.nan)
    univ_300_r = stk_rtns['000300.SH']
    univ_500_r = stk_rtns['000905.SH']
    univ_300 = io_obj.load_data(Path_obj.univ_300_path)
    univ_500 = io_obj.load_data(Path_obj.univ_500_path)
    univ_1000 = io_obj.load_data(Path_obj.univ_1000_path)
    univ_3000 = io_obj.load_data(Path_obj.univ_3000_path)
    univ_all = io_obj.load_data(Path_obj.univ_all_path)
    
    new_index = stk_rtns.index[-1400:]
    stk_rtns = stk_rtns.reindex(new_index)
    stk_rtns_close = stk_rtns_close.reindex(new_index)
    stk_price = stk_price.reindex(new_index)
    stk_price_open_open = stk_price_open_open.reindex(new_index)
    stk_price_open = stk_price_open.reindex(new_index)
    stk_rtn_intraday = stk_rtn_intraday.reindex(new_index)
    stk_rtn_night = stk_rtn_night.reindex(new_index)
    forbid_days = forbid_days.reindex(new_index)
    univ_300 = univ_300.reindex(new_index)
    univ_500 = univ_500.reindex(new_index)
    univ_1000 = univ_1000.reindex(new_index)
    univ_all = univ_all.reindex(new_index)
    value = pd.read_pickle('/mnt/clouddisk1/share/work_zhanxy/result/factor/Reverse/MN_Reverse_Factor.pkl')
    value = value.dropna(how='all', axis=0)
    value = value[value.index > '2019-01-01']
    value = Zscore(value, cap = 4.5)
    # value_optim = pd.read_pickle('/mnt/clouddisk1/share/work_zhanxy/result/tmp/PW/position_new20.pkl')
    # value_optim = value_optim.dropna(how='all', axis=0)


    # figure(value, forbid_days, stk_rtns, value_optim, period= 20)
    figure(-value, forbid_days, stk_rtns, period= 20)



if __name__ == '__main__':
    data_type = 'pkl'
    io_obj = AZ_IO(mod=data_type)
    io_obj_csv = AZ_IO(mod='csv')
    Path_obj = Path(mod='bkt')
    main()


