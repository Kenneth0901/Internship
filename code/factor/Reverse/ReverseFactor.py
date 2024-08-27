__author__ = 'zhanxy'

from unittest import result
import pandas as pd
import numpy as np
import datetime as dt
from optparse import OptionParser
import itertools
import multiprocessing
import math

import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from functools import reduce
import pickle
import sys
from tqdm import tqdm
sys.path.append('/mnt/clouddisk1/share/open_lib/')
from shared_tools.io import AZ_IO, AZ_load_hdf
import shared_tools.back_test as bt
# from shared_tools.Factor_Evaluation_Common_Func import AZ_Alpha_Evaluation, AZ_Factor_Evaluation
# from shared_tools.send_email import send_email
from shared_utils.config.config_global import Env
from sqlalchemy import create_engine
import gzip
import pickle
import gc
########### paths from DAT_EQT ##############
class Path:
    def __init__(self, mod):
        Env_obj = Env(mod=mod)
        # self.root_path = Env_obj.BinFiles
        # self.tradedates_path = Env_obj.BinFiles.EM_Funda / "DERIVED_CONSTANTS/TradeDates.pkl"
        self.stk_rtns_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r.pkl"
        # self.close_path    = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p.pkl"
        # self.open_path     = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p_OPEN.pkl"
        # self.high_path     = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p_HIGH.pkl"
        # self.low_path      = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p_LOW.pkl"
        self.eqt_1m_bar = Env_obj.BinFiles.Intraday.eqt_1m_bar
        self.eqt_tick_transaction = Env_obj.BinFiles.Intraday / "eqt_tick_transaction"
        self.ZZ500_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ500_a.pkl"
        self.HS300_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "HS300_a.pkl"
        self.ZZ1000_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "ZZ1000_a.pkl"
        self.AllStock_path = Env_obj.BinFiles.EM_Funda.DERIVED_141 / "AllStock_a.pkl"
        self.mktcap_path = Env_obj.BinFiles.EM_Funda.LICO_YS_STOCKVALUE / "AmarketCap.pkl"
        self.float_free_shareout_path = Env_obj.BinFiles.EM_Funda.LICO_YS_STOCKVALUE / "AmarketCap.pkl"
        self.amount_path = Env_obj.BinFiles.EM_Funda.TRAD_SK_DAILY_JC / "TVALCNY.pkl"
        self.returns_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r.pkl"

        self.ht_consumer_essential_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_consumer_essential.pkl"
        self.ht_consumer_optional_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_consumer_optional.pkl"
        self.ht_cyclical_manufacturing_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_cyclical_manufacturing.pkl"
        self.ht_cyclical_material_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_cyclical_material.pkl"
        self.ht_cyclical_resource_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_cyclical_resource.pkl"
        self.ht_financial_financial_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_financial_financial.pkl"
        self.ht_growth_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_growth_TMT.pkl"
        self.ht_others_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_others_others.pkl"
        self.ht_others_utilities_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "Huatai_2021_Level2_others_Utilities.pkl"
        self.sw_consumer_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1_consumer.pkl"
        self.sw_cyclical_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1_cyclical.pkl"
        self.sw_financial_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1_financial.pkl"
        self.sw_growth_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1_growth.pkl"
        self.sw_others_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1_others.pkl"
        self.sw2021_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1.pkl"

        self.forb_suspend_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "SuspendedStock.pkl"
        self.forb_stpt_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "StAndPtStock.pkl"
        self.forb_limited_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "LimitedBuySellStock.pkl"
        base_dir = Env_obj.BinFilesEQT.parent
        self.transaction_path = base_dir / "dat_zhanxy" / "trans_price.pkl"

def mkt_neutralize_ls(factor_tmp, zz500, hs300, zz1000, all_stock):
    factor_tmp = factor_tmp.dropna(how='all', axis=0)

    zz500 = zz500.reindex(factor_tmp.index)
    hs300 = hs300.reindex(factor_tmp.index)
    zz1000 = zz1000.reindex(factor_tmp.index)

    all_stock_left = all_stock.fillna(0) - zz500.reindex_like(all_stock).fillna(0) - \
                     hs300.reindex_like(all_stock).fillna(0) - zz1000.reindex_like(all_stock).fillna(0)
    all_stock_left = all_stock_left.replace(0, np.nan)

    factor_tmp_zz500 = factor_tmp * zz500
    factor_tmp_hs300 = factor_tmp * hs300
    factor_tmp_zz1000 = factor_tmp * zz1000
    factor_tmp_other = factor_tmp * all_stock_left

    factor_zscore_hs300 = bt.AZ_Row_zscore(factor_tmp_hs300, cap=4.5)
    factor_zscore_zz1000 = bt.AZ_Row_zscore(factor_tmp_zz1000, cap=4.5)
    factor_zscore_zz500 = bt.AZ_Row_zscore(factor_tmp_zz500, cap=4.5)
    factor_zscore_other = bt.AZ_Row_zscore(factor_tmp_other, cap=4.5)

    factor_tmp = factor_zscore_hs300.fillna(0) + factor_zscore_zz1000.fillna(0) + \
                 factor_zscore_zz500.fillna(0) + factor_zscore_other.fillna(0)
    factor_tmp = factor_tmp.replace(0, np.nan)
    return factor_tmp


def add(df1, df2):
    return df1.add(df2, fill_value=0)


def calculate_rolling_sum(df):
    """
    计算滑动窗口的总和
    """
    window_size = 9
    rolling_sum = df.rolling(window=window_size, min_periods=1, center=True).sum()
    return rolling_sum.iloc[4:-4]

# 解压缩对象
def decompress_object(filename):
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def calculate_trans_price(date, result_ls, stk_rtns_columns):
    df = decompress_object(Path_obj.eqt_tick_transaction/ f"{date}_tran.pkl.gz")
    Trans_price = [df[stock]['Price'].quantile(0.8) for stock in df.keys()]
    Trans_price = pd.DataFrame(Trans_price,index = df.keys(), columns=[date]).T.reindex(columns=stk_rtns_columns)
    result_ls.append(Trans_price)



def multic_calc(date, result_ls, stk_rtns_columns):
    # print(date)
    date_record = date
    # high = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "High.pkl")
    # open = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Open.pkl")
    # close = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Close.pkl")
    # low = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Low.pkl")
    # turnover = io_obj.load_data(Path_obj.eqt_5m_bar_base / date[:4] / date[:6] / date[:8] / "Turnover.pkl")
    # volume = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Volume.pkl")

    # open = open.reindex(stk_rtns_columns, axis='columns')
    # high = high.reindex(stk_rtns_columns, axis='columns')
    # close = close.reindex(stk_rtns_columns, axis='columns')
    # low = low.reindex(stk_rtns_columns, axis='columns')
    # turnover = turnover.reindex(stk_rtns_columns, axis='columns')
    # volume = volume.reindex(stk_rtns_columns, axis='columns')   
    tmp_rtns = io_obj.load_data(Path_obj.stk_rtns_path)[:date][-20:]
    trans_price = io_obj.load_data(Path_obj.transaction_path)[:date][-20:]

    tmp_rtns = tmp_rtns.reindex_like(trans_price)
    boundary = trans_price.mean(0)
    reverse = ((trans_price >= boundary) * tmp_rtns).mean(0)
    momentum =  ((trans_price < boundary) * tmp_rtns).mean(0)
    value = reverse - momentum

    value = value.fillna(0)
    value = value.to_frame().T
    value.index = [date_record]

    result_ls.append(value)

#alphaArgc
def main():
    stk_rtns = io_obj.load_data(Path_obj.stk_rtns_path)
    zz500 = io_obj.load_data(Path_obj.ZZ500_path)
    hs300 = io_obj.load_data(Path_obj.HS300_path)
    zz1000 = io_obj.load_data(Path_obj.ZZ1000_path)
    all_stock = io_obj.load_data(Path_obj.AllStock_path)
    trans_price = io_obj.load_data(Path_obj.transaction_path)

    stk_rtns = stk_rtns.reindex_like(all_stock) * all_stock
    stk_rtns = stk_rtns.loc[stk_rtns.index >= '2018-01-01']
    all_stock = all_stock.loc[all_stock.index >= '2018-01-01']
    dates =stk_rtns[stk_rtns.index < '2024-07-15'].index.strftime('%Y%m%d')
    print(dates)
    # dates = list(trans_price.index.strftime('%Y%m%d')[19:])       


#计算transprice
##########################################################
    # trans_price_ls = []
    # for date in tqdm(dates[:]):
    #     calculate_trans_price(date, trans_price_ls, stk_rtns.columns,)
  ###################
    
    # pool = multiprocessing.Pool(20)
    # trans_price_ls = multiprocessing.Manager().list()
    # progress_bar = tqdm(total=len(dates))
    # def update_progress(*a):
    #     progress_bar.update()

    # for date in dates[:]:
    #     pool.apply_async(calculate_trans_price, args=(date, trans_price_ls, stk_rtns.columns,), callback=update_progress, error_callback=print)
    # pool.close()
    # pool.join()
    
    # trans_price = np.concatenate(list(trans_price_ls), axis=0)
    # trans_price = pd.DataFrame(trans_price,index=pd.to_datetime(dates[:], format='%Y%m%d'),columns=stk_rtns.columns)
    # trans_price.to_pickle("/mnt/clouddisk1/share/dat_zhanxy/trans_price.pkl")

##########################################################

        

    pool = multiprocessing.Pool(20)
    result_ls = multiprocessing.Manager().list()
    progress_bar = tqdm(total=len(dates))
    def update_progress(*a):
        progress_bar.update()

    for date in dates:
        pool.apply_async(multic_calc, args=(date, result_ls, stk_rtns.columns,), callback=update_progress, error_callback=print)
    pool.close()
    pool.join()
    factor_raw = pd.concat(list(result_ls), axis=0)
    factor_raw.index = pd.to_datetime(factor_raw.index, format='%Y%m%d')
    factor_new = factor_raw.reindex_like(all_stock) * all_stock


    # initial
    factor_initial = factor_new
    factor_initial.to_pickle('./result/factor/Reverse/Reverse_Factor.pkl')

    # market_neutral
    factor_tmp_mkt = mkt_neutralize_ls(factor_initial, zz500, hs300, zz1000, all_stock)
    factor_tmp_mkt.to_pickle('./result/factor/Reverse/MN_Reverse_Factor.pkl')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest='ENV', default="BKT")
    parser.add_option('-d', dest='Days_back', type='int', default=0)
    (options, args) = parser.parse_args()
    # 创建io对象
    io_obj = AZ_IO(mod='pkl')
    io_obj_csv = AZ_IO(mod='csv')
    days = options.Days_back
    env = options.ENV
    if env == 'BKT':
        if days is None:
            days = 5
        # 创建Path对象
        Path_obj = Path(mod='bkt')
    elif env == 'PRO':
        if days is None:
            days = 3
        # 创建Path对象
        Path_obj = Path(mod='pro')
    script_nm = f'EQT_Quality_Factor_GROWTH_{env}'

    run_start = dt.datetime.now()
    main()
    run_end = dt.datetime.now()
    total_run_time = (run_end - run_start).total_seconds()


    print('\n----------------------\n')
    print(script_nm, " Run time:", (total_run_time / 60).__round__(2), "mins")







