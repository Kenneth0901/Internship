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

sys.path.append('/mnt/clouddisk1/share/open_lib/')
from shared_tools.io import AZ_IO, AZ_load_hdf
import shared_tools.back_test as bt
# from shared_tools.Factor_Evaluation_Common_Func import AZ_Alpha_Evaluation, AZ_Factor_Evaluation
# from shared_tools.send_email import send_email
from shared_utils.config.config_global import Env
from sqlalchemy import create_engine





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
        self.eqt_1m_bar_base = Env_obj.BinFiles.Intraday.eqt_1m_bar
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

        self.factor_path = Env_obj.BinFiles.factor_pool / 'temp'




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

def multic_calc(date, result_ls, stk_rtns_columns):
    date_record = date
    volume = io_obj.load_data(Path_obj.eqt_1m_bar_base / date[:4] / date[:6] / date[:8] / "Volume.pkl")
    close = io_obj.load_data(Path_obj.eqt_1m_bar_base / date[:4] / date[:6] / date[:8] / "Close.pkl")
    close = close.iloc[4:-4]  # 保证Close和Volume对齐
    volume = volume.reindex(stk_rtns_columns, axis='columns')
    close = close.reindex(stk_rtns_columns, axis='columns')
    neighbour_volumes = calculate_rolling_sum(volume)
    index = range(len(neighbour_volumes))  # 索引实际含义为当天第t+5分钟
    neighbour_volumes.index = index
    close.index = index

    position_matrix = np.tile(np.arange(neighbour_volumes.shape[0]).reshape(-1, 1), (1, neighbour_volumes.shape[1]))  # 构造位置矩阵，值为元素所在行数
    peak_moments = np.tile(neighbour_volumes.idxmax(), (neighbour_volumes.shape[0], 1))  # 找到每列最大值所在行数，然后拓展成n行，便于后续比较
    rising_matrix = (position_matrix <= peak_moments).astype(float)  # 逐元素比较，使得最大值前面为1，后面为0
    rising_matrix = np.where(rising_matrix == 0, np.nan, rising_matrix) # 因为volume大于等于0，如果后续都是0，无法得到正确的min
    rising_moments = np.tile((rising_matrix * neighbour_volumes).idxmin().values, (neighbour_volumes.shape[0], 1))  # 找到涨潮点的索引，拓展为n行

    ebbing_matrix = (position_matrix >= peak_moments).astype(float)
    ebbing_matrix = np.where(ebbing_matrix == 0, np.nan, ebbing_matrix)
    ebbing_moments = np.tile((ebbing_matrix * neighbour_volumes).idxmin().values, (neighbour_volumes.shape[0], 1))

    Rise_price_matrix = (position_matrix == rising_moments).astype(int)  # 涨潮点为1，其他为0
    Rise_price = (Rise_price_matrix * close).max()  # 找到涨潮点close
    Rise_time = (Rise_price_matrix * close).idxmax().values  # 找到涨潮点索引

    Ebb_price_matrix = (position_matrix == ebbing_moments).astype(int)
    Ebb_price = (Ebb_price_matrix * close).max()
    Ebb_time = (Ebb_price_matrix * close).idxmax().values
    factor_tmp = (Ebb_price - Rise_price) / Rise_price / (Ebb_time - Rise_time)  # 计算潮汐价格变化率速率
    factor_tmp.fillna(0,inplace = True)
    factor_tmp = factor_tmp.to_frame().T
    factor_tmp.index = [date_record]
    result_ls.append(factor_tmp) ### 目前不是因子，需要整体做完之后rolling 20



def main():
    stk_rtns = io_obj.load_data(Path_obj.stk_rtns_path)
    zz500 = io_obj.load_data(Path_obj.ZZ500_path)
    hs300 = io_obj.load_data(Path_obj.HS300_path)
    zz1000 = io_obj.load_data(Path_obj.ZZ1000_path)
    all_stock = io_obj.load_data(Path_obj.AllStock_path)

    stk_rtns = stk_rtns.reindex_like(all_stock) * all_stock
    stk_rtns = stk_rtns.loc[stk_rtns.index >= '2016-01-01']
    dates = list(stk_rtns.index.strftime('%Y%m%d'))

    if days == 0:
        pool = multiprocessing.Pool(30)
        result_ls = multiprocessing.Manager().list()
        for date in dates:
            pool.apply_async(multic_calc, args=(date, result_ls, stk_rtns.columns, ), error_callback=print)
        pool.close()
        pool.join()
        factor_raw = pd.concat(list(result_ls), axis=0)
        factor_raw = factor_raw.rolling(20,min_periods=1).mean().iloc[20:-20] ###补充操作
        factor_raw.index = pd.to_datetime(factor_raw.index, format='%Y%m%d')
        factor_new = factor_raw.reindex_like(all_stock) * all_stock
    else:
        dates = dates[-days:]
        pool = multiprocessing.Pool(10)
        result_ls = multiprocessing.Manager().list()
        for date in dates:
            pool.apply_async(multic_calc, args=(date, result_ls, stk_rtns.columns,), error_callback=print)
        pool.close()
        pool.join()
        factor_raw = pd.concat(list(result_ls), axis=0)
        factor_raw = factor_raw.rolling(20,min_periods=1).mean().iloc[20:-20] ###补充操作
        factor_raw.index = pd.to_datetime(factor_raw.index, format='%Y%m%d')
        factor_raw = factor_raw.reindex_like(all_stock) * all_stock
        factor_raw = factor_raw.iloc[-days:]
        factor_new = factor_raw ###补充
        # factor_old = pd.read_pickle(Path_obj.factor_path / "Tide_Factor")
        # factor_new = pd.concat([factor_old, factor_raw])
        # factor_new = factor_new[~factor_new.index.duplicated(keep='last')].sort_index()
        # factor_new = factor_new.reindex_like(all_stock) * all_stock

    # raw
    

    # initial
    factor_initial = factor_new
    factor_initial.to_pickle("~/factor/Tide_Factor_f01.pkl")

    # market_neutral
    factor_tmp_mkt = mkt_neutralize_ls(factor_initial, zz500, hs300, zz1000, all_stock)
    factor_tmp_mkt.to_pickle("~/factor/Tide_Factor_f02.pkl")



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






