__author__ = 'zhanxy'
########## 借鉴方正证券潮汐因子的潮汐思想，我构建了一个邻域博弈强度指标：
########## 博弈强度指标：Gaming_Strength = Volume / np.sqrt((Close/Open)*(High/Low)); 邻域博弈强度指标：neighbour_GS = calculate_rolling_sum(Gaming_Strength)
########## 每日交易可以分为核心交易期与噪声交易期，核心交易期可分为以下三个阶段：
########## 1 准备博弈（邻域博弈强度前半场最低点到峰值点） 2 博弈高潮（邻域博弈强度峰值点） 3 结束博弈 （博弈强度峰值点到后半场最低点）
########## 因子：计算前20天平均 准备博弈阶段 到 结束博弈阶段 的close变化率 平均速率
########## 空头胜利，且20天平均变化速率越快（即因子值越小），则代表核心交易时间空头优势越大，长期看跌
########## 多头胜利，且20天平均变化速率越快（即因子值越大），则代表核心交易时间多头优势越大，长期看涨

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
    # print(date)
    date_record = date
    high = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "High.pkl")
    open = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Open.pkl")
    close = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Close.pkl")
    low = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Low.pkl")
    # turnover = io_obj.load_data(Path_obj.eqt_5m_bar_base / date[:4] / date[:6] / date[:8] / "Turnover.pkl")
    volume = io_obj.load_data(Path_obj.eqt_1m_bar / date[:4] / date[:6] / date[:8] / "Volume.pkl")
    open = open.reindex(stk_rtns_columns, axis='columns')
    high = high.reindex(stk_rtns_columns, axis='columns')
    close = close.reindex(stk_rtns_columns, axis='columns')
    low = low.reindex(stk_rtns_columns, axis='columns')
    # turnover = turnover.reindex(stk_rtns_columns, axis='columns')
    volume = volume.reindex(stk_rtns_columns, axis='columns')

    Gaming_Strength = volume/np.sqrt((close / open)*(high / low))
    neighbour_GS = calculate_rolling_sum(Gaming_Strength)
    close = close.iloc[4:-4]
    index = range(len(neighbour_GS))  # 索引实际含义为当天第t+5分钟
    neighbour_GS.index = index
    close.index = index

    position_matrix = np.tile(np.arange(neighbour_GS.shape[0]).reshape(-1, 1), (1, neighbour_GS.shape[1]))  # 构造位置矩阵，值为元素所在行数
    Peak_moments = np.tile(neighbour_GS.idxmax(), (neighbour_GS.shape[0], 1))  # 找到每列最大值所在行数，然后拓展成n行，便于后续比较
    

    rising_matrix = (position_matrix <= Peak_moments).astype(float)  # 逐元素比较，使得最大值前面为1，后面为0
    rising_matrix = np.where(rising_matrix == 0, np.nan, rising_matrix) # 因为volume大于等于0，如果后续都是0，无法得到正确的min
    rising_moments = np.tile((rising_matrix * neighbour_GS).idxmin().values, (neighbour_GS.shape[0], 1))  # 找到涨潮点的索引，拓展为n行

    ebbing_matrix = (position_matrix >= Peak_moments).astype(float)
    ebbing_matrix = np.where(ebbing_matrix == 0, np.nan, ebbing_matrix)
    ebbing_moments = np.tile((ebbing_matrix * neighbour_GS).idxmin().values, (neighbour_GS.shape[0], 1))

    Peak_price_matrix = (position_matrix == Peak_moments).astype(int) # 顶峰点为1，其他为0
    Peak_price = (Peak_price_matrix * close).max()  # 找到顶峰点close
    Peak_time = (Peak_price_matrix * close).idxmax().values  # 找到顶峰点索引

    Rise_price_matrix = (position_matrix == rising_moments).astype(int)  # 涨潮点为1，其他为0
    Rise_price = (Rise_price_matrix * close).max()  # 找到涨潮点close
    Rise_volume = (Rise_price_matrix * neighbour_GS).max() # 找到涨潮点volume
    Rise_time = (Rise_price_matrix * close).idxmax().values  # 找到涨潮点索引

    Ebb_price_matrix = (position_matrix == ebbing_moments).astype(int)
    Ebb_price = (Ebb_price_matrix * close).max()
    Ebb_volume = (Ebb_price_matrix * neighbour_GS).max() 
    Ebb_time = (Ebb_price_matrix * close).idxmax().values


    All_Tide_Value = (Ebb_price - Rise_price) / Rise_price / (Ebb_time - Rise_time)  # 计算潮汐价格变化率速率,还不是因子，还需最后 rolling 20天
    All_Tide_Value.fillna(0,inplace = True)
    All_Tide_Value = All_Tide_Value.to_frame().T
    All_Tide_Value.index = [date_record]
    result_ls.append(All_Tide_Value)

#alphaArgc
def main():
    stk_rtns = io_obj.load_data(Path_obj.stk_rtns_path)
    zz500 = io_obj.load_data(Path_obj.ZZ500_path)
    hs300 = io_obj.load_data(Path_obj.HS300_path)
    zz1000 = io_obj.load_data(Path_obj.ZZ1000_path)
    all_stock = io_obj.load_data(Path_obj.AllStock_path)

    stk_rtns = stk_rtns.reindex_like(all_stock) * all_stock
    stk_rtns = stk_rtns.loc[stk_rtns.index >= '2018-01-01']
    all_stock = all_stock.loc[all_stock.index >= '2018-01-01']
    dates = list(stk_rtns.index.strftime('%Y%m%d'))

    pool = multiprocessing.Pool(60)
    result_ls = multiprocessing.Manager().list()
    progress_bar = tqdm(total=len(dates))
    def update_progress(*a):
        progress_bar.update()

    for date in dates:
        pool.apply_async(multic_calc, args=(date, result_ls, stk_rtns.columns,), callback=update_progress, error_callback=print)
    pool.close()
    pool.join()
    factor_raw = pd.concat(list(result_ls), axis=0).rolling(20,min_periods=1).mean().iloc[20:]
    factor_raw.index = pd.to_datetime(factor_raw.index, format='%Y%m%d')
    factor_new = factor_raw.reindex_like(all_stock) * all_stock


    # initial
    factor_initial = factor_new
    factor_initial.to_pickle('./result/factor/GameStrength/Game_Strength_Factor.pkl')

    # market_neutral
    factor_tmp_mkt = mkt_neutralize_ls(factor_initial, zz500, hs300, zz1000, all_stock)
    factor_tmp_mkt.to_pickle('./result/factor/GameStrength/MN_Game_Strength_Factor.pkl')


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







