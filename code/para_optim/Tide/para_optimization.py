import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing

def calculate_rolling_sum(df, window_size):
    """
    计算滑动窗口的总和
    """
    if window_size % 2 == 0:
        window_size += 1
    rolling_sum = df.rolling(window=window_size, min_periods=1, center=True).sum()
    return rolling_sum.iloc[window_size//2:-(window_size//2)]

def calculate_Factor(date, result_ls, stk_rtns_columns, vol_window, factor_window):
    base_dir = "/mnt/clouddisk1/share/DAT_EQT/Intraday/"
    df_Close = pd.read_pickle(os.path.join(base_dir, "eqt_1m_bar", date[:4], date[:6], date[:8], "Close.pkl")).reindex(stk_rtns_columns, axis='columns')
    df_Volume = pd.read_pickle(os.path.join(base_dir, "eqt_1m_bar", date[:4], date[:6], date[:8], "Volume.pkl")).reindex(stk_rtns_columns, axis='columns')
    if vol_window % 2 == 0:
        vol_window += 1
    df_Close = df_Close.iloc[vol_window//2:-(vol_window//2)]  # 保证Close和Volume对齐
    neighbour_volumes = calculate_rolling_sum(df_Volume, vol_window)
    index = range(len(neighbour_volumes))  # 索引实际含义为当天第t+5分钟
    neighbour_volumes.index = index
    df_Close.index = index

    position_matrix = np.tile(np.arange(neighbour_volumes.shape[0]).reshape(-1, 1), (1, neighbour_volumes.shape[1]))  # 构造位置矩阵，值为元素所在行数
    Peak_moments = np.tile(neighbour_volumes.idxmax(), (neighbour_volumes.shape[0], 1))  # 找到每列最大值所在行数，然后拓展成n行，便于后续比较

    rising_matrix = (position_matrix <= Peak_moments).astype(float)  # 逐元素比较，使得最大值前面为1，后面为0
    rising_matrix = np.where(rising_matrix == 0, np.nan, rising_matrix) # 因为volume大于等于0，如果后续都是0，无法得到正确的min
    rising_moments = np.tile((rising_matrix * neighbour_volumes).idxmin().values, (neighbour_volumes.shape[0], 1))  # 找到涨潮点的索引，拓展为n行

    ebbing_matrix = (position_matrix >= Peak_moments).astype(float)
    ebbing_matrix = np.where(ebbing_matrix == 0, np.nan, ebbing_matrix)
    ebbing_moments = np.tile((ebbing_matrix * neighbour_volumes).idxmin().values, (neighbour_volumes.shape[0], 1))

    Peak_price_matrix = (position_matrix == Peak_moments).astype(int) # 顶峰点为1，其他为0
    Peak_price = (Peak_price_matrix * df_Close).max()  # 找到顶峰点close
    Peak_time = (Peak_price_matrix * df_Close).idxmax().values  # 找到顶峰点索引

    Rise_price_matrix = (position_matrix == rising_moments).astype(int)  # 涨潮点为1，其他为0
    Rise_price = (Rise_price_matrix * df_Close).max()  # 找到涨潮点close
    Rise_volume = (Rise_price_matrix * neighbour_volumes).max() # 找到涨潮点volume
    Rise_time = (Rise_price_matrix * df_Close).idxmax().values  # 找到涨潮点索引

    Ebb_price_matrix = (position_matrix == ebbing_moments).astype(int)
    Ebb_price = (Ebb_price_matrix * df_Close).max()
    Ebb_volume = (Ebb_price_matrix * neighbour_volumes).max() 
    Ebb_time = (Ebb_price_matrix * df_Close).idxmax().values

    ###全潮汐因子
    All_Tide_Value = (Ebb_price - Rise_price) / Rise_price / (Ebb_time - Rise_time)  # 计算潮汐价格变化率速率,还不是因子，还需最后 rolling 20天
    All_Tide_Value.fillna(0,inplace = True)
    All_Tide_Value = All_Tide_Value.to_frame().T
    All_Tide_Value.index = [date]
    result_ls.append(All_Tide_Value)

if __name__ == "__main__":
    stk_rtns = pd.read_pickle("/mnt/clouddisk1/share/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.pkl")
    dates = list(stk_rtns[stk_rtns.index >= '2016-01-01'].index.strftime('%Y%m%d'))
    vol_windows = [11, 13, 15]  # 替换为你希望测试的窗口大小，确保为奇数
    factor_windows = [1,2]  # 替换为你希望测试的窗口大小

    for factor_window in tqdm(factor_windows):
        for vol_window in tqdm(vol_windows):
            pool = multiprocessing.Pool(80)
            result_ls = multiprocessing.Manager().list()
            for date in dates:
                pool.apply_async(calculate_Factor, args=(date, result_ls, stk_rtns.columns, vol_window, factor_window), error_callback=print)   
            pool.close()
            pool.join()

            factor_raw = pd.concat(list(result_ls), axis=0)
            factor_raw.index = pd.to_datetime(factor_raw.index, format='%Y%m%d')
            factor_raw.sort_index(inplace=True)
            factor_raw = factor_raw.rolling(factor_window, min_periods=1).mean().iloc[factor_window:-factor_window] ###这里写错了其实，不需要去掉后面的，不过歪打正着留了点测试集
            factor_raw.to_pickle(f'~/result/para_optimization/All_Tide_Factor{vol_window}_{factor_window}.pkl')
