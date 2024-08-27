__author__ = 'jolie'

import pandas as pd
import datetime as dt
import numpy as np
from optparse import OptionParser
import scipy.stats as st  #用于单因素t检验

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
cmap = mpl.colormaps['tab10'].colors

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
        self.stk_rtns_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_twap_r_1000.pkl" # 基于 9:30-10:00 区间内的twap计算的涨跌幅
        self.stk_rtns_close_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r.pkl" # 涨跌幅
        self.stk_rtns_night_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r_CLOSE_to_OPEN.pkl"  # 隔夜收益
        self.stk_rtns_intraday_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_r_OPEN_to_CLOSE.pkl"  # 日内收益

        self.stk_aadj_p_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p.pkl"  # 收盘价
        self.stk_aadj_p_open_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_p_OPEN.pkl" # 开盘价
        self.stk_rtns_p1_path = Env_obj.BinFiles.EM_Funda.DERIVED_14 / "aadj_twap_1445_on_1015.pkl" # t日1015twap 与 t-1日1445twap 计算的涨跌幅
        self.stk_twap_1000_price = Env_obj.BinFiles.EM_Funda / "DERIVED_J1/Stk_Twap_pkl/temp/Twap_1000.pkl" #  9:30-10:00 区间内的twap
        self.mktcap_path = Env_obj.BinFiles.EM_Funda.LICO_YS_STOCKVALUE / "AmarketCap.pkl"

        self.forb_suspend_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "SuspendedStock.pkl"
        self.forb_stpt_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "StAndPtStock.pkl"
        self.forb_limited_path = Env_obj.BinFiles.EM_Funda.DERIVED_01 / "LimitedBuySellStock.pkl"

        self.univ_path = Env_obj.BinFiles.EM_Funda.DERIVED_141  # 股票池所在文件夹
        self.factor_path = Env_obj.BinFiles.factor_pool
        ############ other paths ####################
        self.tradedates_path = Env_obj.BinFiles.EM_Funda / "DERIVED_CONSTANTS/TradeDates.pkl"
        self.sw_indus_l1_path = Env_obj.BinFiles.EM_Funda.DERIVED_12 / "ShenWan2021_Level1.pkl"  # 所属申万 1 级行业


UNIV_FILE = {'CSI300':'HS300_a.pkl',
             'CSI500':'ZZ500_a.pkl',
             'CSI1000':'ZZ1000_a.pkl',
             'T800':'T800_a.pkl',
             'T1800':'T1800_a.pkl',
             'T3000':'T3K_a.pkl',
             'ALL_A':'AllStock_a.pkl'}

SHENWAN_2021_= {'11': "农林牧渔",
                '34': "食品饮料",
                '24': "有色金属",
                '45': "商业贸易",
                '62': "建筑装饰",
                '33': "家用电器",
                '46': "休闲服务",
                '61': "建筑材料",
                '43': "房地产",
                '51': "综合",
                '49': "非银金融",
                '28': "汽车",
                '48': "银行",
                '35': "纺织服饰",
                '22': "基础化工",
                '64': "机械设备",
                '37': "医药生物",
                '63': "电力设备",
                '42': "交通运输",
                '36': "轻工制造",
                '41': "公用事业",
                '27': "电子",
                '23': "钢铁",
                '65': "国防军工",
                '71': "计算机",
                '72': "传媒",
                '73': "通信",
                '74': "煤炭",
                '75': "石油石化",
                '76': "环保",
                '77': "美容护理"}


SHENWAN_2021_ = {'11': "Agriculture F~ A~ Fishing",
                '34': "Food and Beverages",
                '24': "Nonferrous Metals",
                '45': "Commerce and Trade",
                '62': "Construction and Decoration",
                '33': "Home Appliances",
                '46': "Leisure and Services",
                '61': "Building Materials",
                '43': "Real Estate",
                '51': "Conglomerates",
                '49': "Non-Bank Financials",
                '28': "Automobiles",
                '48': "Banks",
                '35': "Textiles and Apparel",
                '22': "Chemicals",
                '64': "Machinery and Equipment",
                '37': "Pharmaceuticals and Biotechnology",
                '63': "Electrical Equipment",
                '42': "Transportation and Logistics",
                '36': "Light Industry Manufacturing",
                '41': "Utilities",
                '27': "Electronics",
                '23': "Steel",
                '65': "National Defense and Military",
                '71': "Computers and Technology",
                '72': "Media",
                '73': "Telecommunications",
                '74': "Coal",
                '75': "Oil and Petrochemicals",
                '76': "Environmental Protection",
                '77': "Beauty Care"}


SHENWAN_2021= {'11': "Agri~ F~ A~ Fishing",
                '34': "Food and Beverages",
                '24': "Nonferrous Metals",
                '45': "Commerce and Trade",
                '62': "Construction and Decor~",
                '33': "Home Appliances",
                '46': "Leisure and Services",
                '61': "Building Materials",
                '43': "Real Estate",
                '51': "Conglomerates",
                '49': "Non-Bank Finan~",
                '28': "Automobiles",
                '48': "Banks",
                '35': "Textiles and Apparel",
                '22': "Chemicals",
                '64': "Machinery and Equip~",
                '37': "Pharmaceu~ and Bio~",
                '63': "Electr~ Equipment",
                '42': "Transpor~ and Logistics",
                '36': "Light Industry Manuf~",
                '41': "Utilities",
                '27': "Electronics",
                '23': "Steel",
                '65': "National ~ Military",
                '71': "Computers Tech~",
                '72': "Media",
                '73': "Telecommun~",
                '74': "Coal",
                '75': "Oil and Petrochemi~",
                '76': "Envir~ Protection",
                '77': "Beauty Care"}



ANNUAL_PARAMS = {1:250, 5:50, 10:25, 20:12, 60:4, 15:16, 40:6}


def extreme_process_MAD(factor:pd.DataFrame,
                        num: int = 3) -> pd.DataFrame:
    '''
    异常值处理
    '''
    factor_ = factor.replace(0,np.nan)
    median = factor_.median(axis=1)  # 获取中位数
    mad = abs(factor_.sub(median,axis=0)).median(axis=1)
    ret = factor.clip(lower=median-num*1.4826*mad, upper=median+num*1.4826*mad, axis=0)
    return ret


def neutralize_process_zscore(factor_tmp: pd.DataFrame,
                              group: pd.DataFrame,
                              market_cap: pd.DataFrame) -> pd.DataFrame:
    '''
    行业内按市值权重进行标准化处理
    '''
    if market_cap is None:
        market_cap = factor_tmp.where(pd.isnull(factor_tmp), 1)
    group_type = group.stack().unique()
    f_scale = factor_tmp.where(pd.isnull(factor_tmp), 0)
    for g in group_type:
        group_ = group[group==g]
        group_ = group_.where(pd.isnull(group_), 1)
        factor = factor_tmp * group_
        weight = market_cap * group_
        weight = weight.div(weight.sum(axis=1, min_count=1), axis=0)
        mean = (factor * weight).sum(axis=1)
        std = np.sqrt((weight * (factor.sub(mean, axis=0)**2)).sum(axis=1))
        std[std < 0.00001] = np.nan
        f_scale = f_scale + factor.sub(mean, axis=0).div(std, axis=0).fillna(0)
    return f_scale


def normalize_process_zscore(factor:pd.DataFrame) -> pd.DataFrame:
    '''
    标准化处理
    '''
    factor_std = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)
    return factor_std


def AZ_Row_zscore(df, cap=None, check_std=False):
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    if check_std:
        df_std = df_std[df_std >= 0.00001]
    target = df.sub(df_mean, axis=0).div(df_std, axis=0)
    if cap is not None:
        target[target > cap] = cap
        target[target < -cap] = -cap
    return target


def AZ_Normal_IC(signal, stk_rtn, stk_price, min_valids=None, lag=1, method='open'):
    if method == 'open':
        if lag == 1:
            signal = signal.shift(lag + 1)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn, axis=1, method='spearman').dropna()
        else:
            stk_rtn_new = stk_price / stk_price.shift(lag) - 1
            signal = signal.shift(lag + 1)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn_new, axis=1, method='spearman').dropna()
    else:
        if lag == 1:
            signal = signal.shift(lag)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn, axis=1, method='spearman').dropna()
        else:
            stk_rtn_new = stk_price / stk_price.shift(lag) - 1
            signal = signal.shift(lag)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn_new, axis=1, method='spearman').dropna()
    if min_valids is not None:
        signal_valid = signal.count(axis=1)
        signal_valid[signal_valid < min_valids] = np.nan
        signal_valid[signal_valid >= min_valids] = 1
        corr_signal = corr_df * signal_valid
    else:
        corr_signal = corr_df
    return corr_signal


def ic_stats(ic: pd.Series,
             period: int,
             path: bool = False) -> pd.Series:
    '''
    基于原始 ic 序列计算相关统计指标 - 考虑路径
    ic：原始 ic 序列（index 为每个交易日，value 为因子值与下一期收益计算的 rank_ic）
    period：为下一期收益的计算周期，如 5日收益、20日收益等
    '''
    path = period if path else 1
    ret = pd.Series(dtype='float')
    period_num = len(ic)//path
    ic_ = ic.values[-path*period_num:].reshape(-1, path)
    ret['IC'] = ic_.mean(axis=0).mean() # 先间隔时间求 ic 均值，然后再对路径求均值
    ret['IC_abs'] = np.abs(ic_).mean(axis=0).mean()
    ret['ICIR'] = (ic_.mean(axis=0) / ic_.std(axis=0)).mean() * np.sqrt(ANNUAL_PARAMS[period])  # 计算 ICIR
    ret['IC_pvalue'] = np.mean(st.ttest_1samp(ic_, 0).pvalue)  # 计算 ic 的显著性水平
    ret['IC_Skew'] = pd.DataFrame(ic_).skew(axis=0).mean()
    ret['IC_Kurt'] = pd.DataFrame(ic_).kurt(axis=0).mean()
    return ret


def ic_decay(factor_tmp: pd.DataFrame,
             stk_rtns: pd.DataFrame,
             stk_price: pd.DataFrame,
             period: int,
             path: bool = False,
             method: str ='open',
             max_lag: int = 10) -> tuple:
    '''
    ic 衰减分析
    period: 收益计算周期，即调仓周期
    max_lag：最大的收益滞后期
    '''
    path = period if path else 1
    decay = pd.Series(dtype='float')
    for p in range(max_lag):
        ic = AZ_Normal_IC(signal=factor_tmp.shift(p*period), stk_rtn=stk_rtns, stk_price=stk_price, lag=period, method=method)
        period_num = len(ic)//path
        ic_ = ic.values[-path*period_num:].reshape(-1, path)
        decay.loc[p] = ic_.mean(axis=0).mean()
    # 计算半衰期
    t = min(list(decay[(decay - decay.loc[0]/2)<0].index)+[len(decay)-1])
    return t, decay


def ic_stats_group(factor_tmp: pd.DataFrame,
                   stk_rtns: pd.DataFrame,
                   stk_price: pd.DataFrame,
                   group: pd.DataFrame,
                   period: int,
                   path: bool = False,
                   method: str ='open') -> pd.Series:
    '''
    计算组内 ic， 如计算组内 ic、计算行业内 ic、计算波动率分组下的ic，计算下一期涨跌幅分组下的 ic 等
    '''
    path = period if path else 1
    group_type = group.stack().unique()
    ic_g = pd.Series(dtype='float')
    for g in group_type:
        group_ = group[group==g]
        group_ = group_.where(pd.isnull(group_), 1).astype('float')
        factor = factor_tmp * group_
        ic = AZ_Normal_IC(signal=factor, stk_rtn=stk_rtns, stk_price=stk_price, lag=period, method=method)
        period_num = len(ic)//path
        ic_ = ic.values[-path*period_num:].reshape(-1, path)
        ic_g.loc[g] = ic_.mean(axis=0).mean()
    return ic_g


def ic_analysis(factor_tmp: pd.DataFrame,
                stk_rtns: pd.DataFrame,
                stk_price: pd.DataFrame,
                market_cap: pd.DataFrame,
                industry: pd.DataFrame,
                period: int,
                method: str = 'open',
                path: bool = False) -> dict:
    '''
    IC 分析，包括 IC 的统计指标、IC 衰减、IC 市值衰减、IC 行业衰减
    period：下一期收益率计算周期，即持仓期
    method：计算下一期收益所用的价格类型，'open' 按开盘价计算，'close'按收盘价计算
    path：路径数量，False 按每日计算ic统计指标，True 将 period 条路径分别计算ic统计指标再对各条路径求均值
    '''
    ret = {}
    # IC 常规统计指标
    ic = AZ_Normal_IC(signal=factor_tmp, stk_rtn=stk_rtns, stk_price=stk_price, lag=period, method=method)
    ic_stats_ = ic_stats(ic, period=period, path=path)
    ret['ic_series'] = ic
    print('计算 IC 常规统计指标 Done!')
    # IC 衰减
    decay_t, decay_ic = ic_decay(factor_tmp=factor_tmp, stk_rtns=stk_rtns, stk_price=stk_price, period=period, path=path, method=method, max_lag=10)
    ic_stats_.loc['IC_halflife'] = decay_t
    ret['ic_decay_time'] = decay_ic
    print('计算 IC 时间衰减 Done!')
    # IC 市值衰减
    cap_g = market_cap.rank(axis=1).apply(lambda x: pd.cut(x, bins=10, labels=range(10)), axis=1)
    ic_cap = ic_stats_group(factor_tmp=factor_tmp, group=cap_g, stk_rtns=stk_rtns, stk_price=stk_price, period=period, path=path, method=method).sort_index()
    ic_stats_.loc['IC_capdecay'] = round(ic_cap.reset_index().corr(method='spearman').iloc[0,1], 4)
    ret['ic_decay_cap'] = ic_cap
    print('计算 IC 市值衰减 Done!')
    # IC 行业衰减
    ic_indus = ic_stats_group(factor_tmp=factor_tmp, group=industry, stk_rtns=stk_rtns, stk_price=stk_price, period=period, path=path, method=method)
    ic_indus = ic_indus.loc[list(SHENWAN_2021.keys())].sort_values()
    ic_indus.index = ic_indus.index.map(SHENWAN_2021)
    ic_stats_.loc['IC_indusdecay'] = round((ic_indus<0).sum() / len(ic_indus), 4) # 负向 IC 数量占比
    ret['ic_decay_indus'] = ic_indus
    print('计算 IC 行业衰减 Done!')
    ret['ic_stats'] = ic_stats_
    return ret



def to_final_position(factor_score, forbid_day, method_func='standard'):
    '''
    将因子值转换为持仓
    '''
    if method_func == 'simple':
        pos_fin = factor_score.shift(1) * forbid_day
        return pos_fin
    else:
        pos_fin = factor_score.shift(1) * forbid_day
        pos_fin = pos_fin.replace(np.nan, 0)
        pos_fin = pos_fin * forbid_day
        pos_fin = pos_fin.ffill()
        return pos_fin


def to_final_dailypnl(pos_ratio: pd.DataFrame,
                      stk_rtns: pd.DataFrame,
                      period: int,
                      path: bool = False) -> tuple:
    '''
    基于仓位权重计算投资组合的每日收益
    '''
    path = period if path else 1
    pos_df_ = pos_ratio.shift(1)  # 将仓位前置 1 天与换仓的 twap 时间对齐
    dates_all = pos_df_.index  # 全部历史时间
    pos_dt = dates_all[2:] # 预留 2 个时间点用于调仓期与收益率时间的对齐
    period_num = len(pos_dt) // path + 1
    dt_index_ = np.pad(pos_dt, (0, period_num*period-len(pos_dt)), mode='constant', constant_values=np.nan)
    dt_index_ = dt_index_.reshape(-1, path)
    pos_path = []
    for i in range(path):
        dt = dt_index_[:,i]
        dt = dt[~np.isnan(dt)]
        pos = pos_df_.loc[dt].reindex(dates_all)
        if period>1:
            pos = pos.fillna(method='ffill', limit=period-1)
        pos_path.append(pos)
    pos_path = np.concatenate(pos_path, axis=0).reshape(path, -1, pos_df_.shape[-1])  # [path, t, 股票数]
    pnl_path = pos_path * stk_rtns.values
    pnl_path = np.nansum(pnl_path, axis=2) * np.where((~np.isnan(pnl_path)).sum(axis=2)==0, np.nan, 1)
    pnl_path = pd.DataFrame(pnl_path.T, index=dates_all, columns=range(path))
    return pnl_path, pos_path


def to_pnl_stats(ret: pd.Series)-> pd.Series:
    '''
    计算收益评价指标, ret 为以时间为 index 的 series
    '''
    stats = pd.Series(dtype='float64')
    # 年化收益率(%)
    stats['annual_return(%)'] = ((1+ret.mean())**250-1) * 100
    # 累计收益率(%)
    stats['cumprod_return(%)'] = ((ret+1).cumprod().iloc[-1]-1) * 100
    # 年化波动率(%)
    stats['return_volatility_annual(%)'] = ret.std() * np.sqrt(250) * 100
    # 夏普比率(%)
    stats['sharpe_ratio'] =  np.sqrt(250) * (ret.mean()-0)/np.std(ret,ddof=1)
    # 最大回撤(%)
    stats['max_drawdown(%)'] = ((ret+1).cumprod()/(ret+1).cumprod().cummax()-1).min() * 100
    # 盈亏比
    stats['profit_loss_ratio'] = ret[ret>0].mean() / abs(ret[ret<0]).mean()
    return stats


def to_pos_turnover(pos_final: np.array) -> float:
    '''
    基于持仓计算换手率
    '''
    pos_final_ratio = np.nan_to_num(pos_final,0) / np.abs(np.nan_to_num(pos_final,0)).sum(axis=2)[:,:,np.newaxis]
    pos_final_ratio_shift1 = np.roll(pos_final_ratio, 1, axis=1)
    pos_final_ratio_shift1[:,0,:] = np.nan
    turnover_path = np.abs(np.nan_to_num(pos_final_ratio,0) - np.nan_to_num(pos_final_ratio_shift1,0)).sum(axis=2)
    null = (~((pos_final_ratio==0) & (pos_final_ratio_shift1==0))).sum(axis=2)
    turnover_path = np.where(null==0, np.nan, turnover_path)
    turnover = 100 * np.nanmean(np.nanmean(turnover_path, axis=1), axis=0)
    return turnover



def factor_stats_ls(factor_tmp: pd.DataFrame,
                    forbid_days: pd.DataFrame,
                    stk_rtns: pd.DataFrame,
                    period: int,
                    path: bool = False) -> tuple:
    '''
    对因子进行多空测试
    '''
    factor_z = bt.AZ_Row_zscore(factor_tmp, cap=4.5)  # 横截面标准化处理，取值会限制在 [-4.5, 4.5] 区间上
    # 分成多空两大组的表现，
    factor_z_top = factor_z[factor_z > 0]
    factor_z_bot = factor_z[factor_z < 0]
    factor_z_top = factor_z_top.div(factor_z_top.sum(1), axis=0)
    factor_z_bot = factor_z_bot.div(factor_z_bot.abs().sum(1), axis=0)
    pos_top = to_final_position(factor_z_top, forbid_days)
    pos_bot = to_final_position(factor_z_bot, forbid_days)
    pnl_top_path, pos_top_path  = to_final_dailypnl(pos_ratio=pos_top, stk_rtns=stk_rtns, period=period, path=path)
    pnl_bot_path, pos_bot_path  = to_final_dailypnl(pos_ratio=pos_bot, stk_rtns=stk_rtns, period=period, path=path)
    dailypnl_top = pnl_top_path.mean(axis=1) # 多头收益
    dailypnl_bot = (-1) * pnl_bot_path.mean(axis=1) # 空头收益
    dailypnl_final = dailypnl_top.fillna(0) - dailypnl_bot.fillna(0)
    pnl_stats = to_pnl_stats(dailypnl_final)
    pos_final = pos_top_path + pos_bot_path
    pnl_stats.loc['turnover(%)'] = to_pos_turnover(pos_final)
    return dailypnl_top, dailypnl_bot, dailypnl_final, pnl_stats


def factor_stats_group(factor_tmp: pd.DataFrame,
                       forbid_days: pd.DataFrame,
                       stk_rtns: pd.DataFrame,
                       market_cap: pd.DataFrame,
                       period: int,
                       group_num: int = 10,
                       path: bool = False) -> tuple:
    '''
    对因子进行分组测试
    '''
    factor_tmp_rank = factor_tmp.rank(axis=1, method='dense', ascending=True, pct=True)
    factor_tmp_group = factor_tmp_rank.apply(lambda x: pd.cut(x, bins=group_num, labels=range(group_num)), axis=1)
    factor_tmp_group = factor_tmp_group.astype('float')
    # 统计组内股票数量，判断是否等分
    group_counts = factor_tmp_group.apply(lambda x: x.value_counts(), axis=1).describe().T
    # 统计分组收益
    pnl_g = {}
    pnl_stats_g = {}
    pos_g = {}
    cap_g = {}
    for g in range(group_num):
        fg = factor_tmp_group[factor_tmp_group==g]
        fg = fg.where(pd.isnull(fg), 1)
        pos = fg.div(fg.sum(min_count=1,axis=1), axis=0)
        fg_pos = to_final_position(pos, forbid_days)
        pnl_fg_path, pos_fg_path  = to_final_dailypnl(pos_ratio=fg_pos, stk_rtns=stk_rtns, period=period, path=path)
        dailypnl_fg = pnl_fg_path.mean(axis=1) # 分组收益
        pnl_g[g] = dailypnl_fg # 分组日度收益率
        pnl_stats_fg = to_pnl_stats(dailypnl_fg)  # 统计收益评价指标
        pnl_stats_fg.loc['turnover(%)'] = to_pos_turnover(pos_fg_path)
        pnl_stats_g[g] = pnl_stats_fg
        cap_path = pos_fg_path * market_cap.values
        cap_path = np.where(cap_path==0, np.nan, cap_path)
        cap_g[g] = np.nanmean(np.nanmedian(cap_path, axis=2), axis=0)  # 组内市值中位数
        pos_g[g] = pos_fg_path
    pnl_g_df = pd.DataFrame(pnl_g)
    pnl_stats_g_df = pd.DataFrame(pnl_stats_g)
    cap_g_df = pd.DataFrame(cap_g)
    return pnl_g_df, pos_g, cap_g_df, pnl_stats_g_df, group_counts


# 多空超额
def factor_stats_ls_excess(dailypnl_group: pd.DataFrame,
                           stk_rtns: pd.DataFrame) -> tuple:
    '''
    基于分组收益构建多空、多头超额组合
    '''
    g_num = len(dailypnl_group.columns)
    ret = pd.DataFrame({'long': dailypnl_group.loc[:,g_num-1],
                        'long_short': dailypnl_group.loc[:,g_num-1] - dailypnl_group.loc[:,0],
                        'long_excess_CSI300':dailypnl_group.loc[:,g_num-1] - stk_rtns['000300.SH'],
                        'long_excess_CSI500':dailypnl_group.loc[:,g_num-1] - stk_rtns['000905.SH'],
                        'long_excess_CSI1000':dailypnl_group.loc[:,g_num-1] - stk_rtns['000852.SH']})
    ret_cumsum = ret.fillna(0).cumsum()
    # 统计多空收益的收益评价指标
    ls_ret_stats = to_pnl_stats(ret['long_short'])
    ls_ret_year = ret['long_short'].groupby(ret['long_short'].index.to_period('y')).apply(to_pnl_stats).unstack()
    ls_ret_year.loc['all'] = ls_ret_stats
    # 统计多空超额收益的收益评价指标
    l_excess_dict = {}
    for bm in ['CSI300','CSI500','CSI1000']:
        l_excess = ret[f'long_excess_{bm}']
        l_excess_year = l_excess.groupby(l_excess.index.to_period('y')).apply(to_pnl_stats).unstack()
        l_excess_stats = to_pnl_stats(l_excess)
        l_excess_year.loc['all'] = l_excess_stats
        l_excess_dict[bm] = l_excess_year
    l_excess_df = pd.concat(l_excess_dict.values(), keys= l_excess_dict.keys())
    l_excess_df = l_excess_df.loc[:,['annual_return(%)','sharpe_ratio','max_drawdown(%)']]
    l_excess_df = l_excess_df.swaplevel().unstack().swaplevel(axis=1).sort_index(axis=1)
    return ret_cumsum, ls_ret_year, l_excess_df


def factor_cale(factor_tmp: pd.DataFrame,
                univ_all: pd.DataFrame,
                stk_rtns: pd.DataFrame,
                stk_price: pd.DataFrame,
                forbid_days: pd.DataFrame,
                market_cap:pd.DataFrame,
                industry:pd.DataFrame,
                extreme_process: bool,
                neutralize_process:bool,
                normalize_process:bool,
                period:int,
                is_path: bool = False,
                ic_method: str = 'open',
                group_num: int = 10,
                plot: bool = False,
                fig_name: str = None,
                fig_path: str = None) :
    # 将数据与股票池对齐
    univ_all = univ_all.reindex(stk_rtns.index)
    univ_all = univ_all.loc[univ_all.index <= factor_tmp.index[-1]]
    factor_tmp = factor_tmp.reindex_like(univ_all) * univ_all
    stk_rtns_ = stk_rtns.reindex_like(univ_all) * univ_all  # twap1000 计算的涨跌幅，剔除指数部分数据
    stk_price = stk_price.reindex_like(univ_all) * univ_all  # twap1000
    forbid_days = forbid_days.reindex_like(univ_all) * univ_all
    market_cap = market_cap.reindex_like(univ_all) * univ_all
    industry = industry.reindex_like(univ_all)
    # 因子预处理
    if extreme_process:
        factor_tmp = extreme_process_MAD(factor_tmp)  # 进行异常值处理
    if neutralize_process:
        factor_tmp = neutralize_process_zscore(factor_tmp=factor_tmp, group=industry, market_cap=market_cap) # 市值行业中心化处理
    if normalize_process:
        factor_tmp = normalize_process_zscore(factor_tmp) # 进行标准化处理
    print('因子预处理 Done!')
    # IC 分析
    ic = ic_analysis(factor_tmp = factor_tmp,
                    stk_rtns = stk_rtns_,
                    stk_price = stk_price,
                    market_cap = market_cap,
                    industry = industry,
                    period = period,
                    method = ic_method,
                    path = is_path)
    print(ic)
    print('因子IC分析 Done!')
    # 二分多空表现
    dailypnl_top, dailypnl_bot, dailypnl_final, pnl_stats = factor_stats_ls(factor_tmp=factor_tmp,
                                                                            forbid_days=forbid_days,
                                                                            stk_rtns=stk_rtns_,
                                                                            period=period,
                                                                            path=is_path)
    print(pnl_stats.T)
    print('二分多空表现 Done!')
    # 分组收益表现
    pnl_g_df, pos_g, cap_g, pnl_stats_g_df, group_counts = factor_stats_group(factor_tmp=factor_tmp,
                                                                              forbid_days=forbid_days,
                                                                               stk_rtns=stk_rtns_,
                                                                               market_cap=market_cap,
                                                                               period=period,
                                                                               group_num=group_num,
                                                                               path=is_path)
    print(pnl_stats_g_df.T)
    print(group_counts)
    print('分组收益表现 Done!')
    # 因子超额表现
    ret_cumsum, ls_ret_year, l_ret_excess_year = factor_stats_ls_excess(dailypnl_group=pnl_g_df,
                                                                        stk_rtns=stk_rtns)
    print('ls_ret_year 表现')
    print(ls_ret_year.T)
    print('l_ret_excess_year 表现')
    print(l_ret_excess_year.T)
    print('因子超额表现 Done!')
    if plot:
        fig = plt.figure(figsize=[20, 33])
        fig.suptitle(fig_name, fontsize=14, x=0.5, y=1)
        gs = GridSpec(29, 28)
        # 绘制 iC 累计曲线
        ic_series = ic['ic_series']
        ax1_x = fig.add_subplot(gs[0:3, :])
        ax1_y = ax1_x.twinx()
        ax1_x.bar(ic_series.index, ic_series, label='ic_series (left)')
        ic_series.cumsum().plot(ax=ax1_y, color=cmap[1], label='ic_cumsum (right)', rot=0, fontsize=12, grid=False)
        ax1_y.set_xbound(lower=ic_series.index.min(), upper=ic_series.index.max())
        ax1_x.set_title(str(round(ic['ic_stats'], 4).to_dict()), fontsize=12)
        h1, l1 = ax1_x.get_legend_handles_labels()
        h2, l2 = ax1_y.get_legend_handles_labels()
        plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)
        # 绘制 IC 衰减
        ax2 = fig.add_subplot(gs[3:5, :13])
        ic['ic_decay_time'].plot.bar(ax=ax2, rot=0)
        ax2.set_title('ic_decay_time', fontsize=12)
        # 绘制 IC 市值衰减
        ax3 = fig.add_subplot(gs[3:5, 14:])
        ic['ic_decay_cap'].plot.bar(ax=ax3, rot=0, color=cmap[1])
        ax3.set_title('ic_decay_cap', fontsize=12)
        # 绘制 IC 行业衰减
        ax4 = fig.add_subplot(gs[5:7, :])
        ic['ic_decay_indus'].sort_values(ascending=False).plot.bar(ax=ax4, rot=15, color=cmap[2])
        ax4.set_title('ic_decay_indus', fontsize=12)
        # 绘制多空组收益
        ax5 = fig.add_subplot(gs[7:10, :])
        dailypnl_top.fillna(0).cumsum().plot(ax=ax5, label='top')
        dailypnl_bot.fillna(0).cumsum().plot(ax=ax5, label='bottom')
        dailypnl_final.fillna(0).cumsum().plot(ax=ax5, label='net',grid=True)
        ax5.set_xbound(lower=dailypnl_final.index.min(), upper=dailypnl_final.index.max())
        ax5.set_title(str(round(pnl_stats, 2).to_dict()), fontsize=12)
        ax5.legend()
        # 绘制top组和bottom组的表现
        ax12 = fig.add_subplot(gs[10:13, :])
        tb_ret = pnl_g_df.loc[:, [0, group_num - 1]]
        tb_ret.loc[:, 'long-short'] = tb_ret.loc[:, group_num - 1] - tb_ret.loc[:, 0]
        tb_ret.fillna(0).cumsum().plot(ax=ax12, grid=True)
        ax12.set_xbound(lower=tb_ret.index.min(), upper=tb_ret.index.max())
        ax12.legend(['short', 'long', 'long-short'], loc='upper left')
        ax12.set_title('Long/Short Cumsum Return')
        # 绘制分组收益
        ax6 = fig.add_subplot(gs[13:16, :])
        pnl_g_df.fillna(0).cumsum().plot(ax=ax6)
        stk_rtns['000905.SH'].cumsum().plot(ax=ax6, label='CSI500', color='grey', linestyle='--')
        stk_rtns['000300.SH'].cumsum().plot(ax=ax6, label='CSI300', color='black', linestyle='--')
        stk_rtns['000852.SH'].cumsum().plot(ax=ax6, label='CSI1000', color='navy', linestyle='--', grid=True)
        ax6.set_xbound(lower=pnl_g_df.index.min(), upper=pnl_g_df.index.max())
        ax6.legend(loc='upper left', ncol=3)
        ax6.set_title('Group Cumsum Return')
        # 绘制各组内市值分布
        ax7 = fig.add_subplot(gs[16:19, :])
        cap_g.boxplot(ax=ax7)
        ax7.set_title('Market Cap Median Boxplot', fontsize=12)
        # 绘制分组收益评价指标
        ax8 = fig.add_subplot(gs[19:21, :])
        ax8.set_axis_off()  # 除去坐标轴
        table = ax8.table(cellText=round(pnl_stats_g_df, 4).values,
                          bbox=(0.2, 0, 0.8, 1),  # 设置表格位置， (x0, y0, width, height)
                          rowLoc='right',  # 行标题居中
                          cellLoc='right',
                          colLabels=[f'G{i}' for i in pnl_stats_g_df.columns],  # 设置列标题
                          rowLabels=pnl_stats_g_df.index,
                          colLoc='right',  # 列标题居中
                          edges='open'  # 不显示表格边框
                          )
        table.set_fontsize(12)
        ax8.set_title('Group Return Performance', fontsize=12)
        # 绘制多空、多头等超额收益
        ax9 = fig.add_subplot(gs[21:24, :])
        ret_cumsum.plot(ax=ax9, grid=True)
        ax9.set_xbound(lower=ret_cumsum.index.min(), upper=ret_cumsum.index.max())
        ax9.legend(loc='upper left', ncol=2)
        ax9.set_title('Excess Return Cumsum', fontsize=12)
        # 绘制多空超额统计指标
        ax10 = fig.add_subplot(gs[24:26, :])
        ax10.set_axis_off()  # 除去坐标轴
        table = ax10.table(cellText=round(ls_ret_year, 4).values,
                           bbox=(0.1, 0, 0.9, 1),  # 设置表格位置， (x0, y0, width, height)
                           rowLoc='right',  # 行标题居中
                           cellLoc='right',
                           colLabels=ls_ret_year.columns,  # 设置列标题
                           rowLabels=ls_ret_year.index,
                           colLoc='right',  # 列标题居中
                           edges='open'  # 不显示表格边框
                           )
        table.set_fontsize(12)
        ax10.set_title('Long-Short Return Performance', fontsize=12)
        # 绘制多头超额统计指标
        ax11 = fig.add_subplot(gs[26:, :])
        ax11.set_axis_off()  # 除去坐标轴
        cols = l_ret_excess_year.columns
        col_labels = ['\n'.join(cols[i]) for i in range(len(cols))]
        table = ax11.table(cellText=round(l_ret_excess_year, 4).values,
                           bbox=(0.1, 0, 0.9, 1),  # 设置表格位置， (x0, y0, width, height)
                           rowLoc='right',  # 行标题居中
                           cellLoc='right',
                           colLabels=col_labels,  # 设置列标题
                           rowLabels=l_ret_excess_year.index,
                           colLoc='right',  # 列标题居中
                           edges='open'  # 不显示表格边框
                           )
        table.set_fontsize(12)
        ax11.set_title('Long Excess Return Performance', fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_path, dip=100)
        plt.show()
        plt.close()


def main():
    # 提取股票日度收益率
    stk_rtns = io_obj.load_data(Path_obj.stk_rtns_path)
    stk_price_open = io_obj.load_data(Path_obj.stk_twap_1000_price)
    stk_price_open.index = pd.to_datetime(stk_price_open.index.strftime('%Y-%m-%d'))
    # 提取股票市值、所属申万一级行业
    market_cap = io_obj.load_data(Path_obj.mktcap_path)
    industry = io_obj.load_data(Path_obj.sw_indus_l1_path)
    # 剔除停牌等股票
    forb_suspend = io_obj.load_data(Path_obj.forb_suspend_path)
    forb_stpt = io_obj.load_data(Path_obj.forb_stpt_path)
    forb_limited = io_obj.load_data(Path_obj.forb_limited_path)
    forbid_days = forb_suspend * forb_stpt * forb_limited
    forbid_days = forbid_days.replace(0, np.nan)
    # 提取股票池
    univ = io_obj.load_data(Path_obj.univ_path / UNIV_FILE[stock_univ])
    # 截取时间
    stk_rtns = stk_rtns.loc[st_date:ed_date]
    new_index = stk_rtns.index
    forbid_days = forbid_days.reindex(new_index)
    univ = univ.reindex(new_index)
    market_cap = market_cap.reindex(new_index)
    industry = industry.reindex(new_index)

    fig_path = f"{temgG_path}/{name}_{dt.datetime.today().strftime('%Y%m%d%H%M%S')}_fig.png"
    input_factor = pd.read_pickle(file_path)
    input_factor.index = pd.to_datetime(input_factor.index)
    factor_cale(factor_tmp=input_factor,
                univ_all=univ,
                stk_rtns=stk_rtns,
                stk_price=stk_price_open,
                forbid_days=forbid_days,
                market_cap=market_cap,
                industry=industry,
                extreme_process=extreme_process,
                neutralize_process=neutralize_process,
                normalize_process=normalize_process,
                period=period,
                is_path=path,
                ic_method='open',
                group_num=group_num,
                plot=True,
                fig_name = name,
                fig_path=fig_path)



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest='ENV', default="BKT")
    parser.add_option('-f', dest='File', default='/mnt/clouddisk1/share/dat_jjl/relationship_factor/limit_relationship_mom_allratio/OVERNIGHTMOM_0.05sh_20d.pkl')
    parser.add_option('-p', dest='Picture', default="/mnt/clouddisk1/share/dat_jjl/relationship_factor/limit_relationship_mom_allratio")
    parser.add_option('-t', dest='Path', default=False)
    parser.add_option('-n', dest='Period', default=20, type='int')
    parser.add_option('-g', dest='Group', default=10)
    parser.add_option('--st', dest='st_date', default='2019-01-01')  # 2018-01-01
    parser.add_option('--ed', dest='ed_date', default='2024-06-24') # 2024-03-01
    parser.add_option('-u', dest='stock_univ', default='CSI300')
    parser.add_option('--ext', dest='extreme_process', default=True)
    parser.add_option('--neu', dest='neutralize_process', default=False)
    parser.add_option('--nor', dest='normalize_process', default=True)
    parser.add_option('-s', dest='Send_email', default=False)

    (options, args) = parser.parse_args()

    # 创建io对象
    data_type = 'pkl'
    io_obj = AZ_IO(mod=data_type)
    io_obj_csv = AZ_IO(mod='csv')
    env = options.ENV
    file_path = options.File
    temgG_path = options.Picture
    email_address = options.Send_email
    path = options.Path
    period = options.Period
    stock_univ = options.stock_univ
    st_date = options.st_date
    ed_date = options.ed_date
    extreme_process = options.extreme_process
    neutralize_process = options.neutralize_process
    normalize_process = options.normalize_process
    group_num = options.Group

    if env == 'BKT':
        Path_obj = Path(mod='bkt')

    elif env == 'PRO':
        Path_obj = Path(mod='pro')

    script_nm = f'EQT_Factor_Evaluation_{env}'
    name = 'OVERNIGHTMOM_0.05sh_20d' + f'_{stock_univ}'
    main()
