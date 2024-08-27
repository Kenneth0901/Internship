__author__ = 'jerry'

import pandas as pd
import datetime as dt
import numpy as np
import multiprocessing
from optparse import OptionParser

import warnings

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


def to_final_position(factor_score, forbid_day, method_func='standard'):
    if method_func == 'simple':
        pos_fin = factor_score.shift(1) * forbid_day
        return pos_fin
    else:
        pos_fin = factor_score.shift(1) * forbid_day
        pos_fin = pos_fin.replace(np.nan, 0) ###处理第一行变成nan
        pos_fin = pos_fin * forbid_day
        pos_fin = pos_fin.ffill()
        return pos_fin



def factor_stats(factor_df, univ_data, forbid_days, rtn_df, method_func):
    '''
    :param factor_df:   factor df
    :param univ_data:   T1800 / HS300 / ZZ500 / T1000
    :param index_data_r:  series of index_rtn
    :param forbid_days:  df of tradevalid
    :param rtn_df:    df of stock rtn
    :param method_func:   feature/factor/ls_alpha/
    :return:  pos df / pnl series , depend on method func
    '''
    factor_sel = factor_df.reindex(univ_data.columns, axis="columns") * univ_data #限定股票池
    forbid_days = forbid_days.reindex_like(factor_sel)
    return_df = rtn_df.reindex_like(factor_sel)

    if method_func == 'feature' or method_func == 'factor':
        factor_z = bt.AZ_Row_zscore(factor_sel, cap=4.5) #标准化因子，cap=4.5限制个股权重大小

        factor_z_top = factor_z[factor_z > 0]
        factor_z_bot = factor_z[factor_z < 0]

        factor_z_top = factor_z_top.div(factor_z_top.sum(1), axis=0) # 因子大于零的算每天各股权重
        factor_z_bot = factor_z_bot.div(factor_z_bot.abs().sum(1), axis=0)
        # factor_z = factor_z_top.fillna(0) - factor_z_bot.fillna(0)
        # factor_z = factor_z.replace(0, np.nan)
        #
        # pos_final = to_final_position(factor_z, forbid_days)
        # pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)

        pos_top = to_final_position(factor_z_top, forbid_days)
        pnl_top = (pos_top.shift(1) * return_df).sum(axis=1) #axis=0，代表对不同行操作，axis=1，代表对不同列操作
        pos_bot = to_final_position(factor_z_bot, forbid_days)
        pnl_bot = (pos_bot.shift(1) * return_df).sum(axis=1)
        pnl_final = pnl_bot.fillna(0) + pnl_top.fillna(0)

        if bt.AZ_Sharpe_y(pnl_final) < 0:
            long_short_ratio = factor_z_bot.count(1).mean() / factor_z_top.count(1).mean() ###没懂有啥用，等下回来看
        else:
            long_short_ratio = factor_z_top.count(1).mean() / factor_z_bot.count(1).mean()

        factor_z = factor_z_top.fillna(0) + factor_z_bot.fillna(0)
        factor_z = factor_z.replace(0, np.nan)
        pos_final = to_final_position(factor_z, forbid_days)

        pos_turnover = pos_final.fillna(0).diff().abs().sum(axis=1) / pos_final.abs().sum(axis=1).shift(1)
        last_10days_turnover = pos_turnover.rolling(10).mean().iloc[-1]
        last_120days_turnover = pos_turnover.rolling(120).mean().iloc[-1]

        return last_10days_turnover, last_120days_turnover, pos_final, pnl_final, pnl_top, pnl_bot, long_short_ratio
    elif method_func == 'ls_alpha':
        pos_final = to_final_position(factor_sel, forbid_days)
        pnl_final = (pos_final.shift(1) * return_df).sum(axis=1)
        return pos_final, pnl_final


def AZ_Factor_Evaluation(factor_df, univ_nm, forbid_days, rtn_df, factor_type):
    last_10days_turnover, last_120days_turnover, pos_df, dailypnl_gross, dailypnl_top, dailypnl_bot, long_short_ratio = factor_stats(factor_df, univ_nm, forbid_days,
                                                                                       rtn_df, factor_type)
    return dailypnl_gross, dailypnl_top, dailypnl_bot, last_10days_turnover, last_120days_turnover, long_short_ratio


def AZ_Normal_IC(signal, stk_rtn, stk_price, min_valids=None, lag=1, method='close'):
    if method == 'open':
        if lag == 1:
            signal = signal.shift(lag + 1)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn, axis=1).dropna()
        else:
            stk_rtn_new = stk_price / stk_price.shift(lag) - 1
            signal = signal.shift(lag + 1)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn_new, axis=1).dropna()
    else:
        if lag == 1:
            signal = signal.shift(lag)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn, axis=1).dropna()
        else:
            stk_rtn_new = stk_price / stk_price.shift(lag) - 1
            signal = signal.shift(lag)
            signal = signal.replace(0, np.nan)
            corr_df = signal.corrwith(stk_rtn_new, axis=1).dropna()
    if min_valids is not None:
        signal_valid = signal.count(axis=1)
        signal_valid[signal_valid < min_valids] = np.nan
        signal_valid[signal_valid >= min_valids] = 1
        corr_signal = corr_df * signal_valid
    else:
        corr_signal = corr_df
    return corr_signal


def AZ_Normal_IR(signal, pct_n, stk_price, min_valids=None, lag=1, method='close'):
    corr_signal = AZ_Normal_IC(signal, pct_n, stk_price, min_valids, lag, method)
    ic_mean = corr_signal.mean()
    ic_std = corr_signal.std()
    ir = ic_mean / ic_std * np.sqrt(250)
    return ir


def calc_func(factor_tmp, univ_all, univ_300, univ_500, univ_1000, stk_rtns, stk_rtns_close, stk_price, stk_price_open,
              stk_price_open_open,
              stk_rtns_night, stk_rtns_intraday, forbid_days, fig_path, factor_type='factor'):
    # format
    univ_all = univ_all.reindex(stk_rtns.index)
    univ_all = univ_all.loc[univ_all.index <= factor_tmp.index[-1]]
    factor_tmp = factor_tmp.reindex_like(univ_all) * univ_all
    stk_rtns = stk_rtns.reindex_like(univ_all) * univ_all
    stk_rtns_close = stk_rtns_close.reindex_like(univ_all) * univ_all
    stk_price = stk_price.reindex_like(univ_all) * univ_all
    stk_price_open = stk_price_open.reindex_like(univ_all) * univ_all
    forbid_days = forbid_days.reindex_like(univ_all) * univ_all

    factor_tmp_original = factor_tmp.copy()
    factor_tmp = bt.AZ_Row_zscore(factor_tmp, cap=4.5)

    # last 300 count
    factor_tmp_1 = factor_tmp.replace(0, np.nan)
    # median_count = factor_tmp_1.count(1).iloc[-300:].mean()

    median_count = (factor_tmp_1.count(1) / univ_all.count(1)).mean()

    stk_rtns_open_to_t1_close = stk_price / stk_price_open_open.shift(1) - 1

    # IC1 IC3 IC5 close
    ic_closet0_opent1 = AZ_Normal_IC(factor_tmp, stk_rtns_night, stk_price, lag=1).rolling(60).mean().mean()
    ic_opent1_closet1 = AZ_Normal_IC(factor_tmp, stk_rtns_open_to_t1_close, stk_price, lag=1, method='open').rolling(
        60).mean().mean()
    ic_opent1_opent2 = AZ_Normal_IC(factor_tmp, stk_rtns, stk_price_open, lag=1, method='open').rolling(
        60).mean().mean()
    ic_opent1_opent3 = AZ_Normal_IC(factor_tmp, stk_rtns, stk_price_open, lag=2, method='open').rolling(
        60).mean().mean()
    ic_opent1_opent4 = AZ_Normal_IC(factor_tmp, stk_rtns, stk_price_open, lag=3, method='open').rolling(
        60).mean().mean()
    ic_opent1_opent6 = AZ_Normal_IC(factor_tmp, stk_rtns, stk_price_open, lag=5, method='open').rolling(
        60).mean().mean()

    # IR1 IR3 IR5
    icir_closet0_opent1 = AZ_Normal_IR(factor_tmp, stk_rtns_night, stk_price, lag=1).__abs__()
    icir_opent1_closet1 = AZ_Normal_IR(factor_tmp, stk_rtns_open_to_t1_close, stk_price, lag=1, method='open').__abs__()

    icir_opent1_opent2 = AZ_Normal_IR(factor_tmp, stk_rtns, stk_price_open, lag=1, method='open').__abs__()
    icir_opent1_opent3 = AZ_Normal_IR(factor_tmp, stk_rtns, stk_price_open, lag=2, method='open').__abs__()
    icir_opent1_opent4 = AZ_Normal_IR(factor_tmp, stk_rtns, stk_price_open, lag=3, method='open').__abs__()
    icir_opent1_opent6 = AZ_Normal_IR(factor_tmp, stk_rtns, stk_price_open, lag=5, method='open').__abs__()

    # row_sum skew
    measure_1 = factor_tmp.abs().sum(axis=1)
    percentile15, percentile85 = measure_1.quantile(q=[0.15, 0.85])
    try:
        row_sum_skew = percentile15 / percentile85
        row_sum_skew = row_sum_skew.__round__(3)
    except Exception as e:
        row_sum_skew = 0

    # row_pos_skew
    percentile_50 = measure_1.quantile(q=0.5)
    closest_measure_index = measure_1.iloc[(measure_1 - percentile_50).abs().argsort()[:1]].index
    closest_measure = factor_tmp.loc[closest_measure_index].abs()
    closest_measure_q = closest_measure.quantile(q=[0.15, 0.85], axis=1)
    try:
        row_pos_skew = closest_measure_q.iloc[0, 0] / closest_measure_q.iloc[1, 0]
        row_pos_skew = row_pos_skew.__round__(3)
    except Exception as e:
        row_pos_skew = 0

    # t_cost = 0.0015
    # pos_df, dailypnl_gross = factor_stats(factor_tmp, univ_all, forbid_days, stk_rtns, 'ls_alpha')
    # daily_cost = (pos_df.diff().abs() * t_cost).sum(axis=1)
    # dailypnl_net = dailypnl_gross - daily_cost
    # ls_sharpe = bt.AZ_Sharpe_y(dailypnl_net)
    # if ls_sharpe < 0:
    #     dailypnl_net = dailypnl_net * (-1)
    #     ls_sharpe = ls_sharpe * (-1)
    # plt.figure(figsize=[16, 12])
    # p1 = plt.subplot(111)
    # p1.plot(dailypnl_net.fillna(0).cumsum(), label=f"PNL_Net, SP:{ls_sharpe}")
    # p1.set_title('long short')
    # p1.grid(linestyle='--')
    # p1.legend(loc='upper left')
    # plt.grid(True)
    # fig_path = f"{temgG_path}/temp_{dt.datetime.today().strftime('%Y%m%d%H%M%S')}_fig.png"
    # plt.savefig(fig_path)
    # plt.close()

    dailypnl_net, dailypnl_top, dailypnl_bot, last_10days_turnover, last_120days_turnover,long_short_ratio = AZ_Factor_Evaluation(factor_tmp, univ_all, forbid_days,
                                                                                     stk_rtns, 'factor')
    ls_sharpe = bt.AZ_Sharpe_y(dailypnl_net)
    reverse_flag = 0
    if ls_sharpe < 0:
        dailypnl_net = dailypnl_net * (-1)
        ls_sharpe = ls_sharpe * (-1)
        # long_short_ratio = 1 / long_short_ratio
        reverse_flag = 1
    dailypnl_bot = dailypnl_bot * (-1)
    plt.figure(figsize=[32, 16])
    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    p1.plot(dailypnl_net.fillna(0).cumsum(), label=f"PNL_Net, SP:{ls_sharpe}")
    p1.plot(dailypnl_top.fillna(0).cumsum(), label=f"PNL_top")
    p1.plot(dailypnl_bot.fillna(0).cumsum(), label=f"PNL_bot")

    p1.set_title(f"factor long short, ic_closet0_opent1: {ic_closet0_opent1}, ic_opent1_closet2: {ic_opent1_closet1}, "
                 f"ic_twapt1_twapt2: {ic_opent1_opent2}, ic_twapt1_twapt3: {ic_opent1_opent3}, "
                 f"ic_twapt1_twapt4: {ic_opent1_opent4}, ic_twapt1_twapt6: {ic_opent1_opent6} ")
    p1.grid(linestyle='--')
    p1.legend(loc='upper left')

    factor_tmp_rank = factor_tmp_original.rank(axis=1, method='dense', ascending=True, pct=True)
    factor_tmp_rank_group1 = factor_tmp_original[factor_tmp_rank <= 0.1]
    factor_tmp_rank_group10 = factor_tmp_original[factor_tmp_rank >= 0.9]

    factor_tmp_rank_group1 = factor_tmp_rank_group1.where(pd.isnull(factor_tmp_rank_group1), 1)
    factor_tmp_rank_group10 = factor_tmp_rank_group10.where(pd.isnull(factor_tmp_rank_group10), 1)

    factor_tmp_rank_group1 = factor_tmp_rank_group1.div(factor_tmp_rank_group1.sum(1), axis=0)
    factor_tmp_rank_group10 = factor_tmp_rank_group10.div(factor_tmp_rank_group10.sum(1), axis=0)

    pos_final_group1 = to_final_position(factor_tmp_rank_group1, forbid_days)
    pnl_final_group1 = (pos_final_group1.shift(1) * stk_rtns).sum(axis=1)

    pos_final_group10 = to_final_position(factor_tmp_rank_group10, forbid_days)
    pnl_final_group10 = (pos_final_group10.shift(1) * stk_rtns).sum(axis=1)

    if reverse_flag == 1:
        dailypnl_net = pnl_final_group1 - pnl_final_group10
    else:
        dailypnl_net = pnl_final_group10 - pnl_final_group1
    ls_sharpe_1 = bt.AZ_Sharpe_y(dailypnl_net)

    p2.plot(dailypnl_net.fillna(0).cumsum(), label=f"PNL_Net, SP:{ls_sharpe_1}")
    p2.plot(pnl_final_group1.fillna(0).cumsum(), label=f"PNL_group1")
    p2.plot(pnl_final_group10.fillna(0).cumsum(), label=f"PNL_group10")
    p2.set_title('LS group1 vs group 10')
    p2.grid(linestyle='--')
    p2.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close()

    factor_pos_check = factor_tmp.replace(0, np.nan)
    factor_pos_check = factor_pos_check.where(pd.isnull(factor_pos_check), 1)

    # last_1000_abs_pos = factor_pos_check.iloc[-1000:].abs()
    # last_1000_count_all = last_1000_abs_pos.sum().sum()
    # last_1000_abs_pos_hs300 = (last_1000_abs_pos * univ_300).sum().sum()
    # last_1000_abs_pos_zz500 = (last_1000_abs_pos * univ_500).sum().sum()
    # last_1000_abs_pos_t1000 = (last_1000_abs_pos * univ_1000).sum().sum()

    # pos_hs300_percent = (last_1000_abs_pos_hs300 / last_1000_count_all).__round__(2)
    # pos_zz500_percent = (last_1000_abs_pos_zz500 / last_1000_count_all).__round__(2)
    # pos_t1000_percent = (last_1000_abs_pos_t1000 / last_1000_count_all).__round__(2)
    #
    # other_pct = 1 - (pos_hs300_percent + pos_zz500_percent + pos_t1000_percent)

    one_factor_ls = [median_count, ls_sharpe, long_short_ratio,
                     ic_closet0_opent1, ic_opent1_closet1, ic_opent1_opent2, ic_opent1_opent3,
                     ic_opent1_opent4, ic_opent1_opent6,
                     icir_closet0_opent1, icir_opent1_closet1, icir_opent1_opent2, icir_opent1_opent3,
                     icir_opent1_opent4, icir_opent1_opent6,
                     last_10days_turnover, last_120days_turnover,
                     row_sum_skew, row_pos_skew,
                     # pos_hs300_percent, pos_zz500_percent, pos_t1000_percent, other_pct,
                     ]
    return one_factor_ls


def main():
    pd.set_option('max_colwidth', 200)

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

    new_index = stk_rtns.index[-1250:]
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

    fig_path = f"{temgG_path}/temp_{dt.datetime.today().strftime('%Y%m%d%H%M%S')}_fig.png"

    input_factor = pd.read_pickle(file_path)
    result_ls = calc_func(input_factor, univ_all, univ_300, univ_500, univ_1000, stk_rtns, stk_rtns_close, stk_price,
                          stk_price_open, stk_price_open_open,
                          stk_rtn_night, stk_rtn_intraday, forbid_days, fig_path, factor_type='factor')

    stats_table = pd.DataFrame([result_ls])
    stats_table = stats_table.round(3)
    stats_table.columns = ['Coverage',
                           'LS_SP', 'LS_Ratio',
                           "ic_closet0_opent1", "ic_opent1_closet2", "ic_twapt1_twapt2", "ic_twapt1_twapt3",
                           "ic_twapt1_twapt4", "ic_twapt1_twapt6",
                           "icir_closet0_opent1", "icir_opent1_closet2", "icir_twapt1_twapt2", "icir_twapt1_twapt3",
                           "icir_twapt1_twapt4", "icir_twapt1_twapt6",
                           'last_10_TR_mean', 'last_120_TR_mean',
                           'PosCol_Skew',
                           'PosRow_Skew',
                           # 'Pos_HS300_Pct',
                           # 'Pos_ZZ500_Pct',
                           # 'Pos_T1000_Pct',
                           # 'Other_Pct',
                           ]
    stats_table.index = [file_path.split('/')[-1]]

    # stats_table = stats_table.T
    print(stats_table.T)
    if email_address:
        send_email(stats_table.T.to_html(justify='center'),
                   [email_address], [fig_path], f"Factor Evaluation {file_path}")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest='ENV', default="BKT")
    parser.add_option('-f', dest='File', default=None)
    parser.add_option('-p', dest='Picture', default="/mnt/clouddisk1/share/work_zhanxy/result/figure")
    parser.add_option('-s', dest='Send_email', default='zhanxuanyou@gmail.com')

    (options, args) = parser.parse_args()

    # 创建io对象
    data_type = 'pkl'
    io_obj = AZ_IO(mod=data_type)
    io_obj_csv = AZ_IO(mod='csv')
    env = options.ENV
    file_path = options.File
    temgG_path = options.Picture
    email_address = options.Send_email
    if env == 'BKT':
        Path_obj = Path(mod='bkt')

    elif env == 'PRO':
        Path_obj = Path(mod='pro')

    script_nm = f'EQT_Factor_Evaluation_{env}'
    main()
