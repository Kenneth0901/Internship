import pandas as pd
import sys

from optparse import OptionParser
sys.path.append('/mnt/clouddisk1/share/open_lib')
import os
import numpy as np
# import json
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

        # industry_data
        # self.Huatai_Level2_consumer_essential = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_consumer_essential.pkl"
        # self.Huatai_Level2_consumer_optional = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_consumer_optional.pkl"
        # self.Huatai_Level2_cyclical_manufacturing = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_cyclical_manufacturing.pkl"
        # self.Huatai_Level2_cyclical_material = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_cyclical_material.pkl"
        # self.Huatai_Level2_cyclical_resource = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_cyclical_resource.pkl"
        # self.Huatai_Level2_financial_financial = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_financial_financial.pkl"
        # self.Huatai_Level2_growth_TMT = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_growth_TMT.pkl"
        # self.Huatai_Level2_others_Utilities = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_others_Utilities.pkl"
        # self.Huatai_Level2_others_others = Env_obj.BinFiles.EM_Funda / "DERIVED_12/Huatai_Level2_others_others.pkl"

        self.factor_path = Env_obj.BinFiles.factor_pool
        self.feature_factor_pnl_path = Env_obj.BinFiles.factor_pool / "f_stats" / "feature_factor_pnl_PRO.csv"
        self.ls_alpha_pnl_path = Env_obj.BinFiles.factor_pool / "f_stats" / "ls_alpha_pnl_PRO.csv"
        # cost structure
        self.cost_structure = "/mnt/mfs/dat_zyy/RETURN/daily_spreadratio_mean.pkl"
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
        #
        # self.leverage_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'leverage_factor.pkl'
        # self.btop_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'btop_factor.pkl'
        # self.liquid_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'liquid_factor.pkl'
        # self.momentum_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'momentum_factor.pkl'
        # self.size_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'size_factor.pkl'
        # self.residual_vol_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'resvol_factor.pkl'
        # self.beta_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'beta_factor.pkl'
        # self.earn_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'earnyield_factor.pkl'
        # self.nlsize_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'nlsize_factor.pkl'
        # self.growth_factor = Env_obj.BinFiles.EM_Funda / 'DERIVED_Barra_Standard' / 'growth_factor.pkl'
        self.portfolio = '/mnt/clouddisk1/share/dat_hjy/ZZ500_yushuiC3'

def single_day_regession(ii, stk_rtns, leverage_factor,
                             btop_factor, liquid_factor,
                             momentum_factor, size_factor,
                             residual_vol_factor, beta_factor,
                             earn_factor, nlsize_factor,
                             growth_factor, m_ls):
    result_df = pd.DataFrame(np.nan, index=[stk_rtns.index[ii]], columns=['leverage', 'btop', 'liquidity',
                                                                          'momentum', 'size', 'res_vol',
                                                                          'beta', 'earn', 'nlsize', 'growth'])
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
    # reg = LinearRegression().fit(x, y)
    #
    # for ii in range(0, 10):
    #     result_df.iloc[0, ii] = reg.coef_[ii]

    if x.shape[0] > 0:
        reg = LinearRegression().fit(x, y)

        for ii in range(0, 10):
            result_df.iloc[0, ii] = reg.coef_[ii]
    else:
        print(stk_rtns.index[ii])

    m_ls.append(result_df)



def multi_process_regression(stk_rtns, leverage_factor,
                             btop_factor, liquid_factor,
                             momentum_factor, size_factor,
                             residual_vol_factor, beta_factor,
                             earn_factor, nlsize_factor,
                             growth_factor):
    result_ls = []
    for ii in range(len(stk_rtns)-1):
        print(ii)
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
    # print(portfolio)
    # portfolio = portfolio.drop('000300.SH', axis=1).drop('000852.SH', axis=1)
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
    # stk_rtns = stk_rtns.loc[stk_rtns.index <= '2023-02-10']

    stk_rtns = stk_rtns.shift(-2)

    # univ_select = T3k_a.loc[T3k_a.index>='2020-01-01']
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

    # reg_df.to_pickle('/mnt/mfs/dat_jerry/Alpha_Optimize/barra_rtn_T3k_standard.pkl')

    reg_df.to_pickle('/mnt/clouddisk1/share/dat_hjy/barra_portfolio/barra_rtn_standard.pkl')

    # 计算portfolio在Barra上的暴露
    leverage_factor = bt.AZ_Row_zscore(leverage_factor, cap=3.5)
    btop_factor = bt.AZ_Row_zscore(btop_factor, cap=3.5)
    liquid_factor = bt.AZ_Row_zscore(liquid_factor, cap=3.5)
    momentum_factor = bt.AZ_Row_zscore(momentum_factor, cap=3.5)
    size_factor = bt.AZ_Row_zscore(size_factor, cap=3.5)
    residual_vol_factor = bt.AZ_Row_zscore(residual_vol_factor, cap=3.5)
    beta_factor = bt.AZ_Row_zscore(beta_factor, cap=3.5)
    earn_factor = bt.AZ_Row_zscore(earn_factor, cap=3.5)
    nlsize_factor = bt.AZ_Row_zscore(nlsize_factor, cap=3.5)
    growth_factor = bt.AZ_Row_zscore(growth_factor, cap=3.5)

    leverage_500_expo = (leverage_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    btop_500_expo = (btop_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    liquid_500_expo = (liquid_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    momentum_500_expo = (momentum_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    size_500_expo = (size_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    residual_vol_500_expo = (residual_vol_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    beta_500_expo = (beta_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    earn_500_expo = (earn_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    nlsize_500_expo = (nlsize_factor.reindex_like(zz500_w) * zz500_w).sum(1)
    growth_500_expo = (growth_factor.reindex_like(zz500_w) * zz500_w).sum(1)

    portfolio_select = portfolio  # .shift(1)

    portfolio_leverage_expo = (portfolio_select * leverage_factor).sum(1) - leverage_500_expo
    portfolio_btop_expo = (portfolio_select * btop_factor).sum(1) - btop_500_expo
    portfolio_liquid_expo = (portfolio_select * liquid_factor).sum(1) - liquid_500_expo
    portfolio_momentum_expo = (portfolio_select * momentum_factor).sum(1) - momentum_500_expo
    portfolio_size_expo = (portfolio_select * size_factor).sum(1) - size_500_expo
    portfolio_residual_vol_expo = (portfolio_select * residual_vol_factor).sum(1) - residual_vol_500_expo
    portfolio_beta_expo = (portfolio_select * beta_factor).sum(1) - beta_500_expo
    portfolio_earn_expo = (portfolio_select * earn_factor).sum(1) - earn_500_expo
    portfolio_nlsize_expo = (portfolio_select * nlsize_factor).sum(1) - nlsize_500_expo
    portfolio_growth_expo = (portfolio_select * growth_factor).sum(1) - growth_500_expo

    expo_df = pd.concat([portfolio_leverage_expo,
                         portfolio_btop_expo,
                         portfolio_liquid_expo,
                         portfolio_momentum_expo,
                         portfolio_size_expo,
                         portfolio_residual_vol_expo,
                         portfolio_beta_expo,
                         portfolio_earn_expo,
                         portfolio_nlsize_expo,
                         portfolio_growth_expo], axis=1)
    # expo_df = expo_df.loc[expo_df.index >= '2023-01-01']
    expo_df = expo_df.loc[expo_df.index >= '2018-01-01']

    expo_df.columns = ['leverage', 'btop', 'liquidity', 'momentum', 'size', 'res_vol', 'beta', 'earn', 'nlsize',
                       'growth']

    expo_df.to_pickle('/mnt/clouddisk1/share/dat_hjy/barra_portfolio/barra_expo_df_zz500.pkl')

    # portfolio相比于zz500在barra因子上的收益 = 收益关于Barra的回归 * （portfolio在Barra上的暴露 - zz500在Barra上的暴露）
    barra_return_df = reg_df * expo_df
    barra_return_df = barra_return_df.shift(2)
    barra_return_df.to_pickle('/mnt/clouddisk1/share/dat_hjy/barra_portfolio/barra_return_df.pkl')

    # 计算portfolio在行业上的暴露
    shenwan_2021 = {'11': "农林牧渔",
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
                    '77': "美容护理",
                    }

    di_indus_uni = pd.read_pickle(Path_obj.sector_path)
    indus_list = list(di_indus_uni.unstack().dropna().drop_duplicates().values)
    indus_list.sort()

    result_ls = []
    # result_ls_2 = []

    # 计算选股相对于zz500在行业上的暴露造成的收益：将portfolio在行业上的暴露行和调整到和zz500在行业上的暴露行和一致，相减之后乘以return table再进行行求和即得每个行业相对于zz500的选股收益
    for indus_data in indus_list:
        indus_uni = di_indus_uni[di_indus_uni == indus_data]
        indus_uni = indus_uni.where(pd.isnull(indus_uni), 1)
        zz500_expo = zz500_w * indus_uni
        portfolio_expo = (portfolio_select * indus_uni).div((portfolio_select * indus_uni).sum(axis=1).replace(0, np.nan), axis=0)
        portfolio_expo = portfolio_expo.mul((zz500_w * indus_uni).sum(axis=1).replace(0, np.nan), axis=0)
        # record = (zz500_w * indus_uni).sum(axis=1).replace(0, np.nan)
        indus_return = ((portfolio_expo - zz500_expo) * stk_rtns).sum(1)
        indus_return = pd.DataFrame(indus_return, columns=[indus_data])
        result_ls.append(indus_return)

    indus_return_df = pd.concat(result_ls, axis=1)
    indus_return_df = indus_return_df.shift(2)
    indus_return_df.to_pickle('/mnt/clouddisk1/share/dat_hjy/barra_portfolio/indus_return_df.pkl')


        # zz500_expo = (zz500_w * indus_uni).sum(1)
        # portfolio_expo = (portfolio_select * indus_uni).sum(1)
        # expo_indus = (portfolio_expo - zz500_expo) / zz500_expo
        # expo_indus = pd.DataFrame(expo_indus, columns=[indus_data])
        # expo_indus = expo_indus.loc[expo_indus.index >= '2018-01-01']
        # result_ls.append(expo_indus)
        #
        # expo_indus_2 = (portfolio_expo - zz500_expo)
        # expo_indus_2 = pd.DataFrame(expo_indus_2, columns=[indus_data])
        # expo_indus_2 = expo_indus_2.loc[expo_indus_2.index >= '2018-01-01']
        # result_ls_2.append(expo_indus_2)

    # indus_expo_df = pd.concat(result_ls, axis=1)
    # indus_expo_df_2 = pd.concat(result_ls_2, axis=1)
    # indus_expo_df_2.to_pickle('/mnt/clouddisk1/share/dat_hjy/barra_portfolio/indus_expo_df.pkl')


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