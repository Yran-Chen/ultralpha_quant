import pandas as pd
import numpy as np
from datetime import datetime
# from dataapi import DataApiselector
# import datetime_process
# from Dataview import DataView
# import ts
# import psycopg2
# import Database

name_map = {'A': "黄大豆",
             'AG': "白银" ,
             'AL': "铝",
             'AP': "苹果",
             'AU': "黄金",
             'B': "大豆二号",
             'BB': "胶合板",
             'BU': "石油沥青",
             'C': "玉米",
             'CF': "棉花",
             'CJ': "",
             'CS': "玉米淀粉",
             'CU': "铜",
             'CY': "棉纱",
             'D': "",
             'EB': "",
             'EG': "乙二醇",
             'ER': "籼稻",
             'FB': "中密度纤维板",
             'FG': "玻璃",
             'FU': "燃油",
             'HC': "热轧卷板",
             'I': "铁矿石",
             'IC': "中证500",
             'IF': "沪深300",
             'IH': "上证50",
             'J': "冶金焦炭",
             'JD': "鲜鸡蛋",
             'JM': "焦炭",
             'JR': "粳稻",
             'L': "塑料",
             'LR': "晚籼稻",
             'M': "豆粕",
             'MA': "新甲醇",
             'ME': "",
             'NI': "镍",
             'OI': "新菜籽油",
             'P': "",
             'PB': "铅",
             'PM': "普通小麦",
             'PP': "聚丙烯",
             'RB': "螺纹钢",
             'RI': "新早粙稻",
             'RM': "菜籽粕",
             'RO': "菜籽油",
             'RR': "粳米",
             'RS': "油菜籽",
             'RU': "天然橡胶",
             'SC': "原油",
             'SF': "硅铁",
             'SM': "锰硅",
             'SN': "锡",
             'SP': "",
             'SR':"白糖",
             'SS': "不锈钢",
             'T': "10年期国债",
             'TA': "PTA",
             'TC': "郑商所动力煤",
             'TF': "5年期国债",
             'UR': "尿素",
             'V': "聚氧乙烯",
             'WH': "新强麦",
             'WR': "线材",
             'WS':"强麦",
             'WT':"硬麦",
             'Y':"豆油",
             'ZC':"动力煤",
             'ZN':"锌"}

type_future_dic = {"energy":["FU", "MA"],
                   "grains": ["A", "B", "C", "SR", "WH", "PM"],
                   "oilseeds": ["M", "P", "Y", "RU", "OI", "RM", "RS"],
                   "industrial": ["L", "V", "J", "JM", "CF", "TA", "FG"],
                   "metal":["AL", "AU", "CU", "PB", "RB", "WR", "ZN", "AG"]}

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def get_fut_data(future_name='al', csv_name='fut_day',
                 start_date='2013-01-01', end_date='2019-06-01'):
    def find_date(df, start_date, end_date):
        pointer = 'date>="{}" and date<="{}"'.format(
            start_date, end_date)
        return df.query(pointer)

    fut = find_date(pd.DataFrame.from_csv('..\\save\\{}.csv'.format(csv_name)), start_date, end_date)
    name_index = fut['asset'].str.startswith(future_name)
    fut = fut.loc[name_index]
    fut = fut.reset_index(['date']).set_index(['date', 'asset']).sort_index(axis=0)
    # print(fut)
    return fut


def get_multi_fut_data(future_name1='al', future_name2="cu", csv_name='fut_main_day',
                       start_date='2013-01-01', end_date='2019-06-01'):
    def find_date(df, start_date, end_date):
        pointer = 'date>="{}" and date<="{}"'.format(
            start_date, end_date)
        return df.query(pointer)

    fut = find_date(pd.DataFrame.from_csv('..\\save\\{}.csv'.format(csv_name)), start_date, end_date)
    name_index1 = fut['asset'].str.startswith(future_name1)
    name_index2 = fut['asset'].str.startswith(future_name2)
    fut1 = fut.loc[name_index1][["code", "close"]]
    fut2 = fut.loc[name_index2][["code", "close"]]
    # fut = fut.reset_index(['date']).set_index(['date','asset']).sort_index(axis=0)
    # print(fut)
    df = fut1.join(fut2, lsuffix="_" + future_name1, rsuffix="_" + future_name2)
    s1 = "close_" + future_name1
    s2 = "close_" + future_name2
    return df[[s1, s2]]


def get_fut_main_data(future_name='al', csv_name='fut_main_day',
                      start_date='2013-01-01', end_date='2019-06-01'):
    def find_date(df, start_date, end_date):
        pointer = 'date>="{}" and date<="{}"'.format(
            start_date, end_date)
        return df.query(pointer)

    fut = find_date(pd.DataFrame.from_csv('..\save\\{}.csv'.format(csv_name)), start_date, end_date)
    name_index = fut['asset'].str.startswith(future_name)
    fut = fut.loc[name_index]
    # fut = fut.reset_index(['date']).set_index(['date', 'code']).sort_index(axis=0)
    return fut


def get_all_fut_main_data(csv_name='fut_main_day',
                      start_date='2012-01-01', end_date='2019-06-01'):
    def find_date(df, start_date, end_date):
        pointer = 'date>="{}" and date<="{}"'.format(
            start_date, end_date)
        return df.query(pointer)
    fut = find_date(pd.DataFrame.from_csv('..\save\\{}.csv'.format(csv_name)), start_date, end_date)
    fut = fut.reset_index(['date']).set_index(['date', 'asset']).sort_index(axis=0)
    return fut

def get_fut_index( csv_name='fut_index',start_date='2012-01-01', end_date='2019-06-01'):

    def find_date(df, start_date, end_date):
        pointer = 'date>="{}" and date<="{}"'.format(
            start_date, end_date)
        return df.query(pointer)
    idx = find_date(pd.DataFrame.from_csv('..\save\\{}.csv'.format(csv_name)), start_date, end_date)
    return idx

def data_processing(fut):
    # print(fut)
    fut = fut.reset_index(['date'])
    fil = fut.groupby('date').apply(lambda x: x.sort_values('volume', ascending=False).head(2)) \
        .drop(['date'], axis=1)
    fil_near = fil.reset_index(['date']).groupby('date').apply(
        lambda x: x.sort_values('asset', ascending=True).head(1)).drop(['date'], axis=1)
    fil_far = fil.reset_index(['date']).groupby('date').apply(
        lambda x: x.sort_values('asset', ascending=False).head(1)).drop(['date'], axis=1)

    return {'far': fil_far, 'near': fil_near}


def adf_test(s1, pvalues=[0.1, 0.05, 0.01]):
    s1 = s1.dropna()
    p = adfuller(s1)[1]
    print("The p value is {}".format(p))
    for i in pvalues:
        print("The adf test of under {}% is {}:".format(i * 100, p < i))
    return (adfuller(s1)[1] < pvalues[0])


def half_life(se):
    # se = ba.yport
    se = se.dropna()
    delta = se - se.shift(1)

    Y = np.array(delta[1:])
    X = np.array(se.shift(1)[1:])
    X = sm.add_constant(X)
    reg = sm.OLS(Y, X)
    result = reg.fit()

    # print(result.params)
    lookback = -np.log(2.0) / result.params[-1]
    lookback_ = int(lookback) + 1
    print('len', len(se))
    print('lookback', lookback_)
    return lookback_


def adf_backtest(se, p1=5, lookback=60):
    m1_ = se.rolling(window=p1, center=False).mean()
    ma_ = se.rolling(window=lookback, center=False).mean()
    std_ = se.rolling(window=lookback, center=False).std()
    profi = (m1_ - ma_) / std_
    # print(profi)

    delta = se.pct_change(1)
    delta = delta.shift(-1)
    delta * x
    pnl = profi.shift(1) * delta

    return pnl


def johansen_coint_(merged_s, pvalues=0.05):
    merged_s = merged_s.dropna()
    result = coint_johansen(merged_s, 0, 1)
    trace_stat = result.lr1
    max_stat = result.lr2
    cvm = result.cvm
    cvt = result.cvt

    def crit_range(st, crits):
        for i, _ in enumerate(st):
            print("The t-stat of it is {}".format(st[i]))
            if (st[i] <= crits[i][0]):
                print('r<{} failed being rejected.'.format(i + 1))
            elif (st[i] <= crits[i][1]):
                print('r<{} rejected at 90%.'.format(i + 1))
            elif (st[i] <= crits[i][2]):
                print('r<{} rejected at 95%.'.format(i + 1))
            else:
                print('r<{} rejected at 99%.'.format(i + 1))

    print("Maximum statistic testing...")
    crit_range(max_stat, cvm)
    print("Tracing statistic testing...")
    crit_range(trace_stat, cvt)
    return result.eig


def coint_test(data, select_col='close', pvalue=0.1):
    coint_pairs = []
    df = data.copy().dropna()[select_col]
    asset_index = list(set(df.index.get_level_values(1)))
    # print(asset_index)

    for i in range(len(asset_index)):
        for j in range(i + 1, len(asset_index)):
            s1 = df.loc[(slice(None), asset_index[i])]
            s2 = df.loc[(slice(None), asset_index[j])]
            date_index = s1.index.intersection(s2.index)
            # print(date_index)
            s1 = s1.loc[date_index]
            s2 = s2.loc[date_index]
            # print(s1.index)
            if (len(s1) != 0):
                conint_ = coint(s1, s2)
                if conint_[1] < pvalue:
                    coint_pairs.append([asset_index[i], asset_index[j]])
    return coint_pairs


def pairs_zs_backtest(df, zscore, half_lf, select_col='close'):
    #    # df = df.copy()[select_col]
    #    # print(coint_pair)
    #    # s1 = df[(slice(None),coint_pair[0])]
    #    # s2 = df[(slice(None),coint_pair[1])]
    #    # forward_ret = pd.concat([s1,s2],axis=1)
    #    # print(forward_ret)
    #    date_index = s1.index.intersection(s2.index)
    #    ratios = s1.loc[date_index] / s2.loc[date_index]
    #    # print(ratios)
    #    m1_ = ratios.rolling(window=p1,center=False).mean()
    #    m2_ = ratios.rolling(window=p2,center=False).mean()
    #    std_ = ratios.rolling(window=p2,center=False).std()
    #    zscore = (m1_ - m2_) / std_
    #    print(zscore)
    #    # print(zscore)
    money = 0
    countS1 = 0
    countS2 = 0
    date_index = df.index
    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]
    zscore = zscore[date_index]
    ratios = s1.loc[date_index] / s2.loc[date_index]

    upper = np.sqrt(half_lf / (half_lf - 2))
    lower = -np.sqrt(half_lf / (half_lf - 2))
    print("upper is {}".format(upper))
    print("lower is {}".format(lower))
    for i in date_index:

        if (zscore[i] > upper):
            money += s1[i] - s2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
        elif zscore[i] < lower:
            money -= s1[i] - s2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
        elif abs(zscore[i]) < 0.5:
            money += countS1 * s1[i] + countS2 * s2[i]
            countS1 = 0
            countS2 = 0
    print("The return is {}".format(money))
    return money


def data_reformat(df, name):
    return pre_data_[name]['close'].reset_index(['asset']).drop(labels=['asset'], axis=1)


def count_z_score(df, p1=5, p2=10):
    ratios = df.iloc[:, 0] / df.iloc[:, 1]
    m1_ = ratios.rolling(window=p1, center=False).mean()
    m2_ = ratios.rolling(window=p2, center=False).mean()
    std_ = ratios.rolling(window=p2, center=False).std()

    zscore = (m1_ - m2_) / std_
    return zscore.dropna()


if __name__ == "__main__":
    df = get_multi_fut_data(future_name1='al', future_name2="cu")
    print(df)
    print("Johnson test is:")
    johansen_coint_(df)
    columns = df.columns
    print("\n")
    print("The adf test of {} and {} :".format(columns[0], columns[1]))
    adf_test(df.iloc[:, 0] / df.iloc[:, 1])
    print("\n")

    half_lf = half_life(df.iloc[:, 0] / df.iloc[:, 1])
    print(half_lf)
    z_score = count_z_score(df, p2=half_lf)
    # print(upper,lower)
    #   print(z_score)
    z_score.plot()
    money = pairs_zs_backtest(df, z_score, half_lf)
    # print(df)
    # print(near,far)
    # print(pre_data_)
    # print(merged_s)

    # print(test_st)
    # lookback = half_life(test_st)
    # print(adf_backtest(test_st,lookback=lookback))
    # johansen_coint_(pd.concat([test_st,test_st2],axis=1))