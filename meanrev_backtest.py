
from fastbacktest import FastBacktest
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from scipy import stats
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from Performance import Performance
import re
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)

from backtest_params_pool import params_meanrev

class MeanrevBacktest(FastBacktest):
    def __init__(self,params,data_proxy=None):
        FastBacktest.__init__(self, params=params, data_proxy=data_proxy)
        self.method_ = params['method']
        self.data_forward = None
        self.return_ = None
        self.fut_df = None
        self.fut_main_df = None
        self.start_date = params["start_date"]
        self.end_date = params["end_date"]

    def run(self):
        self.data_processing()
        self.data_test()
        self.compute_forward()
        return self.cal_performance()

    def cal_performance(self):
        def ret_period_m(df, period=1):
            delta = df['close'].groupby(level='code').pct_change(period)
            delta = delta.groupby(level='code').shift(-period)
            df['period_{p}'.format(p=period)] = delta
            return df.dropna()

        def factor_returns_adjusted(factor_data, selected_factor, demeaned):
            def get_forward_returns_columns(columns):
                syntax = re.compile("^period_\\d+$")
                return columns[columns.astype('str').str.contains(syntax, regex=True)]

            def to_weights(group, is_long_short):
                if is_long_short:
                    demeaned_vals = group - group.mean()
                    if (len(demeaned_vals) != 1) or (any(demeaned_vals.tolist())):
                        return demeaned_vals / demeaned_vals.abs().sum()
                    else:
                        demeaned_vals.values[0] = 0.0
                        return demeaned_vals
                else:
                    demeaned_vals = group
                    if not all(i_ == 0.0 for i_ in demeaned_vals.tolist()):
                        return group / group.abs().sum()
                    else:
                        return demeaned_vals

            weights = factor_data.groupby('date')[selected_factor].apply(to_weights, demeaned)
            self.weights = weights
            weighted_returns = \
                factor_data[get_forward_returns_columns(factor_data.columns)] \
                    .multiply(weights, axis=0)
            # print(weighted_returns)
            returns = weighted_returns.groupby(level='date').sum()
            return returns
        multi_alpha_weights = ret_period_m(self.data_forward)
        self.return_ = pd.DataFrame()
        self.return_[self.method_] = factor_returns_adjusted(multi_alpha_weights, 'trade', demeaned=False)
        return self.return_

    def compute_forward(self):
        def half_life(se):
            se = se.dropna()
            delta = se - se.shift(1)
            Y = np.array(delta[1:])
            X = np.array(se.shift(1)[1:])
            X = sm.add_constant(X)
            reg = sm.OLS(Y, X)
            result = reg.fit()
            lookback = -np.log(2.0) / result.params[-1]
            lookback_ = int(lookback) + 1
            # print('halflife is ...',lookback_)
            return lookback_
        def count_z_score(df, p1=5, p2=10):
            ratios = df.iloc[:, 0] / df.iloc[:, 1]
            m1_ = ratios.rolling(window=p1, center=False).mean()
            m2_ = ratios.rolling(window=p2, center=False).mean()
            std_ = ratios.rolling(window=p2, center=False).std()
            zscore = (m1_ - m2_) / std_
            return zscore.dropna()
        def zscore_backtest(df,threshold,compute_forward_index):
            ratios = (df.iloc[:, 0] / df.iloc[:, 1])
            zs = df['zscore'].dropna()
            lower = threshold[0]
            upper = threshold[1]
            compute_forward_df = pd.DataFrame(index=compute_forward_index,columns=['trade'])
            countS1 = 0.0
            countS2 = 0.0
            for date in zs.index:
                if zs[date] > upper:
                    countS1 -= 1.0
                    countS2 += ratios[date]
                    compute_forward_df.loc[date].iloc[0] = countS1
                    compute_forward_df.loc[date].iloc[1] = countS2
                elif zs[date] < lower:
                    countS1 += 1.0
                    countS2 -= ratios[date]
                    compute_forward_df.loc[date].iloc[0] = countS1
                    compute_forward_df.loc[date].iloc[1] = countS2
                elif abs(zs[date]) < 0.5:
                    countS1 = 0
                    countS2 = 0
                    compute_forward_df.loc[date].iloc[0] = countS1
                    compute_forward_df.loc[date].iloc[1] = countS2
                else:
                    compute_forward_df.loc[date].iloc[0] = countS1
                    compute_forward_df.loc[date].iloc[1] = countS2
            return compute_forward_df
            # print(compute_forward_df)
        df = MeanrevBacktest.merged_data(self.data_forward)
        df = df.dropna()
        half_lf = half_life(df.iloc[:,0] / df.iloc[:,1])
        df['zscore'] = count_z_score(df, p1=int(half_lf/5),p2=half_lf)
        upper = np.sqrt(half_lf / (half_lf - 2))
        lower = -np.sqrt(half_lf / (half_lf - 2))
        self.data_forward['trade'] = zscore_backtest(df,[lower,upper],self.data_forward.index)

    def data_processing(self):
        "data processing..."
        if self.method_ == '2nearest':
            self.fut_df = self.data_proxy.get_daily_df(fut_list=self.future_pool,start_date=self.start_date, end_date=self.end_date)
            self.data_forward = self.nearest_processing(self.future_pool[0])
        elif self.method_ == '2fut':
            self.fut_main_df = self.data_proxy.get_main_df(fut_list=self.future_pool,start_date=self.start_date, end_date=self.end_date)
            self.data_forward = self.two_fut_processing(self.future_pool)

    def data_test(self):
        "data testing..."
        df_to_test = MeanrevBacktest.merged_data(self.data_forward).dropna()
        print('Johansen test started...')
        self.johansen_coint_(df_to_test)
        print("ADF test started...")
        self.adf_test(df_to_test.iloc[:,0] / df_to_test.iloc[:,1])
        return

    def nearest_processing(self,fut_name):
        pointer = ' instrument == "{}" '.format(fut_name)
        fil = self.fut_df.query(pointer).groupby('date').\
            apply(lambda x: x.sort_values('volume',ascending=False).head(2)).reset_index(drop=True,level=-1)
        fil_near = fil.groupby('date').\
            apply(lambda x: x.sort_values('code',ascending=True).head(1)).reset_index(drop=True,level=-1)
        fil_far = fil.groupby('date').\
            apply(lambda x: x.sort_values('code',ascending=False).head(1)).reset_index(drop=True,level=-1)
        fil_near = MeanrevBacktest.dateindex_2multiindex(fil_near)
        fil_far = MeanrevBacktest.dateindex_2multiindex(fil_far)
        df = pd.DataFrame(pd.concat([fil_near["close"],fil_far["close"]]).sort_index(axis=0))
        return df

    def two_fut_processing(self, fut_list):
        pointer = ' instrument in {} '.format(fut_list)
        fut_name1 = fut_list[0]
        fut_name2 = fut_list[1]
        fil = self.fut_main_df.query(pointer)
        fil1 = MeanrevBacktest.dateindex_2multiindex(fil[fil['instrument']==fut_name1])
        fil2 = MeanrevBacktest.dateindex_2multiindex(fil[fil['instrument']==fut_name2])
        df = pd.DataFrame(pd.concat([fil1["close"],fil2["close"]]).sort_index(axis=0))

        return df

    def johansen_coint_(self,merged_s,pvalues=0.05):
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

    def adf_test(self,s1,pvalues=[0.1, 0.05, 0.01]):
        s1 = s1.dropna()
        p = adfuller(s1)[1]
        print("The p value is {}".format(p))
        for i in pvalues:
            print("The adf test of under {}% is {}:".format(i * 100, p < i))
        return (adfuller(s1)[1] < pvalues[0])

    @staticmethod
    def merged_data(df):
        merged_df = pd.DataFrame()
        df_ = MeanrevBacktest.multiindex_2dateindex(df)
        for date in df.index.levels[0]:
            merged_df.loc[date,'close_0'] = df_.loc[date]['close'][0]
            merged_df.loc[date,'close_1'] = df_.loc[date]['close'][1]
        return merged_df
    @staticmethod
    def multiindex_2dateindex(df,col='code'):
        return df.reset_index(['date',col]).set_index(['date']).sort_index(axis=0)
    @staticmethod
    def dateindex_2multiindex(df,col='code'):
        return df.reset_index(['date']).set_index(['date',col]).sort_index(axis=0)

if __name__ == "__main__":
    mx = MeanrevBacktest(params_meanrev)
    mx.run()
    print(mx.data_forward)

