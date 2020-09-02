import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import datetime_process as dtp
from fastbacktest import FastBacktest
from Performance import Performance
from backtest_params_pool import *
from scipy import stats
import re
import warnings
import time
from params_gen import *

warnings.filterwarnings("ignore")

class IdiosyncraticBacktest(FastBacktest):
    def __init__(self,params,data_proxy=None):
        # print(pa)
        FastBacktest.__init__(self, params=params, data_proxy=data_proxy)
        self.method_ = params['method']
        self.pre_data_ = None
        self.trading_df = None
        self.data_forward = None
        self.return_ = None
        self.fut_df = None
        self.fut_index_df = None
        self.delay = 0
        self.if_opposite = False

        self.qcut = params['qcut']
        self.rev_backtime = params['rev_backtime']
        try:
            self.mom_month = params['mom_month']
        except:
            self.mom_month = 3
        try:
            self.weight_method = params['weight_method']
        except:
            self.weight_method = 'equal'

    def run(self):
        self.data_processing()
        self.compute_forward()
        return self.cal_performance()

    def data_processing(self):
        self.daily_fut_df = self.data_proxy.get_daily_df()
        self.daily_fut_df = self.dateindex_2multiindex(self.daily_fut_df)

        self.fut_df = self.data_proxy.get_main_df()
        self.fut_index_df = self.data_proxy.get_future_index_df()['close']
        # self.pre_data_ = self.fut_main_unique(self.fut_df)
        # breakpoint()
        self.pre_data_ = self.fut_df
        self.pre_data_ = self.dateindex_2multiindex(self.pre_data_)


    def compute_forward(self):
        self.trading_df = self.generate_trading_days(self.pre_data_)

        if self.method_ == 'imom':
            self.data_forward = self.return_period_1m(self.trading_df)
            self.data_forward = self.momentum(self.trading_df, self.mom_month)
            self.data_forward = self.__imom(self.data_forward,'mom',self.fut_index_df,delta=self.rev_backtime,weights = self.weight_method)

        elif self.method_ == 'mom':
            self.data_forward = self.return_period_1m(self.trading_df)
            self.data_forward = self.momentum(self.trading_df, self.mom_month)
            self.data_forward = self.data_forward.dropna()

        elif self.method_ == 'ivol' or self.method_ == 'iskew' or self.method_ == 'e_iskew':
            self.data_forward = self.return_period_1m(self.pre_data_)
            self.data_forward = self.__ivol(self.data_forward,self.method_,self.fut_index_df,qcut=self.qcut, delta=self.rev_backtime,weights = self.weight_method,if_opposite=self.if_opposite)
            self.data_forward = self.generate_trading_days(self.data_forward)

        #
        # elif self.method_ == 'return':
        #     self.data_forward = self.data_forward.dropna()

    def fut_price_filled_(self):
        df = self.daily_fut_df.copy()
        df.update(self.data_forward)
        return

    def fut_main_unique(self,main_df):

        for date in main_df.index:
            asset_pool = set(main_df.loc[date].instrument)
            len1 = len(set(main_df.loc[date].instrument))
            len2 = len(main_df.loc[date].code)
            print(len1-len2)
            # print(len(main_df.loc[date]))
            #
            # main_df.loc[date] = main_df.loc[date].drop_duplicates(['instrument'])
            # print(len(main_df.loc[date]))
            # print('_________________________________________')

        return

    def momentum(self,df,num_month):
        df = df
        def cal_momentum(df):
            df["mom"] = df["return"].rolling(window=num_month, min_periods=1).mean()
            return df
        df = df.groupby(["instrument"]).apply(cal_momentum)
        return df.dropna()

    def generate_trading_days(self,df):
        dx = df.copy()
        return dtp.generate_trading_days(dx)

    def return_period_1m(self,df,period=1):
        delta = df.groupby('instrument')['close'].pct_change(period)
        delta = pd.DataFrame(delta)
        delta['instrument'] = df['instrument']
        delta = delta.groupby('instrument')['close'].shift(-period)
        df['return'] = delta
        return df

    def __imom(self,df,selected_factor,mkt_data,delta=360, weights='equal'):
        r_data = df[[selected_factor,'instrument']].copy()
        r = pd.DataFrame(r_data).dropna()
        ifactor = pd.DataFrame(index=r_data.index)
        dateindex = r.index.get_level_values(0).unique()
        # assetindex = r.index.get_level_values(1).unique()
        mkt = mkt_data.loc[dateindex].copy()
        for date in dateindex:
            end_date = date
            start_date = dtp.shift_time_(date, -delta)
            datelist = mkt[(mkt.index <= end_date) & (mkt.index >= start_date)].index
            mkt_ = mkt.loc[datelist]
            # print(datelist)
            # print(mkt_)
            # print('==================================================')
            if len(mkt_) >= int(delta/45.0):
                x_train = np.array(mkt_).reshape(-1, 1)
                for instrument in r.loc[date,'instrument']:
                    _pre_r = r.loc[datelist]
                    _pre_r = _pre_r.loc[_pre_r['instrument'] == instrument][selected_factor]
                    # index_date_r = _pre_r.index.get_level_values(0)
                    index_asset_r = _pre_r.index.get_level_values(1).unique()
                    y_train = np.array(_pre_r).reshape(-1, 1)
                    # y_train = np.array(r[r['inst']].loc[(datelist,), :]).reshape(-1, 1)
                    if (len(y_train) == len(x_train)):
                        linreg = LinearRegression()
                        linreg.fit(x_train, y_train)
                        e_ = np.std((linreg.predict(X=x_train) - y_train),
                                        ddof=1)
                        ifactor.loc[(date,index_asset_r),'imom'] = e_
        # print('==================================================')
        ifactor = ifactor['imom'].dropna()
        # ifactor['instrument'] = r['instrument']
        # ifactor = ifactor.groupby('instrument')[selected_factor].shift(1).dropna()
        ifactor = self.quantilize_ivol_(ifactor, weights)
        df['imom'.format(selected_factor)] = ifactor
        return df.dropna()


    def __ivol(self,df,selected_factor,mkt_data,delta=50, qcut = 4,weights='equal', if_opposite = False):
        r_data = df[['return','instrument']].copy()
        r = pd.DataFrame(r_data).dropna()
        ifactor = pd.DataFrame(index=r_data.index)
        ifactor['instrument'] = r_data['instrument']
        ifactor['e_iskew'] = np.NaN
        # dateindex = r.index.get_level_values(0).unique()
        dateindex = dtp.generate_month_end(r)
        last_date = dateindex[0]
        total_len = len(dateindex)
        counter = 0
        # assetindex = r.index.get_level_values(1).unique()
        # mkt = mkt_data.loc[dateindex]
        mkt = mkt_data.copy()
        start = time.clock()
        for date in dateindex:
            counter = counter + 1
            if (counter % 5) == 0:
                elapsed = (time.clock() - start)
                total_time  = (elapsed / (counter) * (total_len))
                print ('Time processed remained : {:.2f}/{:.2f}'.format(elapsed,total_time))
                print('{}/{} processing...'.format(counter,total_len))
            end_date = date
            start_date = dtp.shift_time_(date, -delta)
            datelist = mkt[(mkt.index <= end_date) & (mkt.index >= start_date)].index
            mkt_ = mkt.loc[datelist]
            # print(datelist)
            # print(mkt_)
            # print('==================================================')
            if len(mkt_) >= 16:
                for instrument in r.loc[date,'instrument']:
                    _pre_r = r.loc[datelist]
                    _pre_r = _pre_r.loc[_pre_r['instrument'] == instrument]['return']
                    # index_date_r = _pre_r.index.get_level_values(0)
                    index_asset_r = r.loc[date].loc[r.loc[date,'instrument'] == instrument].index
                    index_date_r = _pre_r.index.get_level_values(0).unique()
                    # print(date,index_asset_r)
                    # print(mkt_[index_date_r])
                    x_train = mkt_[index_date_r].values
                    y_train = _pre_r.values
                    if (len(y_train) == len(x_train))and (len(y_train)>=16):

                        X_ = sm.add_constant(x_train)
                        results = sm.OLS(y_train, X_, missing='drop').fit()
                        hist_skew, hist_vol = stats.skew(results.resid), stats.tstd(results.resid)
                        # e_ = np.std((linreg.predict(X=x_train) - y_train),
                        #                 ddof=1)
                        ifactor.loc[(date, index_asset_r), 'ivol'] = hist_vol
                        ifactor.loc[(date, index_asset_r), 'iskew'] = hist_skew
                    else:
                        ifactor.loc[(date, index_asset_r), 'debug_length'] = len(x_train)
                        # print(x_train)
                        # print(y_train)
                        # print(instrument)

                if selected_factor == 'e_iskew':
                    asset_this = ifactor.loc[date]['instrument'].unique()
                    asset_last = ifactor.loc[last_date]['instrument'].unique()
                    asset_pool = list(set(asset_this).intersection(set(asset_last)))

                    Y = ifactor.loc[(date,slice(None)),'iskew'].loc[ifactor.instrument.isin(asset_pool)].values
                    X = ifactor.loc[(last_date,slice(None)),['iskew','ivol']].loc[ifactor.instrument.isin(asset_pool)].values
                    # print('==================================================')
                    # print(len(asset_pool))
                    # print(X)
                    # print(len(X/2) - len(Y))
                    test_x = r_data.loc[(date,slice(None)),'return']

                    if not(np.isnan(X).all()) and not(np.isnan(Y).all()):
                        X = sm.add_constant(X)
                        results = sm.OLS(Y, X, missing='drop').fit()
                        coef = results.params
                        predictor_t = ifactor.loc[(date,slice(None)),['iskew','ivol']].loc[ifactor.instrument.isin(asset_pool)].values
                        ones = np.ones([len(predictor_t), 1])
                        predictor_t = np.append(ones, predictor_t, 1)
                        exp_skew = np.inner(predictor_t, coef)
                        # print(len(exp_skew))
                        # print('==================================================')
                        # sx = ifactor.loc[ifactor.instrument.isin(asset_pool)]
                        guided_asset_index = ifactor.loc[ifactor.instrument.isin(asset_pool)].loc[date].index
                        ifactor.loc[(date,guided_asset_index),['e_iskew']] = exp_skew
                        # print(ifactor.loc[ifactor.instrument.isin(asset_pool)].loc[(date,slice(None)),['e_iskew']])

            last_date = date
        # print('==================================================')
        ifactor = ifactor
        # print(ifactor)
        # ifactor['instrument'] = r['instrument']
        # ifactor = ifactor.groupby('instrument')[selected_factor].shift(1).dropna()
        ifactor_exp = self.quantilize_ivol_(ifactor[selected_factor], weights, qcut=qcut,if_opposite = if_opposite)
        # df['{}'.format(selected_factor)] = ifactor_exp
        dx = self.daily_fut_df.copy()
        dx['{}'.format(selected_factor)] = ifactor_exp
        dx['indicator'] = ifactor[selected_factor]
        return dx

    def quantilize_ivol_(self,df, weights, qcut = 4, if_opposite = False):
        def quantile_calc(x, q):
            return pd.qcut(x, q=q, labels=False, duplicates='drop')
        print(qcut)
        qindex = df.groupby('date').apply(quantile_calc, qcut)
        lower_ = qindex[qindex == (0.0)].index.tolist()
        upper_ = qindex[qindex == (qcut - 1)].index.tolist()
        ignored_ = qindex[(qindex < (qcut - 1)) & (qindex > (0.0))].index.tolist()
        if (weights == 'equal'):
            df.loc[lower_] = (1.0) * (-1.0 * (-1+2*int(if_opposite)) )
            df.loc[upper_] = (-1.0) * (-1.0 * (-1+2*int(if_opposite)))
        elif (weights == 'weighted'):
            df.loc[upper_] = (-1.0) * df.loc[upper_] * (-1.0 * (-1+2*int(if_opposite)) )
            df.loc[lower_] = (1.0) * df.loc[lower_] * (-1.0 * (-1+2*int(if_opposite)) )
        df.loc[ignored_] = np.NaN
        return df

    @staticmethod
    def multiindex_2dateindex(df,col='code'):
        return df.reset_index(['date',col]).set_index(['date']).sort_index(axis=0)
    @staticmethod
    def dateindex_2multiindex(df,col='code'):
        return df.reset_index(['date']).set_index(['date',col]).sort_index(axis=0)

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

            factor_data[selected_factor] = factor_data.copy().groupby('instrument')[selected_factor].shift(self.delay)
            weights = factor_data.groupby('date')[selected_factor].apply(to_weights, demeaned)
            # print(weights)
            weighted_returns = factor_data[get_forward_returns_columns(factor_data.columns)].multiply(weights, axis=0)
            returns = weighted_returns.groupby(level='date').sum()
            return returns
        multi_alpha_weights = ret_period_m(self.data_forward)
        debug_weights = multi_alpha_weights.dropna()
        self.return_ = {}
        # self.return_[self.method_] = factor_returns_adjusted(multi_alpha_weights, self.method_, demeaned=False)['period_1']
        self.return_[self.method_] = multi_alpha_weights
        return self.return_

if __name__ == "__main__":

    fast_test = params_generator({}, weight_method_=['equal'], method_=['e_iskew'], qcut_=[10],
                                 rev_backtime_=[30])
    # print(fast_test)
    im = IdiosyncraticBacktest(fast_test['idiosyn_6_equal_e_iskew_30_10'])
    df = im.run()
    pt = Performance(multi_alpha_weights=im.data_forward,select_factor='e_iskew')
    # print(dtp.generate_trading_days(df))

