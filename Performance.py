# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import configparser
import Paint
from Paint import plot_daily_pnl
import os
import platform
import sys


class Performance(object):
    # 输入：trade record : dataframe, columns: date, instrument, indicator, code, holding
    # 输入：res_dic: 可能存在 "indi": 因子记录
    #                可能存在 "holfing": 持仓记录
    #                必然存在 "daily_pnl": 日盈亏

    # multi_alpha_weights = None, select_factor = None, demeaned = True, group_adjust = False, neutral_type = 'market',
    def __init__(self, res_dic,  booksize = 10000000):
        print("Peformance init...")
        self.res_dic = res_dic
        self.booksize = booksize

        self.indi_df = self.res_dic["indi"] if "indi" in self.res_dic else None
        self.hold_df = self.res_dic["holding"] if "holding" in self.res_dic else None

        # TODO: name: daily pnl ? or daily value change percent

        if "daily_pnl" in self.res_dic:
            self.daily_pnl = self.res_dic["daily_pnl"]
        else:
            raise ValueError("结果必须包含daily pnl")

        self.performance_processing()

    # todo: create performance:
    #  indicator, holding, pnl, daily pnl, year pnl, ir, sharpe, std, turnover, fitness, daily turnover, margin
    #, trade_record, multi_alpha_weights, select_factor, demeaned, group_adjust, neutral_type
    def performance_processing(self):
        self.performance = {}
        self.res = {}

        # save the existing performance indicators
        if self.indi_df is not None:
            self.performance["indicator"] = self.indi_df
        if self.hold_df is not None:
            self.performance["hold"] = self.hold_df

        self.max_drawdown = self.cal_max_drawdown(self.res_dic["daily_pnl"])

        self.daily_pnl = self.daily_pnl[["pnl"]] * self.booksize
        self.pnl = self.cal_pnl(daily_pnl=self.daily_pnl)       #self.daily_pnl.groupby(self.daily_pnl.index.year).sum()

        # year pnl
        # basic result indicator
        self.returns = self.cal_return(pnl=self.pnl) # self.pnl * 2.0 / self.booksize
        self.ir = self.cal_ir(daily_pnl=self.daily_pnl)
        self.sharpe = self.cal_sharpe(ir=self.ir)
        self.std = self.cal_std(daily_pnl=self.daily_pnl)

        if self.hold_df is not None:
            self.turnover = self.cal_turnover( )
            self.fitness = self.cal_fitness(sharpe=self.sharpe,turnover=self.turnover)
            self.margin  = self.cal_margin(pnl = self.pnl)
            self.performance["turnover"] = self.turnover
            self.performance["fitness"] = self.fitness
            self.performance['margin'] = self.margin

        self.performance["daily_pnl"] = self.daily_pnl
        self.performance["pnl"] =  self.pnl
        self.res["pnl"]  = self.pnl["pnl"].mean()
        self.performance['returns'] = self.returns
        self.res["returns"] = self.returns["return"].mean()
        self.performance['ir'] = self.ir
        self.res["ir"] = self.ir["ir"].mean()
        self.performance['sharpe'] = self.sharpe
        self.res["sharpe"] = self.sharpe["sharpe"].mean()
        self.performance["std"] = self.std
        self.res["std"] = self.std["std"].mean()
        self.performance["max_drawdown"] = self.max_drawdown
        self.res["max_drawdown"] = self.max_drawdown["max_drawdown"].values[0]
        # print(self.res)
        self.performance["res"] = pd.DataFrame(list(self.res.items()), columns =  ["indi", "value"])
        self.performance["res"] = self.performance["res"].set_index(["indi"])
        # print(self.res)
        # breakpoint()

    def DailyPnl(self,factor_data, selected_factor,demeaned, group_adjust,neutral_type):
        return self.factor_returns_adjusted(factor_data, selected_factor,demeaned, group_adjust,neutral_type)

    def ret_period_m(self, df, period = 1):
        delta = df['close'].groupby(level='code').pct_change(period)
        delta = delta.groupby(level='code').shift(-period)
        df['period_{p}'.format(p=period)] = delta
        return df.dropna()

    def factor_returns(self,factor_data, selected_factor,demeaned, group_adjust,neutral_type):

        def to_weights(group, is_long_short):
            if is_long_short:
                demeaned_vals = group - group.mean()
                return demeaned_vals / demeaned_vals.abs().sum()
            else:
                return group / group.abs().sum()


        grouper = [factor_data.index.get_level_values('date')]

        if group_adjust:
            grouper.append('group')

        weights = factor_data.groupby(grouper)[selected_factor] \
            .apply(to_weights, demeaned)
        # print(weights)

        if group_adjust:
            weights = weights.groupby(level='date').apply(to_weights, False)

        weighted_returns = \
            factor_data[self.get_forward_returns_columns(factor_data.columns)] \
            .multiply(weights, axis=0)

        returns = weighted_returns.groupby(level='date').sum()
        return returns

    def factor_returns_adjusted(self,factor_data, selected_factor,demeaned,group_adjust,neutral_type):
        def to_weights(group, is_long_short):

            if is_long_short:
                demeaned_vals = group - group.mean()
                if (len(demeaned_vals)!=1) or (any(demeaned_vals.tolist())):
                    return demeaned_vals / demeaned_vals.abs().sum()
                else:
                    demeaned_vals.values[0] = 0.0
                    return demeaned_vals
            else:
                demeaned_vals = group
                if not all(i_ ==0.0 for i_ in demeaned_vals.tolist()):
                    return group / group.abs().sum()
                else:
                    return demeaned_vals

        if neutral_type == 'market':
            # print(factor_data[selected_factor])
            weights = factor_data.groupby('date')[selected_factor].apply(to_weights,demeaned)
            # print('MARKET neutralized...')

        elif neutral_type == 'None':
            weights = factor_data.groupby('date')[selected_factor].apply(to_weights,demeaned)
            # print('None neutralized...')

        elif neutral_type == 'industry':
            grouper = [factor_data.index.get_level_values('date')]
            grouper.append('group')
            # print(factor_data['group'])
            alpha_group_industry = factor_data.groupby(grouper)[selected_factor]
            weights = alpha_group_industry.apply(to_weights,demeaned)
            # print('Start Industry neutralized...')

            # industry_group = factor_data['group']
            # print(industry_group)

            # industry_index = pd.MultiIndex.from_product([industry_group.index.levels[0],industry_group])
            # print(industry_index)
            # alpha_group = alpha_group_industry.mean().loc()
            # print(alpha_group_industry.mean())
            # print('df',alpha_group_industry)
            # print(alpha_group_industry)
        # weights = pd.DataFrame(weights)

        # print(weights.join(factor_data[selected_factor]))
        # print(weights)

        # if group_adjust:
        #     weights = weights.groupby(level='date').apply(to_weights, False)
        # print(factor_data[self.get_forward_returns_columns(factor_data.columns)])
        self.weights = weights
        weighted_returns = \
            factor_data[self.get_forward_returns_columns(factor_data.columns)] \
            .multiply(weights, axis=0)
        # print(weighted_returns)
        returns = weighted_returns.groupby(level='date').sum()
        return returns

    def cal_daily_pnl(self):
        date_df = pd.DataFrame()
        date_df["date"] = self.trade_record["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
        date_df["next_date"] = date_df["date"].shift(-1)
        date_df = date_df.dropna(axis=0, how="any")
        # right： 换仓时间为当天交易日接近结束时，先平仓
        date_df["date_range"] = date_df.apply(
            lambda r: pd.date_range(start=r["date"], end=r["next_date"], closed="right"), axis=1)
        date_index_df = pd.DataFrame({"date": date_df.date.repeat(date_df.date_range.str.len()),
                                      "date_val": np.concatenate(date_df.date_range.values)})
        trade_record = pd.merge(self.trade_record, date_index_df, on=["date"])
        trade_record = trade_record.drop(columns=["date"], axis=1)
        trade_record.rename(columns={"date_val": "date"}, inplace=True)
        df = self.data_proxy.get_daily_df(col_list=["code", "open", "close"])
        df["date"] = df.index
        df = df.reset_index(drop=True)
        trade_record = pd.merge(trade_record, df, on=["date", "code"], how="inner")
        # print("trade record size ")
        # print(sys.getsizeof(trade_record))
        trade_record["return"] = trade_record.apply(lambda r: (r["close"] - r["open"]) / r["open"], axis=1)
        # print(trade_record[:100])
        # print("-" * 7)
        trade_record["return"] = trade_record.apply(lambda r: r["return"] * r["holding"], axis=1)
        # print(trade_record[:100])
        del df, date_df, date_index_df
        daily_pnl = trade_record[["date", "return"]].groupby(["date"], as_index=False).sum()
        daily_pnl["return"] = daily_pnl["return"].map(lambda x: x * self.booksize)
        daily_pnl = daily_pnl.set_index(["date"])
        return daily_pnl

    def cal_max_drawdown(self, daily_pnl):
        # daily_pnl = pd.read_csv("daily_pnl.csv")
        daily_pnl = daily_pnl.copy()
        daily_pnl["daily_v_change"] = daily_pnl["daily_v_change"] + 1
        daily_pnl["cum_v"] = daily_pnl["daily_v_change"].cumprod()

        dates = daily_pnl.index
        values = daily_pnl["cum_v"].values

        high = values[0]
        start_date, end_date = 0, 0
        max_drawdown = 0

        for i in range(len(dates)):
            if values[i] > high:
                high = values[i]
                start_date = dates[i]
            elif (high - values[i]) / high > max_drawdown:
                max_drawdown = (high - values[i]) / high
                end_date = dates[i]

        drawdown = pd.DataFrame({"max_drawdown": [max_drawdown], "start_date": [start_date], "end_date": [end_date]})
        self.res["max_drawdown"] = max_drawdown
        # print(max_drawdown, start_date, end_date)
        # print(drawdown)
        return drawdown



    def cal_margin(self, pnl):
        yearsum_turnover=self.daily_turnover.groupby(self.daily_turnover.index.year).sum()
        yearsum_turnover = yearsum_turnover.to_frame(name="turnover")
        margin = pd.merge(pnl, yearsum_turnover, left_index=True, right_index=True)
        margin["margin"] = margin.apply(lambda r: r["pnl"] / (r["turnover"] * self.booksize), axis=1)
        margin = margin.drop(columns = ["pnl", "turnover"])
        return  margin

    def get_forward_returns_columns(self,columns):
        syntax = re.compile("^period_\\d+$")
        return columns[columns.astype('str').str.contains(syntax, regex=True)]


    def calc_daily_turnover(self,alpha_weights):
        all_dates = sorted(alpha_weights.keys())
        # print(all_dates)
        last_date = None
        turnover = {}

        for date in all_dates:
            w = alpha_weights[date]
            w.name = 'w'
            w_prev = alpha_weights[last_date] if last_date is not None else pd.Series(0, index=w.index)
            w_prev.name = 'w_prev'
            tmp = pd.concat([w, w_prev], axis=1).fillna(0)
            # print(tmp)
            turnover[date] = (tmp['w'] - tmp['w_prev']).abs().sum()
            # print(turnover[date])
            last_date = date
        turnover = pd.Series(turnover)
        turnover /= 2
        # print(turnover)
        self.daily_turnover = turnover
        return turnover

    def cal_turnover(self):
        alpha_weights = self.hold_df.copy()
        alpha_weights = pd.Series(alpha_weights["holding"].values, index=[alpha_weights["date"], alpha_weights["code"]])
        alpha_weights = alpha_weights.unstack(level=-2)
        dt =  self.calc_daily_turnover(alpha_weights)
        dt_year = pd.DataFrame(dt.groupby(dt.index.year).mean())
        dt_year.columns = ['turnover']
        return dt_year

    def cal_fitness(self,sharpe,turnover):
        fitness = pd.merge(sharpe, turnover, left_index=True, right_index=True)
        fitness["fitness"] = fitness.apply(lambda r: r["sharpe"]/r["turnover"], axis = 1)
        fitness = fitness.drop(columns = ["sharpe", "turnover"], axis =1)
        return fitness

    def cal_pnl(self, daily_pnl):
        pnl = daily_pnl.groupby(self.daily_pnl.index.year).sum()
        pnl.columns = ["pnl"]
        return pnl

    def cal_return(self, pnl):
        return_df = pnl * 2.0 / self.booksize
        return_df.columns = ["return"]
        return return_df

    def cal_ir(self, daily_pnl):
        ir = daily_pnl.groupby(daily_pnl.index.year).mean() / daily_pnl.groupby(daily_pnl.index.year).std()
        ir.columns = ["ir"]
        return ir

    def cal_sharpe(self, ir):
        sharpe = ir * np.sqrt(252)
        sharpe.columns = ["sharpe"]
        return sharpe

    def cal_std(self, daily_pnl):
        std = (daily_pnl.groupby(daily_pnl.index.year).std()) * np.sqrt(252)
        std.columns = ["std"]
        return std

    # 函数功能： 写入参数文件
    def save_config_file(self, params, file_name):
        config = configparser.ConfigParser()

        for key, value in params.items():
            params[key] = str(value)
        config["params"] = params

        with open(file_name, "w") as configfile:
            config.write(configfile)

    def save_performance(self, _name, param, if_print = False, booksize = 1000000):
        system = platform.system()
        sep_str = "/" if system.lower() == "linux" else "\\"

        eval_root_dir = "eval_opt_res"
        # 创建主目录歘来，策略目录
        strategy_dir  = eval_root_dir + sep_str +  _name.replace("*", sep_str)
        strategy_dir = ".." + sep_str + strategy_dir

        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)

        i = 0
        while os.path.exists('{}{}{}'.format(strategy_dir, sep_str, "res" + str(i))):
            i += 1
        os.makedirs('{}{}{}'.format(strategy_dir, sep_str, "res" + str(i)))

        param_file = '{}{}{}{}param.txt'.format(strategy_dir, sep_str, "res" + str(i), sep_str)
        self.save_config_file(params = param, file_name=param_file)

        for guideline in self.performance.keys():
            if if_print:
                print('Evaluate <{}>...'.format(guideline), self.performance[guideline])
            file_name = '{}{}{}{}{}'.format(strategy_dir,sep_str, "res" + str(i), sep_str,guideline)
            if guideline == "daily_pnl":
                Paint.plot_daily_pnl(self.performance[guideline], save_dir=file_name+".png")
            self.performance[guideline].to_csv(file_name + ".csv")

    @staticmethod
    def plot_all_Performance(factor_analyzer, _name,booksize = 1000000):
        for factor_i in factor_analyzer.factors_list:
            print('Evaluate <{}> performance...'.format(factor_i))
            # print(factor_i)
            p_demo_industry = Performance(factor_analyzer.cleaned_factor_data, factor_i, demeaned=True,
                                          neutral_type='industry')
            print(p_demo_industry.performance['turnover'])
            p_demo_none = Performance(factor_analyzer.cleaned_factor_data, factor_i, demeaned=True, neutral_type='None')
            pnl_industry_daily = p_demo_industry.performance['daily_pnl']
            pnl_none_daily = p_demo_none.performance['daily_pnl']

            plot_industry = {}
            plot_none = {}
            plot_industry['data'], plot_industry['name'] = Paint.plot_cumsum_pnl(pnl_industry_daily,'{}_{}_{}'.format(_name,factor_i, 'industry'),booksize,if_show=False)
            plot_none['data'], plot_none['name'] = Paint.plot_cumsum_pnl(pnl_none_daily,'{}_{}_{}'.format(_name, factor_i,'none'),booksize,if_show=False)
            Paint.compare_plot(plot_none, plot_industry,if_show=False)
            for guideline in ['turnover','fitness','sharpe']:
                Paint.plot_annual_guideline(guideline=p_demo_industry.performance[guideline],name='{}_{}_{}'.format(_name,factor_i,guideline),if_show=False)

