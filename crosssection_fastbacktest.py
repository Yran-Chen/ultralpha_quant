# -*- coding: utf-8 -*-
from indicator_generator import  IndicatorGenerator
from fastbacktest import  FastBacktest
from future_pool_generator import FuturePoolGen
import pandas as pd
import numpy as np
import time
import math
import os
import sys
import configparser
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from Performance import Performance
import Paint

class FastCrossSectionalBacktest(FastBacktest):

    def __init__(self, params, data_proxy = None):
        FastBacktest.__init__(self, params=params, data_proxy=data_proxy)

        # 创造策略指标生成器
        self.indicator_genarator = IndicatorGenerator(self.data_proxy)
        self.set_params(params)
        self.ascending_df = {"term_structure": False,
                             "hedging_pressure": False,
                             "momentum": False,
                             "volatility": False,
                             "open_interest": False,
                             "inflation": False,
                             "skewness": True,
                             "amivest": True, # liquidity
                             "amihud": False,  #liquidity
                             "currency": True,
                             "value": False}

    def set_params(self, params):

        self.params = params

        #########################################################################
        # todo： 是否提到fastbacktest类来确定是否固定底池
        try:
            self.fut_pool_gen_method = params["fut_pool_gen_method"]
        except:
            raise RuntimeError("请确定回测时期货池生成方式， 动态生成 / 固定方式生成")

        if self.fut_pool_gen_method == "dynamic":
             try:
                self.fut_pool_gen_param = params["fut_pool_gen_param"]
                self.fut_pool_gen = FuturePoolGen(data_proxy = self.data_proxy, indi_generator=self.indicator_genarator)
             except:
                 raise RuntimeError("请确定动态期货池产生参数")
        elif self.fut_pool_gen_method == "static":
            try:
                self.future_pool = params["future_pool"]
            except:
                raise RuntimeError("请确定固定期货池")

        #########################################################################

        # 确定操作过程所使用那种合约，主力合约还是次月合约
        try:
            self.is_main_contract_op = params["is_main_contract_op"]
        except:
            raise RuntimeError("请确定回测时根据指标操作的合约")

        # 确定计算指标过程所使用哪种合约，主力合约还是次月合约
        try:
            self.is_main_contract_indi = params["is_main_contract_indi"]
        except:
            raise RuntimeError("请确定回测时根据指标操作的合约")

        # 假如不是操作主力合约，需要输入nearest index，比如1，2， 3， 4
        try:
            self.n_nearest_index_op = params["n_nearest_index_op"] if not self.is_main_contract_op else None
        except:
            raise RuntimeError("假如不是操作主力合约，需要输入nearest index，比如1，2， 3， 4")

        # 假如不是主力合约来计算因子，需要输入nearest index，比如1，2， 3， 4
        try:
            self.n_nearest_index_indi = params["n_nearest_index_indi"] if not self.is_main_contract_indi else None
        except:
            raise RuntimeError("假如不是操作主力合约，需要输入nearest index，比如1，2， 3， 4")

        #########################################################################

        try:
            self.freq_unit = params["freq_unit"]
        except:
            raise RuntimeError("请输入操作时间节点频率， M 或者 D")

        try:
            self.freq = params["freq"]  if self.freq_unit == "D" else 1
        except:
            raise RuntimeError("请输入操作时间节点天数")

        #########################################################################
        try:
            self.indicator_cal_method = params["indicator_cal_method"]
        except:
            raise RuntimeError("请输入希望测算的计算截面因子的方式")

        try:
            self.if_save_res = params["if_save_res"]
        except:
            raise RuntimeError("请输入是否希望存储结果")

    def run(self, if_print = False):

        self.res = {}
        future_pool = None

        # 生成期货池: future_pool
        # 判断生成期货池的方式： 固定底池/动态期货池
        if self.fut_pool_gen_method == "dynamic":
            self.fut_pool_gen_param["start_date"] = self.start_date
            self.fut_pool_gen_param["end_date"] = self.end_date
            self.fut_pool_gen_param["freq_unit"] = self.freq_unit
            self.fut_pool_gen_param["freq"] = self.freq
            future_pool = self.fut_pool_gen.get_future_pool(param=self.fut_pool_gen_param)
        elif self.fut_pool_gen_method == "static":
            future_pool = self.future_pool

        # print ("future pool ")
        # print (future_pool[:200])
        # print (future_pool[-200:])
        if isinstance(future_pool, pd.DataFrame):
            print (future_pool["date"].drop_duplicates().tolist())
        elif isinstance(future_pool, list):
            print (future_pool)

        t0 = time.time()
        # 得到每次操作的期货品种 op_code_df
        # date0 a
        #       b
        #       c
        # date1 a...
        # print(self.is_main_contract_indi)

        if self.is_main_contract_op:
            op_code_df = self.data_proxy.get_main_df_with_freq(if_main=True, start_date=self.start_date,
                                                               end_date=self.end_date, future_pool=future_pool,
                                                               freq=self.freq, freq_unit=self.freq_unit)
        else:
            if_check_vol = self.params["if_check_vol"] if "if_check_vol" in self.params else False
            print(" if check vol ", if_check_vol)
            op_code_df = self.data_proxy.get_nearest_contract_with_time_freq(n_nearest_index=self.n_nearest_index_op,
                                                                             start_date=self.start_date,
                                                                             end_date=self.end_date,
                                                                             future_pool=future_pool, freq=self.freq,
                                                                             freq_unit=self.freq_unit,
                                                                             if_check_vol=if_check_vol)
        op_code_df["date"] = op_code_df.index
        op_code_df = op_code_df.reset_index(drop = True)
        op_code_df = op_code_df.sort_values(by = ["instrument", "date"], ascending = [True, True ])


        op_code_df["date"] = op_code_df.index
        op_code_df = op_code_df.reset_index(drop=True)
        print( "Time for fetching operation code df ", time.time() - t0)

        # print("op code df")
        # print (op_code_df["date"].drop_duplicates().tolist())

        function_params = {"method": self.indicator_cal_method,
                           "op_code_df": None,
                           "future_pool": future_pool,
                           "start_date": self.start_date,
                           "end_date": self.end_date,
                           "is_main_contract_indi": self.is_main_contract_indi,
                           "n_nearest_index_indi": self.n_nearest_index_indi,
                           "freq_unit": self.freq_unit,
                           "freq": self.freq}

        # 属于每个因子计算特有的default参数
        num_day_value = {
            "hedging_pressure": 30,
            "volatility": 200 ,
            "liquidity": 60,
            "skewness": 252,
            "open_interest": 30
        }

        num_duration_value = {
        "momentum": 12,
        "currency": 42,
        "inflation": 42,
        }

        if self.indicator_cal_method in ["hedging_pressure", "volatility", "open_interest"] :
            num_days = self.params["num_days"] if "num_days" in self.params else num_day_value[self.indicator_cal_method]
            function_params["num_days"] = num_days
        elif self.indicator_cal_method ==  ["momentum", "currency"]:
            num_durations = self.params["num_durations"] if "num_durations" in self.params else num_duration_value[self.indicator_cal_method]
            function_params["num_durations"] = num_durations
        elif self.indicator_cal_method == "value":
            num_days = self.params["num_days"] if "num_days" in self.params else 30
            num_years = self.params["num_years"] if "num_years" in self.params else 3
            weighted = self.params["weighted"] if "weighted" in self.params else False
            function_params["num_days"] = num_days
            function_params["num_years"] = num_years
            function_params["weighted"] = weighted
        elif self.indicator_cal_method == "liquidity":
            num_days = self.params["num_days"] if "num_days" in self.params else num_day_value[self.indicator_cal_method]
            method = self.params["method"] if "method" in self.params else "amivest"
            function_params["liq_method"] = method
            function_params["num_days"] = num_days
        elif self.indicator_cal_method == "inflation":
            num_durations = self.params["num_durations"] if "num_durations" in self.params else num_duration_value[self.indicator_cal_method]
            index_col = self.params["index_col"] if "index_col" in self.params else "y2y_0"
            function_params["index_col"] = index_col
            function_params["num_durations"] = num_durations

        print(self.params)
        # 计算因子指数
        self.indicator_df = self.indicator_genarator.cal_indicator(function_params)


        # 计算对于每种期货买入卖出量
        op_df = pd.merge(self.indicator_df, op_code_df, on=["date", "instrument"], how="right")
        op_df = op_df.dropna(how = "any", axis = 0)

        top = self.params["top"] if "top" in self.params else 0.25
        bottom = self.params["bottom"] if "bottom" in self.params else 0.25
        weighted = self.params["weighted"] if "weighted" in self.params else False
        ascending_met = self.indicator_cal_method if self.indicator_cal_method != "liquidity" else self.params["method"]
        ascending = self.ascending_df[ascending_met]
        self.op_df = self.cal_hold(op_df, ascending = ascending,
                              top=top, bottom=bottom, weighted=weighted)

        # df = self.data_proxy.get_daily_df(start_date=min(op_df["date"]), end_date= max(op_df["date"]), col_list=["code", "open", "close"])
        # df["date"] = df.index
        # df = df.reset_index(drop = True)
        #
        # df = pd.merge(df, op_df, on =["date", "code"], how = "outer")
        # print(df)
        # print(sys.getsizeof(df))
        # breakpoint()

        # 计算dailypnl
        self.daily_pnl = self.cal_daily_pnl()
        # return_df = self.cal_return(op_code_df=op_code_df)
        # res_df = pd.merge(op_df, return_df, on=["date", "code"], how="inner")
        # res_df["pnl"] = res_df.apply(lambda r: r["return"] * r["holding"], axis =1)
        # res_df = res_df.groupby(["date"])["pnl"].agg(["sum"]).rename(columns = {"sum": "return"})

        # indicator_df = indicator_df.sort_values(["date", "code"]).set_index(["date", "code"])
        # indicator_df = indicator_df.loc[~ indicator_df.index.duplicated(keep ="first")]
        # op_df = op_df.reset_index(drop = True)

        return {self.indicator_cal_method: {"indi": self.indicator_df, "holding": self.op_df, "daily_pnl": self.daily_pnl}}

    # 输入：ascending：False/True, False: 指标高者买入，指标低者卖出
    #       top：买入期货占总期货数比例
    #       bottom： 卖出期货数占总期货数比例
    #       weighted：False/True, False：等权交易，True：加权交易
    def cal_hold(self, df, ascending=False, top=0.25, bottom=0.25, weighted=False):
        def cal_holding(sub_df):
            #等权交易
            if not weighted:
                tmp_df = sub_df.sort_values(by = ['indicator'], ascending=ascending)
                num_futures = sub_df.shape[0]
                n0, n1 = math.ceil(num_futures * top), math.ceil(num_futures * bottom)
                if n0 + n1 > num_futures:
                    i_range = n0 + n1 - num_futures
                    n0, n1 = math.floor(num_futures * top), math.floor(num_futures * bottom)
                    for i in range(i_range):
                        if sub_df.iloc[n0 + i]["indicator"] > 0 and (not ascending):
                            n0 += 1
                        else:
                            n1 += 1
                weight_list_0 = [1 for _ in range(n0)]
                weight_list_1 = [-1 for _ in range(n1)]
                if len(weight_list_0) > 1:
                    weight_list_0[-1] = num_futures * top - math.floor(num_futures * top) \
                        if n0 != math.floor(num_futures * top) else 1
                if len(weight_list_1) > 1:
                    weight_list_1[0] = math.floor(num_futures * bottom) - num_futures * bottom \
                        if n1 != math.floor(num_futures * bottom) else -1
                num_mid = num_futures - len(weight_list_0) - len(weight_list_1)
                weight_mid = []
                if num_mid > 0:
                    weight_mid = [0  for _ in range(num_mid)]
                weight_list = weight_list_0 + weight_mid + weight_list_1
                abs_sum = sum([abs(ele) for ele in weight_list])
                weight_list = [ele / abs_sum for ele in weight_list]
                tmp_df["holding"] = weight_list
                return tmp_df
            # 加权交易
            else:
                # print("加权交易")
                sub_df = sub_df.sort_values('indicator', ascending=not ascending)
                sub_df["holding"] = np.arange(sub_df.shape[0]) + 1
                if sub_df.shape[0] > 1:
                    rank_mean = sub_df["holding"].mean()
                    sub_df["holding"] = sub_df["holding"].map(lambda x: x - rank_mean)
                    rank_sum = sum([abs(i) for i in list(sub_df["holding"])])
                    sub_df["holding"] = sub_df["holding"].map(lambda x: x / rank_sum)
                return sub_df
        df = df.groupby(["date"], as_index = False).apply(cal_holding)
        df = df.drop(columns= ["instrument", "indicator"], axis =1)
        df = df.reset_index(drop = True)
        return df

    def cal_return(self, op_code_df):
        df = self.data_proxy.get_daily_df(col_list=["code", "close"])
        df["date"] = df.index
        df = df.reset_index(drop=True)

        date_df = pd.DataFrame()
        date_df["date"] = op_code_df["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
        date_df["next_date"] = date_df["date"].shift(-1)
        op_code_df = pd.merge(op_code_df, date_df, on = ["date"], how = "inner")
        op_code_df = op_code_df.dropna(axis=0, how = "any")
        op_code_df = pd.merge(op_code_df, df, on = ["code", "date"], how = "inner")
        op_code_df.rename(columns = {"close": "close0", "date": "op_date", "next_date": "date"}, inplace = True)
        op_code_df = pd.merge(op_code_df, df, on = ["code", "date"], how = "inner")
        op_code_df.rename(columns = {"close": "close1"}, inplace = True)
        op_code_df["close0"] = op_code_df["close0"].map(lambda r: np.nan if r == 0 else r)
        op_code_df = op_code_df.dropna(how="any", axis=0)
        op_code_df["return"] = op_code_df.apply(lambda r: r["close1"]/r["close0"] -1, axis =1)
        op_code_df  = op_code_df[[  "op_date", "code", "return"]]

        op_code_df.rename(columns = {"op_date": "date"}, inplace=True)
        # print(op_code_df)
        return  op_code_df


    def cal_daily_pnl(self):
        date_df = pd.DataFrame()
        date_df["date"] = self.op_df["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
        date_df["next_date"] = date_df["date"].shift(-1)
        date_df = date_df.dropna(axis=0, how="any")
        # right： 换仓时间为当天交易日接近结束时，先平仓
        date_df["date_range"] = date_df.apply(
            lambda r: pd.date_range(start=r["date"], end=r["next_date"], closed="right"), axis=1)
        date_index_df = pd.DataFrame({"date": date_df.date.repeat(date_df.date_range.str.len()),
                                      "date_val": np.concatenate(date_df.date_range.values)})
        trade_record = pd.merge(self.op_df, date_index_df, on=["date"])
        date_index_df.rename(columns = {"date": "op_date", "date_val": "date"},  inplace = True)
        trade_record.rename(columns = {"date_val": "date", "date": "op_date"}, inplace = True)

        # 计算 daily value change
        df = self.data_proxy.get_daily_df(col_list=["code", "close"])
        df["date"] = df.index
        df = df.reset_index(drop=True)
        def cal_ratio(sub_df):
            sub_df = sub_df.sort_values(by = ["date"], ascending = True)
            sub_df["close_pre"] = sub_df["close"].shift(1)
            sub_df["close_ratio"] = sub_df.apply(lambda r: r["close"]/r["close_pre"] - 1, axis =1)
            # print(sub_df[-5:])
            return sub_df
        df = df.groupby(["code"], as_index = False).apply(cal_ratio)
        df = df.reset_index(drop  = True)
        # print("-" * 77)
        # print(df[:100])

        trade_record = pd.merge(trade_record, df, on=["date", "code"], how="inner")
        # trade_record["return"] = trade_record.apply(lambda r: (r["close"] - r["open"]) / r["open"], axis=1)
        # print(trade_record[:100])
        # print("-" * 7)
        trade_record["close_ratio"] = trade_record.apply(lambda r: r["close_ratio"] * r["holding"], axis=1)

        # del df, date_df, date_index_df
        daily_pnl = trade_record[[ "date", "close_ratio"]].groupby(["date"], as_index=False).sum()
        daily_pnl.rename(columns = {"close_ratio": "daily_v_change"}, inplace = True)
        daily_pnl = pd.merge(daily_pnl, date_index_df, on = ["date"])

        # print("-" *7)
        # print(daily_pnl[:100])
        daily_pnl["daily_v_ratio"] = daily_pnl["daily_v_change"] + 1

        def cal_pnl(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending = True)
            sub_df["cum_v"] = sub_df["daily_v_ratio"].cumprod()
            sub_df["cum_v_prev"] = sub_df["cum_v"].shift(1)
            sub_df = sub_df.fillna(1)
            sub_df["pnl"] = sub_df.apply(lambda r: r["cum_v"] - r["cum_v_prev"] , axis = 1)
            # print(sub_df)
            return  sub_df

        daily_pnl = daily_pnl.groupby(["op_date"]).apply(cal_pnl)
        # print(daily_pnl)
        # breakpoint()
        daily_pnl = daily_pnl[["date", "daily_v_change", "pnl"]]
        daily_pnl = daily_pnl.reset_index(drop = True)

        # print(daily_pnl)
        daily_pnl = daily_pnl.set_index(["date"])
        return daily_pnl


    """
    # 对于每种策略的月利率轨迹生成统计数值总结
    def generate_statistics_dataframe(self):

        # 计算基本统计数值
        statistics_res_df = []
        for stra, sub_res in self.result.groupby(["strategy"]):
            annualized_arithmetic_mean = sub_res["profit"].mean()
            t_statistics, _ = stats.ttest_1samp(sub_res["profit"], 0)
            skew = sub_res["profit"].skew()
            kurtosis = sub_res["profit"].kurtosis()

            # 计算波动率
            profit_list = list(sub_res["profit"])
            n = len(profit_list)
            s = np.sqrt(
                1 / (n - 1) * sum([ele ** 2 for ele in profit_list]) - (1 / (n * (n - 1)) * sum(profit_list) ** 2))
            volatility = s / ((1 / 12) ** 0.5)

            # sharpe ratio
            sharperatio = (annualized_arithmetic_mean - 0.03)/sub_res["profit"].std()
            # sortino ratio

            res_dic = {"strategy": stra, "annualized arithmetic mean": annualized_arithmetic_mean,
                       "t statistics": t_statistics, "skewness": skew, "kurtosis": kurtosis, "volatility": volatility,
                       "sharperatio": sharperatio}
            statistics_res_df.append(res_dic)

        statistics_res_df = pd.DataFrame(statistics_res_df)
        self.statistics_res_df = statistics_res_df


    def present_performance(self, condition_list):
        # 画净值曲线
        if self.result is None:
            print("请先使用回测run功能")

        statistics_res = self.statistics_res_df
        for condition in  condition_list:
            if "<" in condition:
                cond_alist = condition.split("<")
                var, value = cond_alist[0], cond_alist[1]
                statistics_res = statistics_res[statistics_res[var] < float(value)]
            elif ">" in condition:
                cond_alist = condition.split(">")
                var, value = cond_alist[0], cond_alist[1]
                statistics_res = statistics_res[statistics_res[var] > float(value)]

        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)

        res = pd.merge(self.result, statistics_res, on=["strategy"], how="right")

        for stra, sub_res in res.groupby(["strategy"]):
            sub_res = sub_res.sort_values(by="time", axis=0)
            ax.plot(sub_res["time"], sub_res["currency"], label=stra)

        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("currency")
        plt.show()

        print("基本数值统计结果")
        print(self.statistics_res_df)
    """