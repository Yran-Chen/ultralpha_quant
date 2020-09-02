# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import time
import datetime
from scipy.stats import linregress
from dateutil.relativedelta import relativedelta

class IndicatorGenerator:

    def __init__(self, future_db):
        self.future_db = future_db

    def cal_indicator(self, params):
        # 提取具体的用于测量指标的方法名
        if "method" not in params:
            raise ValueError("请指明策略方式，如momentum, hedging_pressure等")
        method_name = params["method"]

        # 检查基本参数是否存在
        for var in [ "future_pool", "start_date", "end_date", "is_main_contract_indi"]:
            if var not in params:
                raise ValueError("请检查传递参数，基本传递参数不全")

        if method_name == "term_structure":
            return self.__term_structure(param_dic=params)
        elif method_name == "term_structure2":
            return self.__term_structure_2(param_dic=params)
        elif method_name == "hedging_pressure":
            return self.__hedging_pressure(param_dic = params)
        elif method_name == "momentum":
            return self.__momentum(param_dic=params)
        # todo: check
        elif method_name == "volatility":
            return self.__volatility(param_dic=params)
        elif method_name == "open_interest":
            return self.__open_interest(param_dic=params)
        elif method_name == "liquidity":
            return self.__liquidity(param_dic=params)
        elif method_name == "currency":
            return self.__currency(param_dic=params)
        elif method_name == "inflation":
            return self.__inflation(param_dic=params)
        elif method_name == "skewness":
            return self.__skewness(param_dic=params)
        elif method_name == "value":
            return self.__value(param_dic=params)
        else:
            raise RuntimeError("请选择正确的指标计算策略")

    # 传入计算因子所需要的参数，从字典中解析出来，返回变量列表
    # 输入：param_dic：参数字典
    #       param_name_list： 参数变量名
    def __parse_param(self, param_dic, param_name_list):
        var_list = []
        for name in param_name_list:
            try:
                var_list.append(param_dic[name])
            except:
                raise RuntimeError("缺少参数 {}".format(name))
        return var_list

    def __get_indi_code_df(self, is_main_contract_indi, start_date, end_date, future_pool,
                        n_nearest_index_indi, freq, freq_unit, num_days = None, num_durations = None, num_years = None):
        print("freq", freq)
        # 获取 起始时间段的换仓时间点（亦即计算因子时间点）以及其对应的期货合约
        cur_code = self.__get_indicator_code_df(if_main_df=is_main_contract_indi, if_second=False,
                                                start_date=start_date, end_date=end_date, future_pool=future_pool,
                                                n_nearest_index_indi=n_nearest_index_indi,
                                                freq=freq, freq_unit=freq_unit, if_must_end_time_point=False)

        # 若存在回溯时间， 获取 回溯时间段内的计算因子时间点 以及其对应的期货合约
        off_day = 0
        if num_years is not None and num_days is not None:
            off_day =  relativedelta(years=num_years + math.ceil(num_days / (2 * 365)))
        elif num_days is not None:
            off_day = datetime.timedelta(days=num_days)
        elif num_durations is not None:
            off_day = num_durations * freq if freq_unit == "D" else num_durations * 31
            off_day = datetime.timedelta(days=off_day)

        if off_day != 0:
            # 对于每个时间节点，每种期货商品应该参考不同合约来计算指标
            # todo: how to find the previous working day ? not roughly
            # print(type(start_date))
            if isinstance(off_day, np.int8) or isinstance(off_day, np.int16) or isinstance(off_day, np.int32):
                off_day = off_day.item()

            pre_code = self.__get_indicator_code_df(if_main_df=is_main_contract_indi, if_second=False,
                                                       start_date=start_date - off_day * 2.5,
                                                       end_date=min(cur_code["date"]), future_pool=future_pool,
                                                       n_nearest_index_indi=n_nearest_index_indi,
                                                       freq=freq, freq_unit=freq_unit, if_must_end_time_point=True)
            cur_code = pd.concat([pre_code, cur_code], axis=0)
        return cur_code


    def __get_indicator_code_df(self, if_main_df, if_second, start_date, end_date, future_pool,
                                n_nearest_index_indi, freq, freq_unit, if_must_end_time_point = False):
        if if_main_df:
            if not if_second:
                indicator_code_df = self.future_db.get_main_df_with_freq(if_main=True, start_date=start_date,
                                                                         end_date=end_date, future_pool=future_pool,
                                                                         freq=freq, freq_unit=freq_unit,
                                                                         if_must_end_time_point=if_must_end_time_point)
            else:
                indicator_code_df = self.future_db.get_main_df_with_freq(if_main=False, start_date=start_date,
                                                                         end_date=end_date, future_pool=future_pool,
                                                                         freq=freq, freq_unit=freq_unit,
                                                                         if_must_end_time_point=if_must_end_time_point)
        else:
            if not if_second:
                indicator_code_df = self.future_db.get_nearest_contract_with_time_freq(
                    n_nearest_index=n_nearest_index_indi, start_date=start_date, end_date=end_date,
                    future_pool=future_pool, freq=freq, freq_unit=freq_unit, if_must_end_time_point=if_must_end_time_point)
            else:
                indicator_code_df = self.future_db.get_nearest_contract_with_time_freq(
                    n_nearest_index=n_nearest_index_indi + 1, start_date=start_date, end_date=end_date,
                    future_pool=future_pool, freq=freq, freq_unit=freq_unit, if_must_end_time_point=if_must_end_time_point)

        if indicator_code_df is not None:
            indicator_code_df["date"] = indicator_code_df.index
            indicator_code_df = indicator_code_df.reset_index( drop = True )
        return indicator_code_df

    # 输入：code_df: 一定的带有频次的时间节点
    #       set_date_range_index: 如果为True，所有同一时间段的会被安排同序号
    # 输出：每个时间点期货品种应该选择的期货合约
    def __generate_date_code_df(self, code_df, set_date_range_index = False):
        date_df = pd.DataFrame()
        date_df["date"] = code_df["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
        date_df["last_date"] = date_df["date"].shift(1)
        date_df = date_df.dropna(axis=0, how="any")
        date_df["date_range"] = date_df.apply(lambda r: pd.date_range(start=r["last_date"], end=r["date"], closed="right"), axis=1)

        date_index_df = pd.DataFrame({"date": date_df.date.repeat(date_df.date_range.str.len()),
                                      "date_val": np.concatenate(date_df.date_range.values)})
        if set_date_range_index:
            date_list = date_df["date"].tolist()
            alist = [i for i in range(len(date_list))]
            time_dic = {ele: i for (ele, i) in zip(date_list, alist)}
            date_index_df["date_range_flag"] = date_index_df["date"].apply(lambda r: time_dic[r])

        date_code_df = pd.merge(date_index_df, code_df, on=["date"], how="left").drop(columns={"date"}, axis=1).rename(columns={"date_val": "date"})
        return date_code_df

    # 输入：date_code_df: 一定的带有频次的时间节点，以及其期货合约
    #       keep_code：输出是否存在code column
    #       use_log: 是否使用log计算return
    # 输出：对于每个时间节点，该合约产生多少return
    def __cal_duration_return(self, code_df, keep_code = False, use_log = True):
        # 产生对于每天应该选取哪种合约来计算
        date_code_df = self.__generate_date_code_df(code_df, set_date_range_index=True)

        # 选取所有日常数据，包括基础信息加 close 列
        df = self.future_db.get_daily_df(col_list=["code", "close"])
        df["date"] = df.index
        df = df.reset_index(drop = True)
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner")

        def cal_return(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            begin_val = sub_df.iloc[0]["close"]
            end_val = sub_df.iloc[-1]["close"]
            # print(begin_val, end_val)
            if use_log:
                sub_df["return"] = math.log10(end_val / begin_val) if (begin_val != 0) and (end_val != 0) else np.nan
            else:
                sub_df["return"] = end_val / begin_val if begin_val != 0 else np.nan
            return sub_df

        df = df.groupby(["instrument", "date_range_flag"], as_index=False).apply(cal_return)

        # 挑选出需要计算交易时间节点的数据, 并附带它的return
        df = pd.merge(code_df, df, on=["date", "instrument", "code"], how="inner")
        if not keep_code:
            df = df.drop(columns=["close", "code"], axis=1)
        else:
            df = df.drop(columns=["close"], axis=1)
        return df

    # Roll(i, t) = LN( F(it, front) - LN (F (it, 2))
    # (1) 主力合同与次主力合同
    # (2) n nearest 合同与(n+1) nearest 合同
    # (3) 当月合同与次当月合同 n_nearest = 0
    # is_main_contract 如果为True，代表主力合同与次主力合同，n_nearest_index无效
    # is_main_contract 如果为False，则应该传入n nearest 合同
    def __term_structure(self, param_dic):
        print("Calculate term structure")
        t0 = time.time()
        # 解析参数    设置普通参数，必须传输
        param_name_list =  ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param( param_dic, param_name_list)

        # 设置非必须参数，若无则使用默认参数，默认为近期合约
        n_nearest_index_indi = None
        if not is_main_contract_indi:
            n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1

        # 获得用于计算因子的合约
        # cur_instrument_code 获得主力合同或者当月合同
        # next_instrument_code 获得次主力合同或者次月合同
        cur_code = self.__get_indicator_code_df(if_main_df=is_main_contract_indi, if_second=False,
                                                           start_date=start_date, end_date=end_date,
                                                           future_pool=future_pool,
                                                           n_nearest_index_indi=n_nearest_index_indi,
                                                           freq=freq, freq_unit=freq_unit)

        next_code = self.__get_indicator_code_df(if_main_df=is_main_contract_indi, if_second=True,
                                                            start_date=start_date, end_date=end_date,
                                                            future_pool=future_pool,
                                                            n_nearest_index_indi=n_nearest_index_indi,
                                                            freq=freq, freq_unit=freq_unit)

        price_df = self.future_db.get_daily_df(col_list=[ "code", "close"])
        price_df["date"] = price_df.index
        price_df = price_df.reset_index(drop=True)
        cur_code = pd.merge(cur_code, price_df, on=["date", "code"], how="left")
        next_code = pd.merge(next_code, price_df, on=["date", "code"], how="left")

        # 从 basic df获得合约的到期时间
        basic_df = self.future_db.get_basic_df()
        cur_code = pd.merge(cur_code, basic_df, on=["code"], how="left", left_index=True)
        cur_code = cur_code[(cur_code["date"] >= cur_code["start_date"]) & (cur_code["date"] <= cur_code["end_date"])]
        next_code = pd.merge(next_code, basic_df, on=["code"], how="left", left_index=True)
        next_code= next_code[(next_code["date"] >=next_code["start_date"]) & (next_code["date"] <= next_code["end_date"])]

        cur_code.rename(columns = {"close": "close_1", "code": "code1", "end_date": "end_date_1"}, inplace=True)
        next_code.rename(columns = {"close": "close_2", "code": "code2", "end_date": "end_date_2"}, inplace=True)

        def cal_month_diff(r):
            return round((r["end_date_2"] - r["end_date_1"]).days/30)

        indicator_df  = pd.merge(cur_code, next_code, on=["date", "instrument"], how="inner")
        indicator_df["month_diff"] = indicator_df.apply(lambda r: cal_month_diff(r), axis=1)
        indicator_df["close_1"] = indicator_df["close_1"].map(lambda r: np.nan if r == 0 else r)
        indicator_df["close_2"] = indicator_df["close_2"].map(lambda r: np.nan if r == 0 else r)
        indicator_df["indicator"] = indicator_df.apply(lambda r: (math.log(r["close_1"], math.e) - math.log(r["close_2"], math.e))/r["month_diff"] if r["month_diff"] != 0 else np.nan, axis = 1)
        indicator_df = indicator_df.dropna(axis = 0, how ="any")
        indicator_df = indicator_df[["date", "instrument", "indicator"]]

        print("Time for calculating term structure ", time.time() - t0)
        return indicator_df


    # term structure 方式二： 依照当前存在合约的线性回归的slope来计算期货商品
    def __term_structure_2(self, param_dic):
        print("Calculate term structure 2")
        t0 = time.time()

        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)

        # 设置非必须参数，若无则使用默认参数，默认为近期合约
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1
        op_code_df = param_dic["op_code_df"] if "op_code_df" in param_dic.keys() else None

        # cur_code = self.__get_indicator_code_df(if_main_df=is_main_contract_indi, if_second=False,
        #                                         start_date=start_date, end_date=end_date, future_pool=future_pool,
        #                                         n_nearest_index_indi=n_nearest_index_indi,
        #                                         freq=freq, freq_unit=freq_unit)

        # 选取日常数据
        df = self.future_db.get_daily_df(col_list=["instrument", "code", "close"])
        df["date"] = df.index

        basic_df = self.future_db.get_basic_df()
        df = pd.merge(df, basic_df, on=["code"], how="left", left_index=True)
        df["left_m"] = df.apply(lambda r: (r["end_date"] - r["date"]).days / 30, axis=1)

        def cal_ts(r):
            # print (r)
            tmp_df = df[(df["instrument"] == r["instrument"]) & (df["date"] == r["date"])]
            tmp_df = tmp_df.sort_values(by=["date"], ascending=True)
            slope = linregress(tmp_df["left_m"].values, tmp_df["close"].values).slope
            return -slope


        op_code_df["indicator"] = op_code_df.apply(cal_ts, axis =1)
        # indi_df = op_code_df[["instrument", "date", "indicator"]]
        #
        # if op_code_df is None:
        #     return indi_df
        #
        # indi_df = pd.merge(indi_df, op_code_df, on=["date", "instrument"], how="right")
        op_code_df = op_code_df.dropna(how="any", axis=0)
        op_code_df = self.post_process(op_code_df, ascending=False)
        print("Time for calculating term structure 2", time.time() - t0)
        return op_code_df

    # TODO: 存在期货，每月只有部分有效值，只截取该段时间的起始点计算， 存在open interest 或者 volume sum 为0的情况
    # todo: 存在na情况
    # todo: result problem
    # ratio(i,t) = abs(delta OI (i,t)) / vol(i, t)
    # (1) 主力合同
    # (2) n nearest 合同 （既需要选择的那一份）
    def __hedging_pressure(self, param_dic):
        print("Calculate hedging pressure")
        t0 = time.time()
        param_name_list = [ "future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)

        # 设置非必须参数，若无则使用默认参数，默认为近期合约
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1
        # look back period
        num_days = int(param_dic["num_days"]) if "num_days" in param_dic else 30
        # 获得因子计算时间点，以及其对应的合约
        # print(type(start_date))
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                          n_nearest_index_indi, freq, freq_unit, num_days=num_days)

        # 产生对于每天应该选取哪种合约来计算
        date_code_df = self.__generate_date_code_df(cur_code)

        # 选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", "opi", "volume"])
        df["date"] = df.index
        df = df.reset_index(drop=True)
        df = pd.merge(date_code_df, df, on=["date", "code"], how = "inner")

        def cal_hedge_presure(sub_df):
            # print(sub_df)
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["begin_opi"] = sub_df["opi"].shift(num_days)
            sub_df["vol_sum"] = sub_df["volume"].rolling(num_days, min_periods=1).sum()
            sub_df["indicator"] = sub_df.apply(lambda r:  (r["opi"] - r["begin_opi"]) /r["vol_sum"] if r["vol_sum"] != 0 else np.nan, axis=1)
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_hedge_presure).reset_index(drop=True)
        df = df.dropna(how="any", axis=0)
        df = pd.merge(cur_code, df, on=["date", "instrument", "code"], how="inner")
        df = df.drop(columns = ["opi", "volume", "begin_opi", "vol_sum", "code"])

        print("Time for calculating hedging pressure", time.time() - t0)
        return df


    # todo: extend pre time period
    # mom = (1/12)sum(r i,t-j)  log return of commodity i in the month t-j
    # num_month: variable
    def __momentum(self,param_dic):

        print("Calculate momentum")
        t0 = time.time()
        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param( param_dic, param_name_list )

        # 设置非必须参数，若无则使用默认参数，默认为近期合约
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1
        num_durations = int(param_dic["num_durations"]) if "num_durations" in param_dic else 12

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_durations=num_durations)

        # 得到每个时间节点之间时间段的 return 值
        df = self.__cal_duration_return(code_df=cur_code)

        def cal_momentum(df):
            df = df.sort_values(by = ["date_range_flag"], ascending= True)
            df["indicator"] = df["return"].rolling(window=num_durations).mean()
            return df

        df = df.groupby(["instrument"], as_index = False ).apply(cal_momentum)
        df = df.drop(columns = ["date_range_flag", "return"], axis =1)
        df = df.dropna(how="any", axis = 0)

        print("Time for calculating momentum", time.time() - t0)
        return df


    # TODO: daily return
    # volatility(i,t) = past 36 month: vraience of daily return and absolute of daily return
    # (1) 每个月的主力合同 或者 n nearest 合同 (n = 0, 1, 2 ...)
    # is_main_contract 为True，寻找过去每月的主力合同
    # is_main_contract 为False，寻找 nearest contract
    def __volatility(self, param_dic):

        t0 = time.time()
        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)
        # 基于过去num_days的情况进行统计
        num_days = param_dic["num_days"] if "num_days" in param_dic else 200
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_days=num_days)

        date_code_df = self.__generate_date_code_df(cur_code)

        #  选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", "close", "open"])
        df["date"] = df.index
        df = df.reset_index(drop=True)
        df["return"] = df.apply(lambda r: r["close"] / r["open"] if r["open"] != 0 else np.nan, axis=1)

        # 对于每个时间节点，每种期货选取用于计算因子的合约
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner").drop(["close", "open"], axis=1)
        df = df.dropna(axis=0, how="any")

        def cal_vola(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["return_mean"] = sub_df["return"].rolling(num_days, min_periods=1).mean()
            sub_df["return_std"] = sub_df["return"].rolling(num_days, min_periods=1).std()
            sub_df["indicator"] = sub_df.apply(lambda r: r["return_std"]/abs(r["return_mean"]) if r["return_mean"] != 0 else np.nan, axis =1)
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_vola).reset_index(drop=True)
        df = df.dropna(axis=0, how="any")
        df = pd.merge(cur_code, df, on=["date", "instrument", "code"], how="inner")
        df = df.drop(columns=["code", "return_mean", "return_std", "return"])

        print("Time for calculating volatility ", time.time() - t0)
        return df


    # TODO: time dleta: 1 month or 1 day ?
    # TODO: 存在positions总和为0的情况， 很多 !!!
    # delta open interest(i,t) = OI (i, t) - OI (i, t-1)
    # (1) 主力合同 (2) 当月合同  (3) n nearest合同
    def __open_interest(self, param_dic):
        print("Calculating open interest")
        t0 = time.time()

        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)
        # 设置非必须参数，若无则使用默认参数，默认为近期合约
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1
        num_days = int(param_dic["num_days"]) if "num_days" in param_dic else 30

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_days=max(num_days, freq))


        # 产生对于每天应该选取哪种合约来计算
        date_code_df = self.__generate_date_code_df(cur_code)

        # 选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", "opi"])
        df["date"] = df.index
        df = df.reset_index(drop = True)
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner")

        # def cal_avg_opi(sub_df):
        #     sub_df["avg_opi"] = sub_df["opi"].mean()
        #     return sub_df
        # df = df.groupby(["instrument", "date_range_flag"], as_index=False).apply(cal_avg_opi)

        # # 挑选出需要计算交易时间节点的数据
        # df = pd.merge(cur_code, df, on=["date", "instrument", "code"], how="inner")
        # df = df.drop(columns=["opi", "code"], axis=1)
        def cal_oi(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["last_opi"] = sub_df["opi"].shift(num_days)
            sub_df["indicator"] = sub_df.apply(lambda r: r["opi"] - r["last_opi"], axis =1)
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_oi)

        df = pd.merge(cur_code, df, on = ["date", "instrument", "code"], how = "left")
        df = df.drop(columns=["opi", "last_opi", "code"], axis=1)
        df = df.dropna(how="any", axis=0)

        print("Time for calculating open interest", time.time() - t0)
        return df


    # TODO: daily return 概念? 目前的做法都存在可能为0的情况
    # TODO:可能改动，计算流动性的时间区间
    # TODO:存在大量为0的情况
    # liquidity: past two months: avg(daily RMB volume/ daily return)
    # (1) 主力合同 (2) 当月合同 (3) n nearest合同
    def __liquidity(self, param_dic):
        t0  = time.time()
        print("Calculating liquidity")
        param_name_list = [ "future_pool", "start_date", "end_date", "is_main_contract_indi",  "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)

        # 基于过去num_month的情况进行统计
        num_days = param_dic["num_days"] if "num_days" in param_dic else 60
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1
        method = param_dic["liq_method"] if "liq_method" in param_dic else "amivest"

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_days=num_days)

        date_code_df = self.__generate_date_code_df(cur_code)

        #  选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", "close", "open", "amount"])
        df["date"]  = df.index
        df = df.reset_index(drop=True)

        if method == "amivest":
            # amivest: indi higher -- more liq
            df["return"] = df.apply(lambda r: r["close"] / r["open"] if r["open"] != 0 else np.nan, axis=1)
            df["liq"] = df.apply(lambda r: r["amount"]/abs(r["return"]) if r["return"] != 0 else np.nan, axis =1)
            # ascending = True
        elif method == "amihud":
            # amihud: indi higher -- less liq
            df["return"] = df.apply(lambda r: abs(r["close"] / r["open"] - 1) if r["open"] != 0 else np.nan, axis=1)
            df["liq"] = df.apply(lambda r: r["return"] / r["amount"] if r["amount"] != 0 else np.nan, axis=1)
            # ascending = False

        # 对于每个时间节点，每种期货选取用于计算因子的合约
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner"). drop(["close", "open", "return"], axis=1)
        df = df.dropna(axis=0, how="any")

        def cal_liq(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["indicator"] = sub_df["liq"].rolling(num_days, min_periods=1).mean()
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_liq).reset_index(drop=True)
        df = pd.merge(cur_code, df, on=["date", "instrument", "code"], how="inner")

        df = df.drop(columns=["liq", "code", "amount"])
        print("Time for calculating liquidity", time.time() - t0)
        return df



    #   输入：code df: 计算时间节点以及其计算的期货合约
    #         index_df: 指数表， 比如人名币汇率或者通货膨胀
    #   输出： 所需时间节点 以及其对应的最新指数
    def __cal_date_index_df(self, code_df, index_df):
        date_df = pd.DataFrame()
        date_df["date"] = code_df["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
        # print(date_df.dtypes)
        # print(index_df.dtypes)
        # print(date_df[:10])
        # print(index_df[:10])

        def cal_index_at_time_point(r):
            sub_df = index_df[index_df["date"] <= r["date"]]
            if sub_df.shape[0] == 0:
                return np.nan
            return (sub_df[sub_df["date"] == max(sub_df["date"])]["index"].values[0])

        date_df["index"] = date_df.apply(cal_index_at_time_point, axis=1)

        return date_df

    def __cal_beta_indi(self, cur_code, index_df, num_durations):

        # 得到每个时间节点之间时间段的 return 值
        df = self.__cal_duration_return(code_df=cur_code)
        # print("duration return ")
        # print(df[:100])
        date_index_df = self.__cal_date_index_df(code_df=cur_code, index_df=index_df)
        df = pd.merge(df, date_index_df, on=["date"], how="left")

        def cal_indi(sub_df):
            sub_df = sub_df.sort_values(by=["date_range_flag"], ascending=True)
            # todo: how to extract the cov ?
            sub_df["cov"] = sub_df[["return", "index"]].rolling(num_durations).cov().values[::2, 1]
            sub_df["var"] = sub_df["index"].rolling(num_durations).var()
            sub_df["indicator"] = sub_df.apply(lambda r: r["cov"] / r["var"], axis=1)
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_indi)
        df = df.drop(columns=["return", "index", "date_range_flag", "cov", "var"], axis=1)
        df = df.dropna(axis=0, how="any")
        return df


    # TODO: 是否加入参数控制时间长度
    # 可能性：对于42个月的monthly return 计算采用主力合约还是 nearest 合约
    def __currency(self, param_dic):
        print("Calculating currency ")
        t0 = time.time()

        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1

        # 基于过去num_month的情况进行统计
        num_durations = param_dic["num_durations"] if "num_durations" in param_dic else 42

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_durations=num_durations)

        index_df = self.future_db.get_rmb_exchange_df()

        df = self.__cal_beta_indi(cur_code= cur_code, index_df=index_df, num_durations=num_durations)
        print("indi df")
        print(df[:10])
        print("Time for calculating  currency", time.time() - t0)

        return df



    def __inflation(self, param_dic):
        print("Calculating inflation")
        t0 = time.time()

        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1

        # 基于过去num_month的情况进行统计
        num_durations = int(param_dic["num_durations"]) if "num_durations" in param_dic else 42
        index_col = param_dic["index_col"] if "index_col" in param_dic else "y2y_0"

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_durations=num_durations)

        index_df = self.future_db.get_cpi_df()
        index_df = index_df[["date", index_col]]
        index_df.rename(columns = {index_col: "index"}, inplace = True)
        df = self.__cal_beta_indi(cur_code=cur_code, index_df=index_df, num_durations=num_durations)
        print("Time for calculating  inflation", time.time() - t0)
        return df



    # 过去 12 个月，每个月的 ...
    # (1) 主力合同
    # (2) nearest合同
    def __skewness(self, param_dic):
        print("Calculating skewness")
        t0 = time.time()
        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)

        # 基于过去num_month的情况进行统计
        num_days = param_dic["num_days"] if "num_days" in param_dic else 252
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else 1

        # 获得因子计算时间点，以及其对应的合约
        cur_code = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_days = num_days)

        date_code_df = self.__generate_date_code_df(cur_code)

        #  选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", "close", "open"])
        df["date"] = df.index
        df = df.reset_index(drop=True)
        df["return"] = df.apply(lambda r: r["close"] / r["open"] if r["open"] != 0 else np.nan, axis=1)

        # 对于每个时间节点，每种期货选取用于计算因子的合约
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner").drop(["close", "open"], axis=1)
        df = df.dropna(axis=0, how="any")

        def cal_skew(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["indicator"] =  sub_df["return"].rolling(num_days).skew()
            return sub_df

        df = df.groupby(["instrument"], as_index=False).apply(cal_skew).reset_index(drop=True)
        df = df.dropna(axis=0, how="any")
        df = pd.merge(cur_code, df, on=["date", "instrument", "code"], how="inner")
        df = df.drop(columns=["code", "return"])

        print("Time for calculating  skewness", time.time() - t0)
        return df


    # 4.5 years ago - 3.5 years ago
    # 一共12个月，每个月的主力合同，当月合同或者n nearest合同
    # 目前时刻的价格如何选择，主力合约，nearest 合约或者是平均价格？
    def __value(self, param_dic):
        print("Calculating value")
        t0 = time.time()

        param_name_list = ["future_pool", "start_date", "end_date", "is_main_contract_indi", "freq", "freq_unit"]
        future_pool, start_date, end_date, is_main_contract_indi, freq, freq_unit = self.__parse_param(param_dic, param_name_list)

        # 基于过去num_month的情况进行统计
        num_days = param_dic["num_days"] if "num_days" in param_dic else 252
        num_years = param_dic["num_years"] if "num_years" in param_dic else 5
        n_nearest_index_indi = param_dic["n_nearest_index_indi"] if "n_nearest_index_indi" in param_dic else None
        price_col = "close"

        # 获得因子计算时间点，以及其对应的合约
        code_df = self.__get_indi_code_df(is_main_contract_indi, start_date, end_date, future_pool,
                                           n_nearest_index_indi, freq, freq_unit, num_days=num_days, num_years = num_years)

        date_code_df = self.__generate_date_code_df(code_df)

        #  选取所有日常数据，包括基础信息加 opi volume 两列
        df = self.future_db.get_daily_df(col_list=["code", price_col])
        df["date"] = df.index
        df = df.reset_index(drop=True)

        # 对于每个时间节点，每种期货选取用于计算因子的合约
        df = pd.merge(date_code_df, df, on=["date", "code"], how="inner")
        df = df.dropna(axis=0, how="any")

        def cal_avg_price(sub_df):
            sub_df = sub_df.sort_values(by=["date"], ascending=True)
            sub_df["avg_price"] = sub_df[price_col].rolling(num_days, center=True).mean()
            return sub_df

        # 每天的当年 avg price
        df = df.groupby(["instrument"], as_index = False).apply(cal_avg_price)

        # 选取当天的该份指标合约的价格
        code_df = pd.merge(code_df, df, on=["date", "instrument", "code"])
        code_df = code_df.drop( columns=["avg_price"], axis =1)

        code_df["n_year_ago"] = code_df["date"].map(lambda d: d-relativedelta(years=num_years))
        # 获取 n 年前该期货商品的平均价格
        df.rename(columns={"date": "n_year_ago"}, inplace=True)
        df = df.drop(columns =["code", price_col], axis =1)
        code_df = pd.merge(code_df, df,  on=["n_year_ago", "instrument"], how="inner")
        code_df["indicator"] = code_df.apply(lambda r: r["avg_price"]/r[price_col], axis=1)
        code_df = code_df.drop(columns=["n_year_ago", "avg_price", price_col, "code"], axis =1)
        code_df = code_df.dropna(how="any", axis=0)
        code_df = code_df[code_df["date"] >= start_date]

        print("Time for calculating value ", time.time() - t0)
        return code_df