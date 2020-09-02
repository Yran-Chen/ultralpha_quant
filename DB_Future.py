# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:58:47 2019

@author: ultralpha
"""
import psycopg2
from DB_Database import Database
from time_process import get_next_month_end
import pandas as pd
import numpy as np
import time
import datetime
import datetime_process as dtp

class FutureDatabase(Database):
    def __init__(self,db_name,host_name,user_name,pwd, port,start_date='2005-01-01',end_date='2019-06-01', set_time_range = True):
        Database.__init__(self,db_name,host_name,user_name,pwd,port)
        print('Future Database init...')
        start = time.clock()
        self.connnect_db()
        self.start_date = start_date
        self.end_date = end_date
        # 加载每日表格
        self.future_df = self.get_future_data(table_name = 'fut_daily', start_date = self.start_date, end_date= self.end_date,
                                              if_includes_date=True,if_set_time_range = set_time_range)
        # 预处理
        self.future_df.rename(columns = {"instrument": "code", "hold": "opi"}, inplace= True)
        self.future_df["instrument"] = self.future_df["code"].map(lambda r:self.extract_instrument_from_code(r))

        # 加载主力合约表格
        self.future_main_df = self.get_future_data(table_name = 'fut_main_day', start_date= self.start_date, end_date= self.end_date,
                                                   if_includes_date=True, if_set_time_range=set_time_range)
        # 预处理instrument, code全部小写
        self.future_df["instrument"] = self.future_df["instrument"].map(lambda s: s.lower())
        self.future_df["code"] = self.future_df["code"].map(lambda s: s.lower())
        self.future_main_df["instrument"] = self.future_main_df["instrument"].map(lambda s: s.lower())
        self.future_main_df["code"] = self.future_main_df["code"].map(lambda s: s.lower())
        # self.future_df.to_csv("future_df.csv")
        # self.future_main_df.to_csv("future_main_df.csv")
        # print("new future df ")
        # print(self.future_df[:100])
        # print(self.future_df.shape[0])

        self.future_basic_df = self.get_future_data(table_name='fut_basic', if_includes_date= False)
        # self.future_basic_df.to_csv("future_basic.csv")
        # self.rmb_exchange_df = self.load_preprocess_rmb_exchange()
        # self.cpi_df = self.load_preprocess_cpi()
        elapsed = (time.clock() - start)

        # self.future_df.to_csv("future_df.csv")
        # print("total rows: ", self.future_df.shape[0])
        # tmp_df = self.future_df[self.future_df["volume"] == 0]
        # print(tmp_df.shape[0])
        # print("*" * 7)
        # tmp_df = self.future_df[self.future_df["amount"] < 1000]
        # # print(tmp_df[:1000])
        # print(tmp_df.shape[0])
        # print("*" * 7)
        # tmp_df = self.future_df[self.future_df["amount"] < 10000]
        # print(tmp_df.shape[0])
        # print("*" * 7)
        # breakpoint()
        print("Time used:", elapsed)

    # date_range_flag: 是否需要根据时间范围来提取数据， 并且将时间设置为index
    def get_future_data(self, table_name, start_date='2005-01-01', end_date='2019-06-01',
                        if_includes_date = True, if_set_time_range = True):
        if (not if_includes_date) or not (if_set_time_range):
            sql = "select * from {}" .format(
                table_name,
            )
        else:
            sql = "select * from {} where " \
                  "date between '{}' and '{}' order by date".format(
                table_name,
                start_date, end_date
            )
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            for col in ["date", "start_date", "end_date"]:
                if col not in df.keys():
                    continue
                df[col] = df[col].map(dtp.date2datetime)
            if if_includes_date:
                df = df.set_index(['date'])
                df = df.sort_index(level='date')
            cursor.close()
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def load_preprocess_rmb_exchange(self):

        rmb_df = pd.read_csv("rmb_exchange.csv")
        rmb_df["date"] = rmb_df["date"].map(dtp.date2datetime)
        return rmb_df


    def load_preprocess_cpi(self):
        cpi_df = pd.read_csv("cpi_monthly.csv")
        cpi_df["date"] = cpi_df["date"].map(lambda x: x.replace("/", "-"))
        cpi_df["date"] = cpi_df["date"].map(dtp.date2datetime)
        return cpi_df


    def get_rmb_exchange_df(self):
        return self.rmb_exchange_df

    def get_basic_df(self):
        return self.future_basic_df

    def get_cpi_df(self):
        return self.cpi_df

    def get_daily_df(self, col_list=None, fut_list=None, start_date=None, end_date=None):
        df = self.future_df.copy()
        if isinstance(fut_list, list):
            pointer = 'instrument in {}'.format(fut_list)
            df = df.query(pointer)
        if isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            df = df[(df.index <= end_date) & (df.index >= start_date)]
        if isinstance(col_list, list):
            df = df[col_list]
        return df

    def get_main_df(self, col_list=None, fut_list=None, start_date=None, end_date=None):
        df = self.future_main_df.copy()
        if isinstance(fut_list, list):
            pointer = 'instrument in {}'.format(fut_list)
            df = df.query(pointer)
        if isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            df = df[(df.index <= end_date) & (df.index >= start_date)]
        if isinstance(col_list, list):
            df = df[col_list]
        return df

    def get_future_index_df(self, col_list=None, start_date = None, end_date = None,):
        table_name = "fut_index"
        data = self.get_future_data(table_name,start_date = self.start_date, end_date = self.end_date)
        if isinstance(col_list, list):
            data = data[col_list]
        return data


    # return_cols: 选择需要返回哪些column
    def get_main_df_per_month_end(self, start_date = None, end_date = None, instrument_list = None,
                                  return_cols = ["instrument", "code", "date_year", "date_month"]):
        df  = self.future_main_df.copy()
        # 可能选择时间范围
        if isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            df = df[ (df.index <= end_date) & (df.index >= start_date)]
        # 可能选择期货池
        if isinstance(instrument_list, list):
            df = df[df.instrument.isin(instrument_list)]
        elif isinstance(instrument_list, pd.DataFrame):
            # todo
            pass

        # 增加年月日属性
        df = self.extend_date_feature(df)
        # todo: 有些期货不存在月末的值
        idx = df.groupby(["date_year", "date_month"])["date_day"].transform(max) == df["date_day"]
        return df[idx][return_cols]

    # 预处理，删除时间段，挑选期货池
    def __preprocess(self, df, start_date=None, end_date=None, future_pool=None):
        # 可能选择时间范围
        if isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date):
            df = df[(df.index <= end_date) & (df.index >= start_date)]

        # 增加年月信心
        df = self.extend_date_feature(df)
        # 增加合约信息，起始日期
        df = self.extend_contract_date_info(df)
        df = df.sort_values(by=["date"], ascending=[True]).reset_index(drop=True)

        # 可能选择期货池, 固定期货池/动态期货池
        if isinstance(future_pool, list):
            df = df[df.instrument.isin(future_pool)]
        elif isinstance(future_pool, pd.DataFrame):
            df = pd.merge(df, future_pool, on=["date", "instrument"], how="inner")
        return df

    # 挑选时间节点上的有效数据，比如每N个工作日，或者每个月底
    # reverse: True/False
    #          如果reverse为True，则结束时间点必然在挑选的频率点中
    #          如果reverse为False，则开始时间点必然在挑选的频率点中
    def __filter_time(self, df, freq_unit, freq, reverse =  False):
        # 挑选每月底的主力期货合约
        if freq_unit == "M":
            # 挑选数据，当月最后一天
            idx = df.groupby(["date_year", "date_month"])["date_day"].transform(max) == df["date_day"]
            df = df[idx].reset_index(drop=True)

            # 挑选数据，选取一月后仍可以执行的数据
            df["next_op_date"] = df.apply(lambda r: get_next_month_end(r["date"]), axis=1)
            df = df[df["next_op_date"] < df["end_date"]]
        # 挑选每N工作日的主力期货合约
        elif freq_unit == "D":
            # 挑选数据， 每N天数据
            if reverse:
                date_list = list(df["date"].drop_duplicates())
                date_list = date_list[::-freq]
                date_list = date_list[::-1]
                date_list = date_list[:-1]
            else:
                date_list = list(df["date"].drop_duplicates())[0::freq]
            # 挑选数据，选取N天后仍可以执行的数据
            time_dic = {}
            for i in range(len(date_list) - 1):
                time_dic[date_list[i]] = date_list[i + 1]
            df = df[df["date"].isin(date_list)].reset_index(drop=True)
            df["next_op_date"] = df["date"].map(lambda d: time_dic[d] if d in time_dic.keys() else d)
            df = df[df["next_op_date"] < df["end_date"]]
            del date_list, time_dic
        return df

    # todo: 简化函数
    # 获得每月底的主力或者次主力合约
    # is_main: True -- 获得主力合约
    # is_main: False -- 获得次主力合约
    # future pool: 期货池： 两种形式: list -- 正对所有时间段， dataframe -- 对于每月，更换期货池
    # freq_unit: M -- 计算周期默认为月，计算每月底的主力/次主力， 若为D -- 计算每间隔 n 天
    # if_must_end_time_point: True/False
    #          如果reverse为True，则结束时间点必然在挑选的频率点中
    #          如果reverse为False，则开始时间点必然在挑选的频率点中
    def get_main_df_with_freq(self, if_main=True, start_date=None, end_date=None, future_pool=None,
                              freq = 10, freq_unit ="M", if_must_end_time_point = False):

        print(start_date, end_date)
        df = self.future_df.copy().drop(columns=["open", "high", "close", "low", "opi", "amount"])

        # 预处理，删除时间段，挑选期货池
        df = self.__preprocess(df, start_date=start_date, end_date=end_date, future_pool=future_pool)
        # print("after preprocess ", df.shape[0])
        # print(df.sample(10))
        print("freq", freq, "date length", len(list(set(df["date"]))))


        # 挑选时间节点上的有效数据，比如月底或者每N个工作日
        # 假设已经经过带有时间节点的future pool了，则无须挑选时间
        if not isinstance(future_pool, pd.DataFrame):
            df = self.__filter_time(df, freq=freq, freq_unit=freq_unit, reverse = if_must_end_time_point)

        # 对于每个是时间节点上所有的合约进行排序，按照volume大小
        def set_volume_index(df):
            df = df.sort_values(axis=0, ascending=False, by=["volume"])
            df["volume_index"] = np.arange(df.shape[0]) + 1
            return df

        index = 1 if if_main else 2
        df = df.groupby(["date", "instrument"], as_index = False).apply(set_volume_index)
        # print("-" * 77)
        # print(start_date, end_date)
        # print(df[:20])
        if df.shape[0] <= 0:
            return None

        df = df[df.volume_index == index].reset_index(drop=True)
        df = df[[ "instrument", "code", "date"]].set_index(["date"])
        return df

    # future pool: 期货池： 两种形式: list -- 正对所有时间段， dataframe -- 对于每月，更换期货池
    # freq_unit: M -- 计算周期默认为月，计算每月底的主力/次主力， 若为D -- 计算每间隔 n 天
    # if_check_vol: 是否检查挑选合约的成交量过小，如果成交量过小，则用主力合约替换
    # if_must_end_time_point: True/False
    #          如果reverse为True，则结束时间点必然在挑选的频率点中
    #          如果reverse为False，则开始时间点必然在挑选的频率点中
    def get_nearest_contract_with_time_freq(self, n_nearest_index, start_date = None, end_date = None,
                                            future_pool = None, freq = 10, freq_unit ="M", if_check_vol = False,
                                            if_must_end_time_point = False):

        df = self.future_df.copy().drop(columns=["open", "high", "close", "low", "opi", "amount"])

        # 预处理，删除时间段，挑选期货池
        df = self.__preprocess(df, start_date=start_date, end_date=end_date, future_pool=future_pool)

        # 挑选时间节点上的有效数据，比如月底或者每N个工作日
        # 假设已经经过带有时间节点的future pool了，则无须挑选时间
        if not isinstance(future_pool, pd.DataFrame):
            df = self.__filter_time(df, freq=freq, freq_unit=freq_unit, reverse = if_must_end_time_point)

        def set_n_nearest_val(df):
            df = df.sort_values(axis=0, ascending=True, by=["end_date"])
            df["n_nearest_index"] = np.arange(df.shape[0]) + 1

            if if_check_vol:
                med_vol = df["volume"].median()
                max_vol = max(df["volume"])
                main_code =  df.loc[df["volume"] == max(df["volume"]), "code"].values[0]
                df.loc[df["volume"] < med_vol * 0.5, ["code", "volume"]] = [main_code, max_vol]
                df.loc[df["volume"] < 1000, ["code", "volume"]] = [main_code, max_vol]
                # print(df)
                # print("!" * 7)
                # if df[df["volume"] < 10000].shape[0] > 0:
                #     print("special situation")
                #     print(df)
            return df

        df = df.groupby(["date", "instrument"]).apply(set_n_nearest_val)
        df = df[df.n_nearest_index == n_nearest_index].reset_index(drop = True)
        df = df[["instrument", "code", "date"]].set_index(["date"])
        # breakpoint()
        return df

    def get_month_end_values(self, col_list,  start_date = None, end_date = None, instrument_list = None):

        df = self.future_df.copy()

        # 可能选择时间范围
        if isinstance(start_date, datetime.datetime) and isinstance(end_date, datetime.datetime):
            df = df[(df.index <= end_date) & (df.index >= start_date)]
        # 可能选择期货池
        if isinstance(instrument_list, list):
            df = df[df.instrument.isin(instrument_list)]

        df = self.extend_date_feature(df)

        # 挑选数据，当月最后一天
        idx = df.groupby(["date_year", "date_month"])["date_day"].transform(max) == df["date_day"]
        df = df[idx].reset_index(drop=True)

        return df [["date", "date_year", "date_month", "instrument", "code"] + col_list]

    def get_daily_df_by_date_cols(self, col_list,  start_date = None, end_date = None):

        df = self.future_df.copy()

        # 可能选择时间范围
        if isinstance(start_date, datetime.date) and isinstance(end_date, datetime.date):
            df = df[(df.index <= end_date) & (df.index >= start_date)]

        df = self.extend_date_feature(df)
        df = self.extend_contract_date_info(df)

        return df[["date", "date_year", "date_month", "date_day", "instrument", "code"] + col_list]

    # 增加合约的起止时间
    def extend_contract_date_info(self, df):
        df = pd.merge(df, self.future_basic_df, on=["code"], how = "left", left_index=True)
        df = df[(df["date"] >= df["start_date"]) & (df["date"] <= df["end_date"])]
        return df

    def extend_date_feature(self, df):
        df["date"] = df.index
        df["date_year"] = df["date"].map(lambda date: date.year)
        df["date_month"] = df["date"].map(lambda date: date.month)
        df["date_day"] = df["date"].map(lambda date: date.day)
        return df

    # 从期货合约中解析出期货商品
    def extract_instrument_from_code(self, code):
        if code[1].isalpha():
            return code[:2]
        else:
            return code[:1]


if __name__ == "__main__":

    db_name = 'ultralpha_db'
    host_name = '192.168.0.116'
    user = 'cyr'
    password = 'cyr'
    port = '5432'
    fut_name = 'al'
    start_date = '2010-04-01'
    end_date = '2014-04-01'

    database = FutureDatabase(db_name = db_name,host_name = host_name, user_name = user, pwd = password,
                              port = port )

    # print(database.get_main_per_month_end(freq= 7, freq_unit="D"))
    df = database.get_nearest_contract_with_time_freq(n_nearest_index=2, freq=7, freq_unit="D")

    # print(df.sample(200))
    # print(df[df["volume"] < 200].shape[0])

    # import matplotlib.pyplot as plt
    # alist = df["volume"]
    # plt.hist()
    # plt.show()
