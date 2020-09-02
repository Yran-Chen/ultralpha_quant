# -*- coding: utf-8 -*-
import math
import pandas as pd
from dateutil.parser import parse
import numpy as np
from DB_Future import FutureDatabase
from fastbacktest import FastBacktest
#from time_process import get_next_month
import matplotlib.pyplot as plt
import datetime
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)


class TestMomBacktest(FastBacktest):
    def __init__(self, params, data_proxy=None):
        FastBacktest.__init__(self, params=params, data_proxy=data_proxy)
        self.params = params

        # # 确定回测的期货池
        # try:
        #     self.future_pool = params["future_pool"]
        #     params.pop("future_pool")
        # except:
        #     raise RuntimeError("请在输入参数中传递期货池")
        #
        # # 创造期货数据类
        # # 用户登录帐号等设置
        # db_name = 'ultralpha_db'
        # host_name = '192.168.0.116'
        # user = 'cyr'
        # password = 'cyr'
        # port = '5432'
        # if data_proxy is None:
        #     self.data_proxy = FutureDatabase(db_name=db_name, host_name=host_name, user_name=user, pwd=password, port=port)
        #     # todo: if necessary to add this function
        # else:
        #     self.data_proxy = data_proxy
        # self.data_proxy.set_pool_range(self.future_pool)
        #确定回测的时间段
        # try:
        #     self.start_date, self.end_date = params["start_date"], params["end_date"]
        # except:
        #     raise RuntimeError("请确定回测的起始时间")
        
        #确定时间序列类参数
        self.Lookback_period = params["Lookback_period"]
        self.holding_period = params["holding_period"]
        self.delete = params["delete"]
        
    
    def run(self):
        return self.TSMOM_Strategy(self.future_pool,
                                   self.start_date, 
                                   self.end_date,
                                   self.Lookback_period,
                                   self.holding_period,
                                   self.delete)

    
    #Time Series Momentem Strategy:
    def TSMOM_Strategy(self,future_pool=["cu","al","au","rb","ru","zn","a","c","j","l","m","p","y","cf","fg","oi","sr","ta","wh"],
                       start_date="2017-09-01",end_date="2019-09-01",
                       Lookback_period=12,holding_period="M",delete=False):
                
        def fun(x):
            if len(x)>0:
                return (x[-1]-x[0])/x[0]
            else:
                return np.nan
            
        def fun2(x):
            if len(x)>0:
                return x[0]
            else:
                return np.nan    
        
        start_date = start_date.strftime('%Y-%m-%d')
        
        #提前300天以便于计算第一天的事前波动率
        start_date = (parse(start_date) + datetime.timedelta(days=-305)).strftime("%Y-%m-%d")
        
        end_date = end_date.strftime('%Y-%m-%d')
        
        data_basic = self.data_proxy.get_future_index_df(col_list=["close"], start_date = start_date, end_date = end_date)
        
        #这里改了DB_database那里，不然start_date,end_date提取不出来
        data_asset = self.data_proxy.get_main_df(col_list=["instrument","close"],fut_list=future_pool, start_date = start_date, end_date = end_date)
        
        data_asset = data_asset[start_date:end_date]
        
        def calculate_ret(j):
            asset = future_pool[j]
            data_asset2 = data_asset[data_asset["instrument"] == future_pool[j]]
            
            data_basic_month_change = data_basic["close"].resample(holding_period).apply(fun)

            data_asset_month_change = data_asset2["close"].resample(holding_period).apply(fun)
        #   print(data_asset_month_change)

            a = data_asset2["close"].diff(1)/ data_asset2["close"].shift(1)     
            data_asset_change = pd.DataFrame(index = a.index,data = a.values,columns=[asset+"_daily_change"])
            
            b = data_basic["close"].diff(1)/ data_basic["close"].shift(1)
            data_basic_change = pd.DataFrame(index = b.index,data = b.values,columns=["basic_daily_change"])
            
            df = data_asset_change.join(data_basic_change).dropna()

        
            cov = df[asset+"_daily_change"].cov(df["basic_daily_change"])
            var = np.var(df["basic_daily_change"])

            corr = cov/var

            
            #calculate the excess return 
            df_asset_excess_return = (data_asset_month_change - corr * data_basic_month_change).dropna() 
        #    print(df_asset_excess_return)
            
            #predict the ex-votality
            df = data_asset2["close"].diff(1)/ data_asset2["close"].shift(1)
            df = df[start_date:end_date]
        
            weights=[]
            delta = 60/61
            
            for j in range(200):
                params =  delta**(200-j)
                weights.append(params)
            #print(data.values.shape)
                
            weights = np.array(weights)
                  
            
            #predict_the_votality
            vot = []
            for i in range(200,len(df)):
                votality = 0    
                data = (df[i-200:i] - np.average(df[i-200:i],weights=weights))**2 * 261 * (1-delta)
            
                weights = np.array(weights)
            
                votality = (data.values[-200:].T @ weights) 
                vot.append(np.sqrt(votality))
            
            index = df.index[200:]
            
            data_votality_cu = pd.DataFrame(vot,index=index,columns=["votality"])
            data_votality_cu = data_votality_cu[start_date:end_date]
            data_votality_cu = data_votality_cu.resample(holding_period).apply(fun2)
            
            data = data_votality_cu.join(df_asset_excess_return).join(data_asset_month_change,rsuffix="_"+asset).join(data_basic_month_change,rsuffix="_Basic")
            data = data.dropna()
        
            
            real_ret = []
            ret = []
            vot = data["votality"].values
              
            
            # if(delete):
            time_range = range(Lookback_period,len(data)-1) if delete else range(len(data)-1)
            #1-month
            for i in time_range:
                if data.iloc[max(0,i-Lookback_period):i,1].mean()>=0:
                    ret.append(data.iloc[i+1,1])
                else:
                    ret.append(-data.iloc[i+1,1])

            #1-month
            for i in time_range:
                if data.iloc[max(0,i-Lookback_period):i,1].mean()>=0:
                    real_ret.append(data.iloc[i+1,2])
                else:
                    real_ret.append(-data.iloc[i+1,2])
            vot = 0.2/vot[Lookback_period:-1]  if delete else 0.2/vot[:-1]
            index = data.index[Lookback_period:] if delete else data.index
            
            # else:
            #     #1-month
            #     for i in range(len(data)-1):
            #         if data.iloc[max(0,i-Lookback_period):i,1].mean()>=0:
            #             ret.append(data.iloc[i+1,1])
            #         else:
            #             ret.append(-data.iloc[i+1,1])
            #
            #     #1-month
            #     for i in range(len(data)-1):
            #         if data.iloc[max(0,i-Lookback_period):i,1].mean()>=0:
            #             real_ret.append(data.iloc[i+1,2])
            #         else:
            #             real_ret.append(-data.iloc[i+1,2])
            #     vot = 0.2/vot[:-1]
            #     index = data.index
            
            #print(ret)
            ret = np.array(ret)
            real_ret = np.array(real_ret)
            
            real_ret = real_ret * vot
            ret_modified = ret * vot
            ret_modified_x = pd.DataFrame(ret_modified,columns=[asset],index=index[:-1])
            return ret_modified_x
        
        ret_total = calculate_ret(0)
        for i in range(1,len(future_pool)):
            ret_total[future_pool[i]] =  calculate_ret(i)
        ret = ret_total.mean(1)
        
        s_real = [100,]
        for i in range(len(ret)):
            s_real.append(s_real[-1]*(1+ret[i]))
        ret0 = {}

        ret0['tesmom'] = ret
        return ret0

if __name__ == "__main__":
    future_pool = ["cu","al","au","rb","ru","zn","a","c","j","l","m","p","y","cf","fg","oi","sr","ta","wh"]
    
    db_name = 'ultralpha_db'
    host_name = '192.168.0.116'
    user = 'cyr'
    password = 'cyr'
    port = '5432'
    fut_name = 'al'
    start_date = datetime.datetime(2017,1,1)
    end_date = datetime.datetime(2019,6,30)   

    
    future_pool_30 = ['a', 'b', 'c', 'l', 'v', 'j', 'jm', 'm', 'p', 'y',
                      'fu', 'ru', 'al', 'au', 'cu', 'pb', 'rb', 'wr', 'zn',
                      'ag', 'ma', 'sr', 'wh', 'pm', 'cf', 'ta', 'fg', 'oi', 'rm', 'rs']
    
    future_pool_14 = ['c', 'a', 'b', 'jm', 'wh', 'm', 'pm', 'y', 'cf', 'sr', 'fu', 'au', 'cu', 'ag']
    
    
    params={
    "future_pool": future_pool_30,
    "start_date": start_date,
    "end_date": end_date,
    "Lookback_period" : 12,
    "holding_period" : "M",
    "delete" : False,
            }

    
    x = TestMomBacktest(params)
    print(x.run())
    
    
    
    
    
    
    
    
    