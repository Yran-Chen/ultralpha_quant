import math
import pandas as pd
from DB_Future import FutureDatabase
from time_process import get_next_month
import matplotlib.pyplot as plt

class FastBacktest:

    def __init__(self, params, data_proxy = None):
        self.params = params
        # 确定回测的期货池
        try:
            self.future_pool = params["future_pool"]
        except:
            raise RuntimeError("请在输入参数中传递期货池")

        #确定回测的时间段
        try:
            self.start_date, self.end_date = params["start_date"], params["end_date"]
        except:
            raise RuntimeError("请确定回测的起始时间")


        # 创造期货数据类
        # 用户登录帐号等设置
        if data_proxy is None:
            db_name = 'ultralpha_db'
            host_name = '192.168.0.116'
            user = 'cyr'
            password = 'cyr'
            port = '5432'

            # todo: how to choose the data cache size, if choos the date range?
            set_time_range = False
            self.data_proxy = FutureDatabase(db_name=db_name, host_name=host_name, user_name=user, pwd=password, port=port,\
                                             start_date = self.start_date,end_date = self.end_date, set_time_range = set_time_range)
            # todo: if necessary to add this function
        else:
            self.data_proxy = data_proxy
    def run(self):
        pass

