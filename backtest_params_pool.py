import pandas as pd
import datetime


pd.set_option('display.max_columns', None)
from dateutil.parser import parse


date_range_start = datetime.datetime(2017,1,1)
date_range_end = datetime.datetime(2019,5,31)

# 是否使用主力合约来作为操作合约，如果false，则需要指明nearest 合约
is_main_contract_op = False
# 是否使用主力合约来作为计算指标的合约
is_main_contract_indicator = True
# 可以取值1，2，3，4，表示近期合约，次近期合约...
n_nearest_index = 1
future_pool_30 = ['a', 'b', 'c', 'l', 'v', 'j', 'jm', 'm', 'p', 'y',
                  'fu', 'ru', 'al', 'au', 'cu', 'pb', 'rb', 'wr', 'zn',
                  'ag', 'ma', 'sr', 'wh', 'pm', 'cf', 'ta', 'fg', 'oi', 'rm', 'rs']
future_pool_14 = ['c', 'a', 'b', 'jm', 'wh', 'm', 'pm', 'y', 'cf', 'sr', 'fu', 'au', 'cu', 'ag']
cal_indicator_method_list = [
               "term_structure",
               "hedging_pressure",
               # "momentum",
               # "volatity",
               # "open_interest",
               # "liquidity",
               #  "currency",
               #  "skewness",
               #  "value"
                ]
start_date = date_range_start.strftime('%Y-%m-%d')
        
#提前300天以便于计算第一天的事前波动率
start_date = (parse(start_date) + datetime.timedelta(days=-305)).strftime("%Y-%m-%d")

param_basic = {
    "future_pool": None,
    "start_date": start_date,
    "end_date": date_range_end,
}

params_meanrev = {
    "future_pool": ['al'],
    "start_date": date_range_start,
    "end_date": date_range_end,
    'method':"2nearest"
}

params_crosssec = {
    "future_pool": future_pool_14,
    "start_date": date_range_start,
    "end_date": date_range_end,
    "is_main_contract_op": is_main_contract_op,
    "is_main_contract_indicator": is_main_contract_indicator,
    "n_nearest_index": n_nearest_index,
    "indicator_cal_method": cal_indicator_method_list}

params_tesmom={
    "future_pool": future_pool_30,
    "start_date": date_range_start,
    "end_date": date_range_end,
    "Lookback_period" : 12,
    "holding_period" : "M",
    "delete" : False,
            }

params_null = {
    "future_pool": None,
    "start_date": None,
    "end_date": None,
}

params_idiosyn = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
    'method':'imom',
    'mom_month':6,
    'rev_backtime':30
}

condition = [
    'sharpe > 0.2',
    'max(sharpe) > 0.5',
    'abs(returns) < 0.5'
]

strategy_param_pool_demo = {
    'condition':condition,
    'basic':param_basic,
    'tesmom': params_tesmom,
    'meanrev':params_meanrev,
    'crosssec':params_crosssec,
}

params_idiosyn9 = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
    'method':'imom',
    'mom_month':9,
}
params_idiosyn6 = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
    'method':'imom',
    'mom_month':6,
}