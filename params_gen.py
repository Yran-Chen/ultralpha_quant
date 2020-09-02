import pandas as pd
import datetime

date_range_faststart = datetime.datetime(2018,1,1)
date_range_start = datetime.datetime(2012,1,1)
date_range_end = datetime.datetime(2019,5,31)

param_basic_fasttest = {
    "future_pool": None,
    "start_date": date_range_faststart,
    "end_date": date_range_end,
}

param_basic_alltime = {
"future_pool": None,
"start_date": datetime.datetime(2005,1,1),
"end_date": datetime.datetime(2019,10,1),
}

param_basic_2005_2017 = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
}

params_idiosyn = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
    'method':'imom',
    # 'mom_month':3,
    # 'weight_method':'equal',
}

params_pool = {
    'basic':param_basic_2005_2017

}

def params_generator(params_pool,weight_method_,method_,num_month_=[6],qcut_ = [4],rev_backtime_ = ['360'],backtest_name = 'idiosyn'):
    for num_month in num_month_:
        for weight_method in weight_method_:
            for method in method_:
                for rev_backtime in rev_backtime_:
                    for qcut in qcut_:
                        name = '{0}_{1}_{2}_{3}_{4}_{5}'.format(backtest_name,num_month,weight_method,method,rev_backtime,qcut)
                        params_pool[name] = {
                            "future_pool": None,
                            "start_date": date_range_start,
                            "end_date": date_range_end,
                            'method': method,
                            'rev_backtime':rev_backtime,
                            'mom_month':num_month,
                            'weight_method':weight_method,
                            'qcut':qcut,
                        }
    return params_pool