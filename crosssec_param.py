# -*- coding: utf-8 -*-
import datetime

date_range_start = datetime.datetime(2010, 1, 1)
date_range_end = datetime.datetime(2018, 12, 31)

# 各行业期货

# 不锈钢, 铝, 铜, 铅, 线材, 锌, 镍, 热轧卷板, 黄金, 白银, 锡
metal =  ['ss', 'al', 'cu', 'pb', 'wr', 'zn', 'ni', 'hc', 'au', 'ag', 'sn']

# 天然橡胶, 原油, 动力煤, 动力煤, '聚乙烯', PTA, 聚丙烯, 石油沥青, 甲醇, 燃料油, 甲醇, 乙二醇, 纸浆, 20号胶, 苯乙烯, 尿素
prod_ener = ['ru', 'sc', 'tc', 'zc', 'l', 'ta', 'pp', 'bu', 'ma', 'fu', 'me', 'eg', 'sp', 'nr', 'eb', 'ur']

# 白糖, 玉米, 棉花, 鸡蛋, 玉米淀粉, 晚籼稻, 强麦, 粳稻谷, 普麦, 苹果, 棉纱, 硬麦, 早籼稻, 红枣, 早籼稻, 强麦, 绿豆, 粳米
grain= ['sr', 'c', 'cf', 'jd', 'cs', 'lr', 'wh', 'jr', 'pm', 'ap', 'cy', 'wt', 'er', 'cj', 'ri', 'ws', 'gn', 'rr']

# 黑色原材料: 铁矿石、焦煤、焦炭、硅铁、锰硅。
black_material = ['i', 'jm', 'j', 'sm', 'sf']

# 油脂油料: 大豆、豆油、豆粕、'菜籽油', '棕榈油', '菜籽粕', '油菜籽', '菜籽油'
oil_oilseeds = ['a', 'b', 'y', 'm', 'oi', 'p', 'rm', 'rs', 'ro']

# 建材: ['玻璃', '螺纹钢', '纤维板', '胶合板', '聚氯乙烯'
constr = ['fg', 'rb', 'fb', 'bb', 'v']

# 年期国债期货, 年期国债期货, 上证股指期货, 沪深指数期货, 年期国债期货, 中证股指期货
finan= ['t', 'ts', 'ih', 'if', 'tf', 'ic']



future_pool_30 = ['a', 'b', 'c', 'l', 'v', 'j', 'jm', 'm', 'p', 'y',
                  'fu', 'ru', 'al', 'au', 'cu', 'pb', 'rb', 'wr', 'zn',
                  'ag', 'ma', 'sr', 'wh', 'pm', 'cf', 'ta', 'fg', 'oi', 'rm', 'rs']

future_pool_14 = ['c', 'a', 'b', 'jm', 'wh', 'm', 'pm', 'y', 'cf', 'sr', 'fu', 'au', 'cu', 'ag']

is_main_contract_op = False       # 是否使用主力合约来作为操作合约，如果false，则需要指明nearest 合约
is_main_contract_indi = True       # 是否使用主力合约来作为计算指标的合约
n_nearest_index_op = 1            # 操作第 n nearest 合约期货，可以取值1，2，3，4，表示近期合约，次近期合约...
n_nearest_index_indi = None            # 凭借第 n nearest 合约计算期货指标，可以取值1，2，3，4，表示近期合约，次近期合约...
if_save_res = True                 # 是否存储运行结果
res_dir = "eval_1211"              # 结果存储文件夹


############### 期货池生成参数 ###############
sel_fut_param = {
        "lower_bound": 2,  # lower bound：每个行业最少期货商品数量
        "upper_bound": 4,  # upper bound: 每个行业最多期货数量
        "if_use_amount": True,  # if_use_amount: 是否选择一定数量的期货，若false，选择当前期货可选比例
        "total_num": 10,  # total_num: 选择期货总数量
        "percent": 0.5,  # percent:  选择期货占当前可选数量的比例
    }

# 多个指标计算期货池
method_1_param = {
        "method": "volatility"  # 指标计算方式： 流动性：amivest，illiq 或者波动率
    }

method_2_param = {
    "method": "liquidity"  # 指标计算方式： 流动性：amivest，illiq 或者波动率
}

method_3_param = {
    "method": "momentum",  # 指标计算方式： 流动性：amivest，illiq 或者波动率
    "num_durations": 12
}

method_4_param = {
    "method": "term_structure",  # 指标计算方式： 流动性：amivest，illiq 或者波动率
    "num_durations": 12
}

multi_indi_method_param = {
    "method_list": [method_1_param, method_2_param],
    "method_weighting": [0.7, 0.3]
}

# 运行多个指标计算
# fut_pool_param = {
#     "type_list": None,  # 期货池所在行业：可选择，金属，能化，黑原等，若为 None，则默认候选项为所有期货商品
#     "cal_method": "multiple", # 通过一个指标还是多个指标来计算排序期货商品, 流动性 + 波动率 组合
#     "method_param": multi_indi_method_param,
#     "sel_fut_param": sel_fut_param
# }

fut_pool_param = {
    "type_list": None,  # 期货池所在行业：可选择，金属，能化，黑原等，若为 None，则默认候选项为所有期货商品
    "cal_method": "single", # 通过一个指标还是多个指标来计算排序期货商品, 流动性 + 波动率 组合
    "method_param": method_2_param,
    "sel_fut_param": sel_fut_param
}

params_crosssec = {  "fut_pool_gen_method": "static",
            "fut_pool_gen_param": None,
            "future_pool": None,                     # 固定底池，如何删除？
            ##############################################
            "start_date": date_range_start,
            "end_date": date_range_end,
            ##############################################
            "is_main_contract_op": is_main_contract_op,
            "is_main_contract_indi": is_main_contract_indi,
            "n_nearest_index_op": n_nearest_index_op,
            "n_nearest_index_indi": n_nearest_index_indi,
            ###############################################
            "freq": 7,
            "freq_unit": "D",
            ###############################################
            "indicator_cal_method": None,
            "if_save_res":if_save_res,
            "res_dir": res_dir}

param_basic = {
    "future_pool": None,
    "start_date": date_range_start,
    "end_date": date_range_end,
}

strategy_param = {
        'basic': param_basic,
        # 'crosssec':params_crosssec,
    }

def gen_param():
    strategy_param = {
        'basic': param_basic,
        # 'crosssec':params_crosssec,
    }
    i = 0
    for method in [ "momentum", "term_structure2", "volatility", "hedging_pressure", "liquidity", "skewness",
                    "open_interest", "value", "term_structure", "currency",  "inflation" ]:
        params_crosssec["indicator_cal_method"] = method

        for fut_pool_gen_method in ["static" ]:
            if fut_pool_gen_method == "dynamic":
                for fut_pool_gen_param in [fut_pool_param]:
                    params_crosssec["fut_pool_gen_method"]  = fut_pool_gen_method
                    params_crosssec["fut_pool_gen_param"]  = fut_pool_gen_param
                    params_crosssec["future_pool"] = None
                    strategy_param["crosssec_"+str(i)] = params_crosssec.copy()
                    i += 1
            elif fut_pool_gen_method == "static":
                for future_pool in [metal, prod_ener, grain, black_material, oil_oilseeds, constr, finan, future_pool_14, future_pool_30]:
                    # , future_pool_14, future_pool_30
                    params_crosssec["fut_pool_gen_method"] = fut_pool_gen_method
                    params_crosssec["fut_pool_gen_param"] = None
                    params_crosssec["future_pool"] = future_pool
                    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                    i += 1

    return strategy_param

def gen_param_diff_industry():
    strategy_param = {
        'basic': param_basic,
    }
    i = 0
    fut_pool_gen_method = "static"
    params_crosssec["fut_pool_gen_method"] = fut_pool_gen_method
    params_crosssec["fut_pool_gen_param"] = None
    # , 30, 50
    for freq in [30]:
        params_crosssec["freq"] = freq
        # metal, prod_ener, grain, black_material, oil_oilseeds, constr, finan, future_pool_14,
        for future_pool in [ future_pool_30]:
            params_crosssec["future_pool"] = future_pool
            for method in [ "open_interest", "momentum","term_structure"]:
                params_crosssec["indicator_cal_method"] = method
                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                i += 1
    return strategy_param


def gen_param_2():
    i = 0
    for  method in [ "inflation", "momentum", "term_structure2", "volatility", "hedging_pressure",
                     "liquidity", "skewness", "open_interest", "value", "term_structure", "currency" ]:
        params_crosssec["indicator_cal_method"] = method
        if method != "term_structure2":
            for is_main_contract_op in [True, False]:
                params_crosssec["is_main_contract_op"] = is_main_contract_op
                # 操作 1， 2， 3， 4 nearest 合约， 使用主力合约或者近期合约来计算指标
                if not is_main_contract_op:
                    for n_nearest_index_op in [1, 2, 3, 4]:
                        params_crosssec["n_nearest_index_op"] = n_nearest_index_op
                        for is_main_contract_indi in [True, False]:
                            params_crosssec["is_main_contract_indi"] = is_main_contract_indi
                            params_crosssec["n_nearest_index_indi"] = 1 if not is_main_contract_indi else None
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1
                #  操作主力合约， 只用主力合约或者近期合约来计算指标
                else:
                    params_crosssec["n_nearest_index_op"] = None
                    for is_main_contract_indi in [True, False]:
                        params_crosssec["is_main_contract_indi"] = is_main_contract_indi
                        params_crosssec["n_nearest_index_indi"] = 1 if not is_main_contract_indi else None
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1

        elif method == "term_structure2":
            for is_main_contract_op in [True, False]:
                params_crosssec["is_main_contract_op"] = is_main_contract_op
                if not is_main_contract_op:
                    for n_nearest_index_op in [1, 2, 3, 4]:
                        params_crosssec["n_nearest_index_op"] = n_nearest_index_op
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1
                else:
                    params_crosssec["n_nearest_index_op"] = None
                    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                    i += 1
    return strategy_param


# def gen_param_currency():
#     strategy_param = {
#         'basic': param_basic,
#         # 'crosssec':params_crosssec,
#     }
#     i = 0
#     params_crosssec["indicator_cal_method"] = "currency"
#
#     ####### first group : prod&energ,grain,black_material-indi_main_op_3
#
#     fut_pool_param["type_list"] = [ "农产品"]
#     params_crosssec["is_main_contract_op"] = False
#     params_crosssec["n_nearest_index_op"] = 1
#     params_crosssec["is_main_contract_indi"] = True
#     params_crosssec["n_nearest_index_indi"] = None
#     params_crosssec["freq"] = 5
#     params_crosssec["num_durations"] = 20
#
#     params_crosssec["fut_pool_gen_method"] = "dynamic"
#     params_crosssec["fut_pool_gen_param"]  = fut_pool_param
#     params_crosssec["future_pool"] = None
#
#     gen_dynamic_param(strategy_param, i)
#     return strategy_param
#


def gen_param_hedge_pressure():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "hedging_pressure"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    params_crosssec["weighted"]  = False
    for percent in [0.1, 0.15, 0.2]:
        params_crosssec["top"] = percent
        params_crosssec["bottom"] = percent

        # 时间, look back period
        for num_days in [15, 30, 45, 60, 120]:
            params_crosssec["num_days"] = num_days

            # operate frenquency
            for freq in [10, 30, 50]:
                params_crosssec["freq"] = freq

                for is_main_contract_op in [True, False]:
                    params_crosssec["is_main_contract_op"] = is_main_contract_op

                    if is_main_contract_op:
                        params_crosssec["n_nearest_index_op"] = None
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1
                    else:
                        params_crosssec["n_nearest_index_op"] = 1
                        for if_check_vol in [True, False]:
                            params_crosssec["if_check_vol"] = if_check_vol
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1

    return strategy_param


def gen_dynamic_param(strategy_param, i):
    for lower_bound in [1, 2, 3]:
        sel_fut_param["lower_bound"] = lower_bound
        for upper_bound in [4, 6,  8]:
            sel_fut_param["upper_bound"] = upper_bound
            for total_num in [10, 15, 20]:
                sel_fut_param["total_num"] = total_num
                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                strategy_param["crosssec_" + str(i)]["fut_pool_gen_param"] = fut_pool_param.copy()
                strategy_param["crosssec_" + str(i)]["fut_pool_gen_param"]["method_2_param"] = method_2_param.copy()
                strategy_param["crosssec_" + str(i)]["fut_pool_gen_param"]["sel_fut_param"] = sel_fut_param.copy()
                i += 1
                print(params_crosssec)
                print("*"*11)
    return i

def gen_param_momentum():
    strategy_param = {
        'basic': param_basic,
        # 'crosssec':params_crosssec,
    }
    i = 0
    params_crosssec["indicator_cal_method"] = "momentum"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    params_crosssec["weighted"] = False
    for percent in [0.1, 0.15, 0.2, 0.25]:
        params_crosssec["top"] = percent
        params_crosssec["bottom"] = percent

        # 时间, look back period
        for num_durations in [3, 6, 12, 24]:
            params_crosssec["num_durations"] = num_durations

            # operate frenquency
            for freq in [10, 30, 50]:
                params_crosssec["freq"] = freq

                for is_main_contract_op in [True, False]:
                    params_crosssec["is_main_contract_op"] = is_main_contract_op

                    if is_main_contract_op:
                        params_crosssec["n_nearest_index_op"] = None
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1
                    else:
                        params_crosssec["n_nearest_index_op"] = 1
                        for if_check_vol in [True, False]:
                            params_crosssec["if_check_vol"] = if_check_vol
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1

    return strategy_param

def gen_param_term_structure():
    strategy_param = {
        'basic': param_basic,
        # 'crosssec':params_crosssec,
    }
    i = 0
    params_crosssec["indicator_cal_method"] = "term_structure"

    ####### first group : prod&energ,grain,black_material-indi_main_op_3

    fut_pool_param["type_list"] = ["能化", "油脂油料", "金属"]
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] = None
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 10

    params_crosssec["fut_pool_gen_method"] = "static" # "dynamic"
    params_crosssec["fut_pool_gen_param"]  = fut_pool_param
    params_crosssec["future_pool"] = None

    gen_dynamic_param(strategy_param, i)

    return strategy_param


def gen_param_open_interest():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "open_interest"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    params_crosssec["weighted"] = False
    for percent in [0.1, 0.15, 0.2, 0.25]:
        params_crosssec["top"] = percent
        params_crosssec["bottom"] = percent

        # 时间, look back period
        for num_days in [15, 30, 45, 60, 120]:
            params_crosssec["num_days"] = num_days

            # operate frenquency
            for freq in [30]:
                params_crosssec["freq"] = freq

                for is_main_contract_op in [True, False]:
                    params_crosssec["is_main_contract_op"] = is_main_contract_op

                    if is_main_contract_op:
                        params_crosssec["n_nearest_index_op"] = None
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1
                    else:
                        params_crosssec["n_nearest_index_op"] = 1
                        for if_check_vol in [True, False]:
                            params_crosssec["if_check_vol"] = if_check_vol
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1

    return strategy_param

def gen_param_skew():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "skewness"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    params_crosssec["weighted"] = False
    for percent in [0.1, 0.15, 0.2, 0.25]:
        params_crosssec["top"] = percent
        params_crosssec["bottom"] = percent

        # 时间, look back period
        for num_days in [120, 240, 360, 480]:
            params_crosssec["num_days"] = num_days

            # operate frenquency
            for freq in [10, 30, 50]:
                params_crosssec["freq"] = freq

                for is_main_contract_op in [True, False]:
                    params_crosssec["is_main_contract_op"] = is_main_contract_op

                    if is_main_contract_op:
                        params_crosssec["n_nearest_index_op"] = None
                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                        i += 1
                    else:
                        params_crosssec["n_nearest_index_op"] = 1
                        for if_check_vol in [True, False]:
                            params_crosssec["if_check_vol"] = if_check_vol
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1

    return strategy_param

def gen_param_inflation():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "inflation"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    for index_col in ["y2y_0", "m2m", "y2y_1"]:
        params_crosssec["index_col"] = index_col

        params_crosssec["weighted"] = False
        #, 0.15, 0.2, 0.25
        for percent in [0.1]:
            params_crosssec["top"] = percent
            params_crosssec["bottom"] = percent

            # 时间, look back period
            # 20, 40, 60
            for num_durations in [20]:
                params_crosssec["num_durations"] = num_durations

                # operate frenquency
                # 10, 30, 50
                for freq in [10]:
                    params_crosssec["freq"] = freq

                    # , False
                    for is_main_contract_op in [True]:
                        params_crosssec["is_main_contract_op"] = is_main_contract_op

                        if is_main_contract_op:
                            params_crosssec["n_nearest_index_op"] = None
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1
                        else:
                            params_crosssec["n_nearest_index_op"] = 1
                            for if_check_vol in [True, False]:
                                params_crosssec["if_check_vol"] = if_check_vol
                                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                i += 1

    return strategy_param



def gen_param_liq():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "liquidity"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    params_crosssec["weighted"] = False
    for method in ["amivest", "amihud"]:
        params_crosssec["method"] = method

        for percent in [0.1, 0.15, 0.2, 0.25]:
            params_crosssec["top"] = percent
            params_crosssec["bottom"] = percent

            # 时间, look back period
            for num_days in [30, 60, 120]:
                params_crosssec["num_days"] = num_days

                # operate frenquency
                for freq in [10, 30, 50]:
                    params_crosssec["freq"] = freq

                    for is_main_contract_op in [True, False]:
                        params_crosssec["is_main_contract_op"] = is_main_contract_op

                        if is_main_contract_op:
                            params_crosssec["n_nearest_index_op"] = None
                            strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                            i += 1
                        else:
                            params_crosssec["n_nearest_index_op"] = 1
                            for if_check_vol in [True, False]:
                                params_crosssec["if_check_vol"] = if_check_vol
                                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                i += 1

    return strategy_param


def gen_param_term_structure2():
    strategy_param = {
        'basic': param_basic,
        # 'crosssec':params_crosssec,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "term_structure2"

    # 不同 期货池

    freq_dic = {
        10:[ metal+constr+black_material],
        5: [metal,oil_oilseeds, prod_ener, finan, metal+oil_oilseeds+prod_ener+finan],
        30:[metal+oil_oilseeds+prod_ener]
    }

    for future_pool in [metal,oil_oilseeds, prod_ener, finan, metal+oil_oilseeds+prod_ener+finan, metal+oil_oilseeds+prod_ener, metal+constr+black_material]:

        params_crosssec["fut_pool_gen_method"] = "static"
        params_crosssec["fut_pool_gen_param"] = None
        params_crosssec["future_pool"] = future_pool

        for key, value in freq_dic.items():
            if future_pool in value:
                params_crosssec["freq"] = key
                print("new update freq ", key , future_pool)
                break

        params_crosssec["is_main_contract_indi"] = False
        params_crosssec["n_nearest_index_indi"] = None

        for is_main_contract_op in [False]:
            params_crosssec["is_main_contract_op"] = is_main_contract_op
            if not is_main_contract_op:
                for n_nearest_index_op in [1, 4]:
                    params_crosssec["n_nearest_index_op"] = n_nearest_index_op
                    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                    i += 1
            else:
                params_crosssec["n_nearest_index_op"] = None
                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                i += 1

    return strategy_param

def gen_param_value():
    strategy_param = {
        'basic': param_basic,
    }

    i = 0
    params_crosssec["indicator_cal_method"] = "value"

    # 期货池
    params_crosssec["fut_pool_gen_method"] = "static"
    params_crosssec["fut_pool_gen_param"] = None
    params_crosssec["future_pool"] = future_pool_30

    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None

    # 时间
    for num_days in [100, 200, 300]:
        params_crosssec["num_days"] = num_days

        for num_years in [ 3, 4, 5]:
            params_crosssec["num_years"] = num_years

            for freq in [10, 30, 50]:
                params_crosssec["freq"] = freq

                for is_main_contract_op in [True, False]:
                    params_crosssec["is_main_contract_op"] = is_main_contract_op

                    if is_main_contract_op:
                        params_crosssec["n_nearest_index_op"] = None

                        for weighted in [True, False]:
                            params_crosssec["weighted"] = weighted

                            if not weighted:
                                for percent in [0.1, 0.15, 0.2]:
                                    params_crosssec["top"] = percent
                                    params_crosssec["bottom"] = percent
                                    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                    i += 1
                            else:
                                strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                i += 1

                    else:
                        params_crosssec["n_nearest_index_op"] = 1
                        for if_check_vol in [True, False]:
                            params_crosssec["if_check_vol"] = if_check_vol

                            for weighted in [True, False]:
                                params_crosssec["weighted"] = weighted

                                if not weighted:
                                    for percent in [0.1, 0.15, 0.2]:
                                        params_crosssec["top"] = percent
                                        params_crosssec["bottom"] = percent
                                        strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                        i += 1
                                else:
                                    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
                                    i += 1
    return strategy_param

