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

is_main_contract_op = True         # 是否使用主力合约来作为操作合约，如果false，则需要指明nearest 合约
is_main_contract_indi = True       # 是否使用主力合约来作为计算指标的合约
n_nearest_index_op = None              # 操作第 n nearest 合约期货，可以取值1，2，3，4，表示近期合约，次近期合约...
n_nearest_index_indi = None            # 凭借第 n nearest 合约计算期货指标，可以取值1，2，3，4，表示近期合约，次近期合约...
if_save_res = True                 # 是否存储运行结果
res_dir = "eval_1211"              # 结果存储文件夹


params_crosssec = {  "fut_pool_gen_method": "static",
            "fut_pool_gen_param": None,
            "future_pool": future_pool_30,                     # 固定底池，如何删除？
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
    "future_pool": future_pool_30,
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
    params_crosssec["indicator_cal_method"] = "term_structure"

    ####### first group : prod&energ,grain,black_material-indi_main_op_3
    params_crosssec["future_pool"] = metal + prod_ener + oil_oilseeds
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] = None
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 10

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1

    params_crosssec["future_pool"] = metal + prod_ener
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] = None
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 30

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1

    params_crosssec["indicator_cal_method"] = "term_structure"

    params_crosssec["future_pool"] = grain + prod_ener + black_material
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] = None
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 10
    params_crosssec["num_durations"] = 12

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1

    params_crosssec["future_pool"] = grain
    params_crosssec["is_main_contract_op"] = False
    params_crosssec["n_nearest_index_op"] = 3
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 10
    params_crosssec["num_durations"] = 12

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1

    params_crosssec["future_pool"] = prod_ener
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] =   None
    params_crosssec["is_main_contract_indi"] = True
    params_crosssec["n_nearest_index_indi"] = None
    params_crosssec["freq"] = 5
    params_crosssec["num_durations"] = 12

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1

    params_crosssec["indicator_cal_method"] = "currency"

    params_crosssec["future_pool"] = grain  + oil_oilseeds + finan
    params_crosssec["is_main_contract_op"] = True
    params_crosssec["n_nearest_index_op"] = None
    params_crosssec["is_main_contract_indi"] = False
    params_crosssec["n_nearest_index_indi"] = 1
    params_crosssec["freq"] = 10
    params_crosssec["num_durations"] = 20

    strategy_param["crosssec_" + str(i)] = params_crosssec.copy()
    i += 1


    return strategy_param


