# -*- coding: utf-8 -*-
import time
import datetime
import pandas as pd
import numpy as np
from DB_Future import FutureDatabase
from indicator_generator import IndicatorGenerator

# 每个月提供当前的期货池，期货商品类别
# 更新频率， 月
class FuturePoolGen:

    def __init__(self, data_proxy, indi_generator):
        # 从数据库获得数据
        self.data_proxy = data_proxy
        self.indi_generator = indi_generator

        code_name_dic = {'a': '豆一', 'ag': '白银', 'al': '铝', 'ap': '苹果', 'au': '黄金', 'b': '豆二', 'bb': '胶合板', 'bu': '石油沥青', 'c': '玉米',
         'cf': '棉花', 'cj': '红枣', 'cs': '玉米淀粉', 'cu': '铜', 'cy': '棉纱', 'eb': '苯乙烯', 'eg': '乙二醇', 'er': '早籼稻',
         'fb': '纤维板', 'fg': '玻璃', 'fu': '燃料油', 'gn': '绿豆', 'hc': '热轧卷板', 'i': '铁矿石', 'ic': '中证股指期货', 'if': '沪深指数期货',
         'ih': '上证股指期货', 'j': '焦炭', 'jd': '鸡蛋', 'jm': '焦煤', 'jr': '粳稻谷', 'l': '聚乙烯', 'lr': '晚籼稻', 'm': '豆粕', 'ma': '甲醇',
         'me': '甲醇', 'ni': '镍', 'nr': '20号胶', 'oi': '菜籽油', 'p': '棕榈油', 'pb': '铅', 'pm': '普麦', 'pp': '聚丙烯', 'rb': '螺纹钢',
         'ri': '早籼稻', 'rm': '菜籽粕', 'ro': '菜籽油', 'rr': '粳米', 'rs': '油菜籽', 'ru': '天然橡胶', 'sc': '原油', 'sf': '硅铁',
         'sm': '锰硅', 'sn': '锡', 'sp': '纸浆', 'sr': '白糖', 'ss': '不锈钢', 't': '年期国债期货', 'ta': 'PTA', 'tc': '动力煤',
         'tf': '年期国债期货', 'ts': '年期国债期货', 'ur': '尿素', 'v': '聚氯乙烯', 'wh': '强麦', 'wr': '线材', 'ws': '强麦', 'wt': '硬麦',
         'y': '豆油', 'zc': '动力煤', 'zn': '锌'}

        # type_future_dic = {"energy": ["FU", "MA", "SC", "TC"],
        #                    "grains": ["A", "B", "C", "SR", "WH", "PM", "WT", "JR", "ER", "WS", "AP", "LR", "RR"],
        #                    "oilseeds": ["M", "P", "Y", "RU", "OI", "RM", "RS", "RO"],
        #                    "industrial": ["L", "V", "J", "JM", "CF", "TA", "FG", "CY", "PP", "CS", "BB", "UR"],
        #                    "metal": ["AL", "AU", "CU", "PB", "RB", "WR", "ZN", "AG", "I", "NI", "SF", "SS", "HC"]}

        self.metal_name = "金属"
        self.prod_ener_name = "能化"
        self.grain_name = "农产品"
        self.black_material_name = "黑色原材料"
        self.oil_oilseeds_name = "油脂油料"
        self.constr_name = "建材"
        self.finan_name = "金融"

        self.type_future_dic = {
            # 不锈钢, 铝, 铜, 铅, 线材, 锌, 镍, 热轧卷板, 黄金, 白银, 锡
            self.metal_name: ['ss', 'al', 'cu', 'pb', 'wr', 'zn', 'ni', 'hc', 'au', 'ag', 'sn'],
            # 天然橡胶, 原油, 动力煤, 动力煤, '聚乙烯', PTA, 聚丙烯, 石油沥青, 甲醇, 燃料油, 甲醇, 乙二醇, 纸浆, 20号胶, 苯乙烯, 尿素
            self.prod_ener_name: ['ru', 'sc', 'tc', 'zc', 'l', 'ta', 'pp', 'bu', 'ma', 'fu', 'me', 'eg', 'sp', 'nr', 'eb', 'ur'],
            # 白糖, 玉米, 棉花, 鸡蛋, 玉米淀粉, 晚籼稻, 强麦, 粳稻谷, 普麦, 苹果, 棉纱, 硬麦, 早籼稻, 红枣, 早籼稻, 强麦, 绿豆, 粳米
            self.grain_name: ['sr', 'c', 'cf', 'jd', 'cs', 'lr', 'wh', 'jr', 'pm', 'ap', 'cy', 'wt', 'er', 'cj', 'ri', 'ws', 'gn', 'rr'],
            # 黑色原材料: 铁矿石、焦煤、焦炭、硅铁、锰硅。
            self.black_material_name: ['i', 'jm', 'j', 'sm', 'sf'],
            # 油脂油料: 大豆、豆油、豆粕、'菜籽油', '棕榈油', '菜籽粕', '油菜籽', '菜籽油'
            self.oil_oilseeds_name: ['a', 'b', 'y', 'm', 'oi', 'p', 'rm', 'rs', 'ro'],
            # 建材: ['玻璃', '螺纹钢', '纤维板', '胶合板', '聚氯乙烯'
            self.constr_name: ['fg', 'rb', 'fb', 'bb', 'v'],
            # 年期国债期货, 年期国债期货, 上证股指期货, 沪深指数期货, 年期国债期货, 中证股指期货
            self.finan_name: ['t', 'ts', 'ih', 'if', 'tf', 'ic']
        }

        future_type_dic = {}
        for key, val_list in self.type_future_dic.items():
            for var in val_list:
                future_type_dic[var.lower()] = key

        self.future_type_dic = future_type_dic


    def __parse_param(self, param, var_list):
        alist = []
        for var in var_list:
            try:
                alist.append(param[var])
            except:
                raise ValueError("不存在参数 {}".format(var))
        return alist

    # 获得期货池
    # params: type_list: 选择希望期货行业范围， ["金属", "能化"...]
    def get_future_pool(self, param):
        print("Generate future pool ...")
        t0  = time.time()
        type_list, cal_method, method_param, sel_fut_param = self.__parse_param(param, ["type_list", "cal_method", "method_param", "sel_fut_param"])

        future_pool = None
        # （1）确定行业范围： None --- 所有期货商品皆为备选
        if isinstance(type_list, list):
            future_pool = self.__check_type_list(type_list)

        # （2）在备选中生成指标
        if cal_method == "single":
            self.__update_param(from_param=param, to_param=method_param, future_pool=future_pool)
            indicator_df = self.__cal_indicator(param = method_param)
        elif cal_method == "multiple":
            for m_param in method_param["method_param_list"]:
                self.__update_param(from_param=param, to_param=m_param, future_pool=future_pool)
            indicator_df = self.__cal_combine_indi(param=method_param)
        print("Time usage for cal future pool ", time.time() - t0)

        # （3）选择期货 -- 对于时间节点
        return self.__select_future(indicator_df=indicator_df, param=sel_fut_param)

    def __update_param(self, from_param, to_param, future_pool):
        for param in ["start_date", "end_date", "freq", "freq_unit"]:
            to_param[param] = from_param[param]
        to_param["future_pool"] = future_pool
        to_param["is_main_contract_indi"] = True

    def __cal_indicator(self, param):
        indi_df = self.indi_generator.cal_indicator(param)
        return indi_df

    def __cal_combine_indi(self, param):
        method_list, method_weighting = self.__parse_param( param=param, var_list = ["method_param_list", "method_weighting"])
        r = 0
        indicator_df = None

        for method_param, weighting in zip(method_list, method_weighting):
            indi_df = self.__cal_indicator(param=method_param)
            mean_val, std_val = indi_df["indicator"].mean(), indi_df["indicator"].std()
            indi_df["indicator"] = indi_df["indicator"].map(lambda x: (x-mean_val)/std_val)
            indi_df["indicator"] = indi_df["indicator"].map(lambda x: weighting*x)
            # print("indi df got ")
            # print( indi_df[:20] )
            if r == 0:
                indicator_df = indi_df.copy()
            else:
                indi_df.rename(columns = {"indicator": "new"}, inplace=True)
                indicator_df  = pd.merge(indicator_df, indi_df, on=["date", "instrument"], how="inner")
                indicator_df["indicator"] = indicator_df.apply(lambda  r: r["indicator"] + r["new"], axis =1)
                indicator_df = indicator_df.drop(columns = ["new"], axis =1)
            r +=1
        return indicator_df

    def __select_future(self, indicator_df, param):
        lower_bound, upper_bound, if_use_amount, total_num, percent  = \
            self.__parse_param(param=param, var_list = ["lower_bound", "upper_bound", "if_use_amount", "total_num", "percent"])
        indicator_df["ins_type"] = indicator_df["instrument"].map(lambda x: self.future_type_dic[x])

        def select_future(sub_df):
            ascending = False
            select_list, unselect_list =[], []
            for k, g in sub_df.groupby(["ins_type"]):
                g = g.sort_values(by = ["indicator"], ascending = ascending)
                # 每个期货商品选定一定数量的期货，其余的定量作为候选
                select_list.append(g[:lower_bound])
                unselect_list.append(g[lower_bound:upper_bound])
            select = pd.concat(select_list)
            unselect = pd.concat(unselect_list)
            if if_use_amount:
                left_amount = total_num - select.shape[0]
                left_amount = left_amount if left_amount >0 else 0
                unselect = unselect.sort_values(by = ["indicator"], ascending=ascending)[:left_amount]
            else:
                left_amount = int(sub_df.shape[0]*percent) - select.shape[0]
                left_amount = left_amount if left_amount> 0 else 0
                unselect = unselect.sort_values(by = ["indicator"], ascending=ascending)[:left_amount]
            return pd.concat([select, unselect], axis = 0)

        indicator_df = indicator_df.groupby(["date"], as_index = False).apply(select_future).reset_index(drop = True)
        indicator_df.drop(columns =["indicator", "ins_type"], inplace = True)
        return indicator_df

    # function:判断行业列表中的所有行业是否在可选择范围内
    # params: type_list: 选择希望期货行业范围， ["金属", "能化"...]
    def __check_type_list(self, type_list):
        poss_choose_list = [self.metal_name, self.prod_ener_name, self.grain_name, self.black_material_name,
                            self.oil_oilseeds_name,
                            self.constr_name, self.finan_name, ]
        future_pool  = []
        for item in type_list:
            if item not in poss_choose_list:
                raise ValueError("请检查期货池选择行业范围，可选择范围：", " ".join(poss_choose_list))
            future_pool += self.type_future_dic[item]
        return future_pool


if __name__ == "__main__":
    # 创造期货数据类
    # 用户登录帐号等设置
    db_name = 'ultralpha_db'
    host_name = '192.168.0.116'
    user = 'cyr'
    password = 'cyr'
    port = '5432'

    sel_fut_param = {
        "lower_bound": 1,  # lower bound：每个行业最少期货商品数量
        "upper_bound": 1,  # upper bound: 每个行业最多期货数量
        "if_use_amount": True,  # if_use_amount: 是否选择一定数量的期货，若false，选择当前期货可选比例
        "total_num": 15,  # total_num: 选择期货总数量
        "percent": 0.5,  # percent:  选择期货占当前可选数量的比例
    }

    method_1_param = {
        "time_range": 120,  # time range: 计算指标基于的时间范围， 单位工作交易日
        "method": "volatility"  # 指标计算方式： 流动性：amivest，illiq 或者波动率
    }

    method_2_param = {
        "time_range": 60,  # time range: 计算指标基于的时间范围， 单位工作交易日
        "method": "liquidity"  # 指标计算方式： 流动性：amivest，illiq 或者波动率
    }

    multi_indi_method_param = {
        "method_param_list": [method_1_param, method_2_param],
        "method_weighting": [0.7, 0.3]
    }

    # 运行多个指标计算
    param = {
        "type_list": None,  # 期货池所在行业：可选择，金属，能化，黑原等，若为 None，则默认候选项为所有期货商品
        "cal_method": "multiple", # 通过一个指标还是多个指标来计算排序期货商品, 流动性 + 波动率 组合
        "method_param": multi_indi_method_param,
        "sel_fut_param": sel_fut_param,
        "start_date": datetime.datetime(2010, 2, 1),
        "end_date": datetime.datetime(2017, 2, 1),
        "freq_unit": "M",
        "freq": 7,
    }

    # 运行单个指标计算
    # param = {
    #     "type_list": None,  # 期货池所在行业：可选择，金属，能化，黑原等，若为 None，则默认候选项为所有期货商品
    #     "cal_method": "single",  # 通过一个指标还是多个指标来计算排序期货商品, 流动性 + 波动率 组合
    #     "method_param": method_param,
    #     "sel_fut_param": sel_fut_param
    # }

    data_proxy = FutureDatabase(db_name=db_name, host_name=host_name, user_name=user, pwd=password, port=port)
    indi_gen = IndicatorGenerator(future_db= data_proxy)
    future_pool_gen = FuturePoolGen(data_proxy=data_proxy, indi_generator=indi_gen)
    future_pool_gen.get_future_pool(param=param)

    ################################# 计算交易金额 ################################################

    # 获得包含成交金额的数据， todo： 是否写入数据库
    # basic_df = pd.read_csv("future_basic_info.csv")
    # day_df = pd.read_csv("future_day.csv")
    # basic_df["code"] = basic_df["security_code"].map(lambda r: r.split(".")[0])
    # future_df = pd.merge(day_df, basic_df, on=["code"], how="inner")
    # future_df = future_df.drop(["security_code", "name", "type"], axis=1)
    # # 存在两个期货合约code名字一样但是时间相差十年的情况
    # future_df = future_df[(future_df["date"]>= future_df["start_date"]) & (future_df["date"] <= future_df["end_date"])]
    # future_df["date_year"] = future_df.apply(lambda r: str(r["date"]).split("-")[0], axis =1)
    # future_df["date_month"] = future_df.apply(lambda r: str(r["date"]).split("-")[1], axis =1)
    # future_df["instrument"] = future_df.apply(lambda r: extract_instrument_from_code(r["code"]), axis = 1)
    # future_df.to_csv("future_daily_df", index=False)

    # future_df = pd.read_csv("future_daily_df")
    # #print (future_df[:50])
    # self.daily_df = future_df

    # todo: 服务器数据库amount 变量， 为什么有些永远是0， 比如银， 铝？ 那个变量代表日成交金额？
    # # method 1 : avergae amount
    #
    # # 选择前n个月的平均交易额
    # time_range = 6
    # df = self.daily_df.copy()
    #
    # t0 = time.time()
    # df = df.groupby(["date_year", "date_month", "instrument"], as_index = False)["money"].agg([np.sum, np.size]).rename(columns = {"size": "size_money",
    #                                                                                                                                 "sum": "sum_money"}).reset_index()
    #
    # print("time usage 0 ", time.time() - t0)
    #
    # def cal_avg_amount(sub_df):
    #     sub_df = sub_df.sort_values(by=["date_year", "date_month"], ascending=[True, True])
    #     sub_df["size"] = sub_df["size_money"].rolling(time_range, min_periods=1).sum()
    #     sub_df["sum"] = sub_df["sum_money"].rolling(time_range, min_periods=1).sum()
    #     sub_df["avg_money"] = sub_df.apply(lambda r: r["sum_money"]/r["size_money"], axis =1)
    #     sub_df = sub_df.drop(["sum_money", "size_money", "size", "sum"], axis =1)
    #     return sub_df
    #
    # t1 = time.time()
    # df = df.groupby(["instrument"], as_index = False).apply(cal_avg_amount).reset_index()
    # #print(df.sample(20))
    # print("time usage 1 ", time.time() - t1)
    #
    # df["f_type"] = df.instrument.map(lambda x: self.future_type_dic[x] if x in self.future_type_dic.keys() else 1)
    # # todo: 给部分期货商品添加类型
    #
    # df = df[df["f_type"] != 1]
    #
    # df = df.groupby(["date_year", "date_month", "f_type"])["avg_money"].mean().reset_index()
    # print(df)
    # def sort_indi(sub_df):
    #     sub_df = sub_df.sort_values(by=["avg_money"], ascending=True)
    #     return sub_df
    # df = df.groupby(["date_year", "date_month"], as_index = False).apply(sort_indi).reset_index()
    # print(df)