import os
import pandas as pd
import configparser
import operator

import matplotlib.pyplot as plt
import datetime

pd.set_option('display.max_columns', None)

def list_to_type(alist):
    alist = eval(alist)
    #print(isinstance(alist, list))
    #print( operator.eq(alist, ['ss', 'al', 'cu', 'pb', 'wr', 'zn', 'ni', 'hc', 'au', 'ag', 'sn']) )
    if operator.eq(alist, ['ss', 'al', 'cu', 'pb', 'wr', 'zn', 'ni', 'hc', 'au', 'ag', 'sn']):
        return "metal"
    elif operator.eq(alist, ['ru', 'sc', 'tc', 'zc', 'l', 'ta', 'pp', 'bu', 'ma', 'fu', 'me', 'eg', 'sp', 'nr', 'eb', 'ur']):
        return "prod&energy"
    elif operator.eq(alist, ['sr', 'c', 'cf', 'jd', 'cs', 'lr', 'wh', 'jr', 'pm', 'ap', 'cy', 'wt', 'er', 'cj', 'ri', 'ws', 'gn', 'rr']):
        return "grain"
    elif operator.eq(alist, ['i', 'jm', 'j', 'sm', 'sf']):
        return "black_material"
    elif operator.eq(alist, ['a', 'b', 'y', 'm', 'oi', 'p', 'rm', 'rs', 'ro']):
        return "oils&oilseeds"
    elif operator.eq(alist, ['fg', 'rb', 'fb', 'bb', 'v']):
        return "building_material"
    elif operator.eq(alist, ['t', 'ts', 'ih', 'if', 'tf', 'ic']):
        return "finan"
    else:
        return  "future pool " + str(len(alist))


config = configparser.ConfigParser()
res_dir= "..\\eval-diff-indi\\crosssec"
stra_files = os.listdir(res_dir)

for stra_file in stra_files:
    stra_path = os.path.join(res_dir, stra_file)
    res_files = os.listdir(stra_path)
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    for res_f in res_files:
        res_f_path = os.path.join(stra_path, res_f)
        if os.path.isfile(res_f_path):
            continue
        pnl_file = os.path.join(res_f_path, "daily_pnl.csv")
        param_file = os.path.join(res_f_path, "param.txt")

        df = pd.read_csv(pnl_file)
        config.read(param_file)
        gen_method = None
        # type = list_to_type(config["params"]["future_pool"])
        contract_label = "indi_"
        if config["params"]["is_main_contract_indi"] == "True":
            contract_label += "main_op_"
        else:
            contract_label += config["params"]["n_nearest_index_indi"] + "_op_"

        if config["params"]["is_main_contract_op"] == "True":
            contract_label += "main"
        else:
            contract_label += config["params"]["n_nearest_index_op"]


        # if config["params"]["fut_pool_gen_method"] == "dynamic":
        #     gen_method = "dynamic"
        # elif  config["params"]["fut_pool_gen_method"] == "static":
        #     alist = eval(config["params"]["future_pool"])
        #     gen_method = "static " + str(len(alist))

        df.index = df["date"]
        df.index = df.index.map(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d"))
        # print(type)
        freq = "M"
        # print(df[:10])
        df = df.groupby(pd.Grouper(freq=freq)).sum()
        df = df.cumsum()
        ax.plot(df, label=contract_label)
    plt.title(stra_file)
    plt.legend()
    plt.show()
    #plt.savefig("D://shanqi_yang//结果//新建文件夹//" + stra_file +".png")
