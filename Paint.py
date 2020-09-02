import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.api import qqplot
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


def plot_daily_pnl(
     data_forward,
     period=1, overlap=True, ax=None,
     save_dir = None,freq='M'
):
    df = data_forward.copy()
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    df = df.groupby(pd.Grouper(freq=freq)).sum()

    df = df.cumsum()
    ax.plot(df.index, df.values)
    ax.set_xlabel("date")
    ax.set_ylabel("daily_pnl")
    if save_dir is None:
        plt.show()
    else:
        fig.savefig(save_dir)
    plt.close()

def plot_risk_analyze(df,ra):
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    # freq = "M"
    # df = df.groupby(pd.Grouper(freq=freq)).sum()
    # df = df.cumsum()
    # ax.plot(df.index, df.values)
    ra_up = ra[ra>=0]
    ra_down = ra[ra<0]
    ax.bar(ra_up.index, ra_up.values,color='green',width=2)
    ax.bar(ra_down.index, ra_down.values,color='red',width=2)
    ax.legend()
    ax.set_xlabel("date")
    ax.set_ylabel("performance")
    plt.show()

def plot_all_daily_pnl(performance_dic):

    plt.close()
    fig = plt.figure(figsize=(12, 9))
    #ax = plt.subplot(111)
    ax = fig.add_subplot(111)
    colormap = plt.cm.nipy_spectral  # I suggest to use nipy_spectral, Set1,Paired
    ax.set_prop_cycle(color = [colormap(i) for i in np.linspace(0, 1, len(performance_dic))])

    for (testname,method_name) in performance_dic.keys():
        df = performance_dic[(testname, method_name)].performance["daily_pnl"]
        freq = "M"
        df = df.groupby(pd.Grouper(freq=freq)).sum()
        df = df.cumsum()
    #    plt.plot(df.index, df.values, label = testname+"_"+method_name)
        if method_name.startswith('portfolio'):
            ax.plot(df.index, df.values, label=testname + "_" + method_name,linewidth = 2.7)
        else:
            ax.plot(df.index, df.values, label = testname + "_" + method_name)

    ax.legend()
    ax.set_xlabel("date")
    ax.set_ylabel("daily_pnl")
    plt.show()

def plot_risk(df,save_dir=None):
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax.plot(df.index, df.values)
    ax.set_xlabel("date")
    ax.set_ylabel("view time")
    if save_dir is None:
        plt.show()
    else:
        fig.savefig(save_dir)
    plt.close()

def plot_annual_guideline(
    guideline,name,if_show=True,period = 1
):
    guideline_q = guideline.copy()
    x = list(guideline_q.index)
    y = list(guideline_q)
    name_ = name.split('_')[-1]

    start_date = guideline_q.index.min()
    end_date = guideline_q.index.max()
    plt.title('{}(annually).'.format(name_))
    plt.plot(x, y, color='skyblue', label='{}_{}_to_{}'.format(name_,start_date,end_date))

    plt.xlabel('date')
    plt.ylabel('{}'.format(name_))
    plt.savefig("D:\!Ultralpha\!MyQuant\demo\{}.png".format(name))
    if if_show:
        plt.show()
    return [x,y],name

def plot_cumsum_pnl(pnl,name,if_show=True,period=1,freq='M'):
    pnl_q = pnl.copy()
    pnl_q = pnl_q.groupby(pd.TimeGrouper(freq = freq)).sum()
    pnl_q = pnl_q.cumsum()
    x = list(pnl_q.index)
    y = list(pnl_q)
    start_date = pnl_q.index.min()
    end_date = pnl_q.index.max()
    plt.title('Pnl(monthly).')
    plt.plot(x, y, color='skyblue', label='Pnl_{}_to_{}'.format(start_date,end_date))
    plt.xlabel('date')
    plt.ylabel('month_pnl')
    plt.savefig("..\\demo\\{}.png".format(name))
    if if_show:
        plt.show()
    plt.clf()
    return [x,y],name

def compare_plot(x,y,if_show=True):
    x_name = x['name']
    x_group = x_name.split('_')[-1]
    x_data = x['data']
    y_name = y['name']
    y_group = y_name.split('_')[-1]
    y_data = y['data']

    plt.title('{}_CompareWith_{}'.format(x_group,y_group))

    plt.plot(x_data[0],x_data[1],color='blue',label='{0}_{1:.2f}'.format(x_group,sum(x_data[1])))
    plt.plot(y_data[0],y_data[1],color='red',label='{0}_{1:.2f}'.format(y_group,sum(y_data[1])))
    plt.legend()

    plt.savefig("D:\!Ultralpha\!MyQuant\demo\{}_{}.png".format(x_name.split('_')[1],'{}_CompareWith_{}'.format(x_group,y_group)))
    if if_show:
        plt.show()
    plt.clf()
