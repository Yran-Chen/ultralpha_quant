import pandas as pd
import datetime
from fastbacktest import FastBacktest
from crosssection_fastbacktest import FastCrossSectionalBacktest
from meanrev_backtest import MeanrevBacktest
from tesmom_backtest import TestMomBacktest
from backtest_params_pool import *
from Performance import Performance
from Paint import plot_all_daily_pnl
from Paint import plot_risk_analyze
from idiosyncratic_backtest import IdiosyncraticBacktest
from sklearn.linear_model import LinearRegression
import Parser_dataview
import numpy as np
import os
from portfopt import Portfopt
from copy import deepcopy
import json
import statsmodels.api as sm
import os
import errno
import codecs
import time
from performace_evaluator import Performace_evaluator
import datetime_process as dtp
from dateutil.parser import parse
from sklearn.preprocessing import PolynomialFeatures
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
from GA import GeneticAlgorithm
# from big_event import BigEventAnalyzer

class StrategyPool(FastBacktest):
    def __init__(self,params_pool):
        self.strategy = params_pool
        self.strategy_pool = {}
        self.strategy_return_forward = {}
        self.performance = {}
        self.exp_group = {}
        self.exp_return_forward = {}
        self.total_ret = None
        self.GA_weights = None

        if params_pool is not None:
            self.data_pool = FastBacktest(params = params_pool['basic'])
            self.data_proxy = self.data_pool.data_proxy
            self.strategy_preprocessing()

        self.all_test_name = []
        try:
            self.condition = params_pool['condition']
        except:
            self.condition = None
        self.filter_name = ['daily_pnl','sharpe','pnl','returns']

        self.parser = None
        self.filtered_performance = {}

    def strategy_preprocessing(self):
        for strategy_i,param_i in self.strategy.items():
            if strategy_i.startswith('meanrev'):
                self.strategy_pool[strategy_i] = MeanrevBacktest(param_i,self.data_proxy)
            elif strategy_i.startswith('crosssec'):
                self.strategy_pool[strategy_i] = FastCrossSectionalBacktest(param_i, self.data_proxy)
            elif strategy_i.startswith('tesmom'):
                self.strategy_pool[strategy_i] = TestMomBacktest(param_i, self.data_proxy)
            elif strategy_i.startswith('idiosyn'):
                self.strategy_pool[strategy_i] = IdiosyncraticBacktest(param_i, self.data_proxy)


    def run(self):
        counter = len(self.strategy_pool.keys()) - 1
        i = 0
        for test_name_i, backtest_i in self.strategy_pool.items():
            self.strategy_return_forward[test_name_i] = backtest_i.run()
            print('{} Finished. {} remained...'.format(test_name_i, counter))
            self.all_test_name.append(test_name_i)
            counter = counter - 1
            i += 1
            # if i == 1:
            #     break
            
    def risk_analyzer(self):
        RA = BigEventAnalyzer()
        tw_timeline = RA.return_trade_war_timeline()
        def nearest(items, pivot):
            return min(items, key=lambda x: abs(x - pivot))
        for (testname, method_name) in self.performance.keys():
            df = self.performance[(testname, method_name)].performance["daily_pnl"]
            df_date_index = df.index
            event_react_index = []
            for event_ in tw_timeline.index:
                # print(nearest(df_date_index,event_))
                event_react_index.append(nearest(df_date_index,event_))
            event_react_index = list(set(event_react_index))
            # print(event_react_index)
            event_re = df[event_react_index]
            # print(event_re)
            plot_risk_analyze(df,event_re)
            print(event_re/(self.performance[(testname, method_name)].booksize))

    # todo
    def eval_exp(self):
        for test_name_i,backtest_result_i in self.exp_return_forward.items():
            for method_name_i,res_i in backtest_result_i.items():
                    self.exp_group[(test_name_i,method_name_i)] = Performance(pnl=res_i)

    def evaluate(self):
        for test_name_i,backtest_result_i in self.strategy_return_forward.items():
            for method_name_i,res_i in backtest_result_i.items():
                self.performance[(test_name_i,method_name_i)] = Performance(res_dic=res_i)

#                 self.performance[(test_name_i,method_name_i)] = Performance(data_proxy=self.data_proxy,multi_alpha_weights=res_i, select_factor = method_name_i)
# >>>>>>> 92583cf35d3e1e4b060e0a466f5b70922ddbc91a

    
    def Calculate_Portfolio(self, selected_strategy=["tesmom","2nearest","term_structure","hedging_pressure"]):
        ret = deepcopy(self.strategy_return_forward)
        for test_name_i,backtest_result_i in deepcopy(ret).items():
            for method_name_i,return_i in backtest_result_i.items():
                if method_name_i not in selected_strategy:
                    del(ret[test_name_i][method_name_i])
            
        for test_name_i in deepcopy(ret).keys():
            if len(ret[test_name_i]) == 0:
                del(ret[test_name_i])
        
        self.selected_strategy_return_forward = ret
        x = Portfopt(ret)
        self.weights = x.maxiumum_sharpe_monthly()
        return self.weights
    
    def GA_portfolio_optimize(self,selected_strategy=["tesmom","2nearest","term_structure","hedging_pressure"],target="Mean_Variance",loss_weight = [1,1,1]):
        # print(selected_strategy)
        ret = deepcopy(self.strategy_return_forward)
        for test_name_i,backtest_result_i in deepcopy(ret).items():
            for method_name_i,return_i in backtest_result_i.items():
                if method_name_i not in selected_strategy:
                    del(ret[test_name_i][method_name_i])
            
        for test_name_i in deepcopy(ret).keys():
            if len(ret[test_name_i]) == 0:
                del(ret[test_name_i])
        
        self.selected_strategy_return_forward = ret

        flag=0
        for test_name_i,backtest_result_i in ret.items():
            for method_name_i,return_i in backtest_result_i.items():
                return_i = return_i.resample("M").sum()
                if flag==0:
                    total_ret = pd.DataFrame(return_i.values,index=return_i.index,columns=['{}_{}'.format(test_name_i,method_name_i)])
                    flag = 1
                elif flag==1:
                    return_i = pd.DataFrame(return_i.values,index=return_i.index,columns=['{}_{}'.format(test_name_i,method_name_i)])
                    total_ret = total_ret.join(return_i,how="outer")
        # print(total_ret)
        total_ret.to_csv('total_ret.csv')
        
        
        #上面的所有步骤都是为了将数据转化为合适的形式传入到优化函数中
        flag = 0
        start = time.clock()
        counter = 0
        total_len = len(total_ret) * 12 -72
        for i in range(1,len(total_ret)+1):
            s={}
            counter += min(i ,12)
            ret = total_ret.iloc[max(0,i-12):i,:]
            x = GeneticAlgorithm(ret)
            if target == 'test_':
                x.run_softmax(target=target,loss_weight=loss_weight)
            else:
                x.run(target)
            _, cleaned_weights = x.best_portfolio("None")
            s[total_ret.index[i-1]] = cleaned_weights
            # print(s)
            df = pd.DataFrame(s, index = ret.columns)
            print(df)
            if flag==0:
                final_weights = df
                flag=1
            else:
                final_weights = final_weights.join(df)

            elapsed = (time.clock() - start)
            total_time = (elapsed / (counter) * (total_len))
            print('Time processed remained : {:.2f}/{:.2f}'.format(elapsed, total_time))
        
        #得到权重配比的dataframe
        final_weights = final_weights.T
        self.weights = final_weights
        return self.weights
     
    
    def Strategy_Combined(self):
        #利用得到的资产配比将策略进行组合
        ret = deepcopy(self.selected_strategy_return_forward)
        flag=0
        for test_name_i,backtest_result_i in deepcopy(ret).items():
            for method_name_i,return_i in backtest_result_i.items():
                return_i = return_i.resample("M").sum()
                return_i = pd.DataFrame(return_i.values, index=return_i.index, columns=[method_name_i])
                if flag==0:
                    df = return_i
                    flag=1
                else:
                    df = df.join(return_i)

        x = df.columns
        string = "+".join(x)
        w_ = [1/len(x)]*len(x)
        w = []
        w.append(w_)
        w.append(w_)
        for j in range(1,len(df.index)-1):
            w_ = []
            for i in range(len(df.columns)):
                w_.append(self.weights.loc[df.index[j],x[i]])
#            print(df.index[j], w_)
            w.append(w_)
        xx = []
        for i in range(len(df)):
            xx.append(np.average(df.iloc[i,:],weights=w[i]))
#        print(xx)
        xx = pd.DataFrame(xx,index=df.index,columns=["combined"])
#        print(xx)
        
        ss = {"Combined":{string:xx}}
        dic = {}
        dic.update(ss)
        dic.update(self.strategy_return_forward)
        self.strategy_return_forward = dic

    def save_strategy_pool(self,save_path,name):
        data_to_store = {}
        abs_folder = os.path.abspath(save_path)
        data_path = os.path.join(save_path, 'DATA_{}.hd5'.format(name))
        for test_name_i,backtest_result_i in self.strategy_return_forward.items():
            for method_name_i,return_i in backtest_result_i.items():
                data_to_store['{}/{}'.format(test_name_i,method_name_i)] \
                    = self.strategy_return_forward[test_name_i][method_name_i]
        print("\nStore data...")
        self._save_h5(data_path, data_to_store)
        print("Stratrgy Pool has been successfully saved to:\n"
              + abs_folder + "\n\n"
              + "You can load it with load_strategy_pool('{:s}')".format(abs_folder))

    @staticmethod
    def _save_h5(fp,dic):
        import warnings
        warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

        StrategyPool.create_dir(fp)
        h5 = pd.HDFStore(fp, mode='w',complevel=9, complib='blosc')
        for key, value in dic.items():
            h5[key] = value
        h5.close()

    @staticmethod
    def create_dir(filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
                else:
                    print("I dont know why but it doesn't work.")

    def load_strategy_pool(self,load_path,name):
        path_data = os.path.join(load_path, 'DATA_{}.hd5'.format(name))
        if not ( os.path.exists(path_data)):
            raise IOError("There is no data file under directory {}".format(load_path))
        dic = StrategyPool._load_h5(path_data)
        for name,ret_i in dic.items():
            nd = {}
            # print(name.split('/'))
            test_name,method_name = name.split('/')[1:]
            nd[method_name] = ret_i
            self.strategy_return_forward[test_name] = nd
            if method_name not in self.all_test_name:
                self.all_test_name.append(method_name)
        print("StrategyPool loaded successfully.")


    @staticmethod
    def _load_h5(fp):
        h5 = pd.HDFStore(fp)
        res = dict()
        for key in h5.keys():
            res[key] = h5.get(key)
        h5.close()
        return res

    @staticmethod
    def read_json(fp):
        content = dict()
        try:
            with codecs.open(fp, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except IOError as e:
            if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
                raise
        # print(content)
        return content

    def plot_exp(self):
        plot_all_daily_pnl(self.exp_group)

    def plot(self):

        plot_all_daily_pnl(self.performance)

    def plot_filter(self,performance):
        plot_all_daily_pnl(performance)

    def filter(self):
        self.filtered_performance = self.performance.copy()
        for test_method_i,perform_i in self.performance.items():
            test_name = test_method_i[0]
            method_name = test_method_i[1]
            for condition_i in self.condition:
                filter_result = self.parser_eval(condition_i, perform_i.performance)
                filter_result = np.array(filter_result).all()
                print ('For Test:<{}> Method:<{}>, Condition <{}> eval...'.format(test_name,method_name,condition_i))
                print ('Eval result: {} '.format(filter_result))
                if not filter_result:
                    self.filtered_performance.pop(test_method_i)
        return self.filtered_performance

    def parser_eval(self, expression , factor_dict):
        if self.parser == None:
            print('Parser init...')
            self.parser = Parser_dataview.Parser()
        expr = self.parser.parse(expression)
        expr_parsed = expr.evaluate(factor_dict)
        return expr_parsed

    def save_profolio_weights(self,path,name):
        pass

    def load_portfolio_weights(self,path):
        self.GA_weights = pd.read_csv(os.path.join(path,'GA_weights.csv')).set_index(['Unnamed: 0']).sort_index(axis=1)
        self.GA_weights.index = self.GA_weights.index.rename('date')
        self.GA_weights.index = [parse(i) for i in self.GA_weights.index]

        self.total_ret = pd.read_csv(os.path.join(path,'total_ret.csv')).set_index(['date']).sort_index(axis=1)
        self.total_ret.index = [parse(i) for i in self.total_ret.index]

    def portfolio_tst(self,name,date=None):
        self.GA_weights_delay = self.GA_weights.shift(1)
        self.strategy_return_forward[name] = {'portfolio_best':self.cal_portfolio(self.GA_weights,self.total_ret)}
        self.exp_return_forward[name] = {'portfolio':self.cal_portfolio(self.GA_weights,self.total_ret,date)}
        self.strategy_return_forward['DELAY-{}'.format(name)] = {'portfolio_best':self.cal_portfolio(self.GA_weights_delay,self.total_ret)}
        self.exp_return_forward['DELAY-{}'.format(name)] = {'portfolio_best':self.cal_portfolio(self.GA_weights_delay,self.total_ret,date)}
        self.lrm_tst(name,date)

    def lrm_tst(self,name,date,delay=[30,100,180,240,300,360,540,700,860],target =['Sharpe'],pairs_=[1.0]):

        for de in delay:
            for ta in target:
                lrm_w = self.lrm_weights(self.GA_weights, self.total_ret, fixed_target=ta, delta=de)
                for pairsx in pairs_:
                    w_pair = lrm_w * pairsx + self.GA_weights_delay * (1-pairsx)
                    # print(w_pair)
                    self.strategy_return_forward['lrm.{}_{}_{}_{}'.format(name,de,ta,pairsx)] = {'portfolio':self.cal_portfolio(w_pair,self.total_ret)}
                    self.exp_return_forward['lrm.{}_{}_{}_{}'.format(name,de,ta,pairsx)] = {'portfolio': self.cal_portfolio(w_pair,self.total_ret,date)}

    def cal_portfolio(self,weight,ret,date = None):
        if date is None:
            return weight.multiply(ret).sum(axis=1)
        else:
            w_ = weight.loc[weight.index>=date]
            r_ = ret.loc[ret.index>=date]
            return w_.multiply(r_).sum(axis=1)

    def lrm_weights(self,weights_,portfolio_,fixed_target = 'Maximize_calmar_ratio',delta = 90):
        weights = deepcopy(weights_)
        portfolio = deepcopy(portfolio_)
        lrm_w = pd.DataFrame(index=weights.index,columns=weights.columns)

        def eval_score(dx, target,func = None):
            # print(dx)
            def ReLuFunc(x):
                x = (np.abs(x) + x) / 2.0
                return x
            def sigmoid(x):
                s = 1.0 / (1.0 + np.exp(-x))
                return s
            def softmax(x):
                dx = np.exp(x) / np.sum(np.exp(x), axis=0)
                return dx
            fit_box = []
            for j in (dx.columns):
                x = Performace_evaluator(None, pd.DataFrame(dx[j]))
                scores = x.run(target)
                fit_box.append(-scores[0])
            if (np.array(fit_box)==0.0).all():
                box_ret = np.zeros(len(dx.columns)).reshape(1,-1)
            else:
                box_ret = fit_box / np.absolute(fit_box).sum(axis=0)
                if func is None:
                    fit_box = (np.array(fit_box))
                elif func == 'softmax':
                    fit_box = softmax(np.array(fit_box))
                elif func == 'sigmoid':
                    fit_box = sigmoid(np.array(fit_box))
                box_ret =  fit_box / np.absolute(fit_box).sum(axis=0)
                if (np.array(fit_box) == 0.0).all():
                    box_ret = np.zeros(len(dx.columns)).reshape(1, -1)
            # print(box_ret)
            return box_ret

        # eval_score(portfolio)
        for date in weights.index[2:]:
            end_date = date
            last_date = weights[(weights.index < end_date)].index[-1]
            last_date_2 = weights[(weights.index < end_date)].index[-2]
            start_date = dtp.shift_time_(date, -delta)
            # Y_score = eval_score(weights.loc[(weights.index < end_date) & (weights.index >= start_date)])
            X_score = eval_score(portfolio.loc[(portfolio.index < end_date) & (portfolio.index >= start_date)].fillna(value=0),target=fixed_target)


            Y = weights.loc[(weights.index < last_date) & (weights.index >= start_date)].values
            X = portfolio.loc[(portfolio.index < last_date) & (portfolio.index >= start_date)].fillna(value=0).values

            # Y_ = Y * X_score
            # X_ = X / X_score

            his_y = weights.loc[(weights.index==last_date_2)].values.reshape(1,-1)
            his_x_sharpe = eval_score(portfolio.loc[(portfolio.index <= last_date_2) & (portfolio.index >= start_date)].fillna(value=0),target=fixed_target).reshape(1,-1)

            # print(his_y)
            # print(his_x_sharpe)
            # print('@@@@@@@@@@@@@@@@@@@')
            now_y = weights.loc[(weights.index == last_date)].values.reshape(-1,1)
            x_conc = np.stack((his_y,his_x_sharpe),axis=2).reshape(-1,2)
            # print(x_conc)
            # print('*******************')
            # print(now_y)

            # breakpoint()
            # ones = np.ones([len(X), 1])
            # X_ = np.append(ones, X, 1)
            # X = sm.add_constant(x_conc)
            # results = sm.OLS(Z, x_conc, missing='drop').fit()

            # lin_ba = LinearRegression()
            # lin_ba.fit(X, Y)
            #
            lin_ba = LinearRegression()
            lin_ba.fit(x_conc,now_y)

            # poly = PolynomialFeatures(degree=2)
            # X_poly = poly.fit_transform(X)
            # poly.fit(X_poly, Y)
            # lin_polo = LinearRegression()
            # lin_polo.fit(X_poly, Y)

            pre_raw = portfolio[portfolio.index==last_date].fillna(value=0).values
            pre_x_sharpe = eval_score(portfolio.loc[(portfolio.index <= last_date) & (portfolio.index >= start_date)].fillna(value=0),target=fixed_target).reshape(1,-1)
            pre_score = eval_score(portfolio[(portfolio.index < end_date) & (portfolio.index >= start_date)].fillna(value=0),target=fixed_target,func='sigmoid')
            # print(pre_raw * pre_score)
            # predictor_t = portfolio[portfolio.index==date].fillna(0).values
            # ones = np.ones([len(predictor_t), 1])
            # predictor_t = np.append(ones, predictor_t, 1)
            # pre_y = results.predict(predictor_t)
            pre_t = np.stack((pre_raw,pre_x_sharpe),axis=2).reshape(-1,2)

            pre_lin_y = (lin_ba.predict(pre_t).reshape(-1)) * (pre_score.reshape(-1))
            print(pre_lin_y)
            pre_y_norm_bva = pre_lin_y / np.abs(pre_lin_y).sum()
            # pre_y_norm =  pre_y / np.abs(pre_y).sum()
            # breakpoint()
            lrm_w.loc[date] = pre_y_norm_bva.reshape(-1)
        # print(lrm_w)
        return lrm_w
        # print(lrm_w)

if __name__ == "__main__":
    ss = StrategyPool(None)
    ss.load_strategy_pool('..\\save_fut_strategy', 'GA_pool')

    ss.load_portfolio_weights('..\\save_weights\\1226_631')
    ss.portfolio_tst('1226')
    ss.load_portfolio_weights('..\\save_weights\\1227_901')
    ss.portfolio_tst('1227')
    ss.evaluate()
    ss.plot()

    # ss.run()
    # ss.GA_portfolio_optimize(selected_strategy=ss.all_test_name,target="Maximize_calmar_ratio")
    # ss.Strategy_Combined()
##  x = Portfopt(ss.strategy_return_forward)
##  print(x.maxiumum_sharpe_monthly())
    # ss.evaluate()
    # ss.plot()

#    ss.filter()