# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:31:04 2019

@author: ultralpha
"""
import pandas as pd
import numpy as np
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt import value_at_risk
# from pypfopt import hierarchical_risk_parity
import Performance

class Portfopt():    
    def __init__(self,returns):
        flag = 0
        for test_name_i,backtest_result_i in returns.items():
            for method_name_i,return_i in backtest_result_i.items():
#                print(test_name_i)
#                print(method_name_i)
#                print(return_i)
                return_i = return_i.resample("M").sum()
                if flag==0:
                    total_ret = pd.DataFrame(return_i.values,index=return_i.index,columns=[method_name_i])
                    flag = 1
                elif flag==1:
                    return_i = pd.DataFrame(return_i.values,index=return_i.index,columns=[method_name_i])
                    total_ret = total_ret.join(return_i,how="outer")
        
        self.returns = total_ret
        print(self.returns)
        self.begin_year = self.returns.index[0].strftime("%Y-%m-%d")[:4]
        self.end_year = self.returns.index[-1].strftime("%Y-%m-%d")[:4]

    def maximum_sharpe(self):
        s={}
        for i in range(int(self.begin_year),int(self.end_year)+1):
            ret = self.returns[str(i)].sum()
            cov = self.returns[str(i)].cov()
            ef = EfficientFrontier(ret, cov)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            print(cleaned_weights)
            s[str(i)]= cleaned_weights
        return s
    
    def maxiumum_sharpe_monthly(self):
        flag = 0
        for i in range(1,len(self.returns)+1):
            s={}
            ret = self.returns.iloc[max(0,i-12):i,:]
            ret_s = ret.sum()
            cov = ret.cov()
            ef = EfficientFrontier(ret_s, cov)
            raw_weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            s[self.returns.index[i-1]] = cleaned_weights
            df = pd.DataFrame(s)
            if flag==0:
                final_weights = df
                flag=1
            else:
                final_weights = final_weights.join(df)
        
        final_weights = final_weights.T
        return final_weights
    
    def minimize_cVaR(self):
        s={}
        for i in range(int(self.begin_year),int(self.end_year)+1):
            ret = self.returns[str(i)]
            ret = ret.dropna()
            x = value_at_risk.CVAROpt(ret)
            s[str(i)] = x.min_cvar()
        return s
    
    def hierarchical_risk_parity(self):
        s = {}
        for i in range(int(self.begin_year),int(self.end_year)+1):
            ret = self.returns[str(i)]
            ret = ret.dropna()
            x = hierarchical_risk_parity.HRPOpt(ret)
            s[str(i)] = x.hrp_portfolio()
        return s
    
if __name__ == "main":
    s = Portfopt()
    