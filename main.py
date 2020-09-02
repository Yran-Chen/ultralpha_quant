from strategy_pool import StrategyPool
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
# from backtest_params_pool import *
import time
from params_gen import *
from crosssec_param import gen_param, gen_param_2, gen_param_diff_industry, \
                            gen_param_hedge_pressure, gen_param_momentum, \
                            gen_param_term_structure, gen_param_open_interest, gen_param_term_structure2, gen_param_value,\
                            gen_param_skew, gen_param_liq, gen_param_inflation

pd.set_option('display.max_rows', None)


basic_params = {
'basic':param_basic_2005_2017,
}

fast_params = {
'basic':param_basic_fasttest,
}

fast_test = params_generator(fast_params,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10],rev_backtime_=[30])

basic_test = params_generator(basic_params,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])

# <<<<<<< HEAD
# from crosssec_param_2 import gen_param
# =======
# factor_weighted_par = params_generator(basic_params,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])
# >>>>>>> 8bbde757cf3c90fca58ddd03862a4e9e2cc9b705

if __name__ == "__main__":
    # t0  = time.time()

    ss = StrategyPool(gen_param_open_interest())
    ss.run()
    ss.evaluate()

# <<<<<<< HEAD
#     pool_crosssec = gen_param()
#     GA_pool = params_generator(pool_crosssec,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])
#
#     ss = StrategyPool(None)
#     # ss.run()
#     # ss.save_strategy_pool('..\\save_fut_strategy','1226_631')
#     # ss.evaluate()
#     # ss.plot()
#     ss.load_strategy_pool('..\\save_fut_strategy','1226_631')
#     print(ss.all_test_name)
#     # ss.strategy_return_forward = ss.strategy_return_forward.loc[ss.strategy_return_forward.index.year>=2012]
#
#     weights = ss.GA_portfolio_optimize(selected_strategy=ss.all_test_name,target="test_",loss_weight=[0.9,0.01,0.09])
#     print(weights)
#     weights.to_csv('GA_weights_1227_901.csv')
#
# =======
#     # print("total usage of time ", time.time() - t0)
#     # ss.plot()
#
#     # pool_crosssec = gen_param_diff_industry()
#     # GA_pool = params_generator(pool_crosssec,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])
#     #
#     # ss = StrategyPool(GA_pool)
#     # ss.run()
#
#     ss = StrategyPool(None)
#     ss.load_strategy_pool('..\\save_fut_strategy', '1226_631')
#     ss.GA_portfolio_optimize(selected_strategy=ss.all_test_name,target="Maximize_calmar_ratio")
#     # ss.GA_portfolio_optimize()
#
#     # ss.save_strategy_pool('..\\save_fut_strategy','GA_pool')
#
#     # test_pool = ss.strategy_return_forward.items()
#     # print(ss.strategy_return_forward)
#     # ss.strategy_return_forward = ss.strategy_return_forward.loc[ss.strategy_return_forward.index.year>=2012]
#
# #     ss = StrategyPool(fast_test)
# #     ss.run()
# #     ss.evaluate()
# #
# # >>>>>>> 92583cf35d3e1e4b060e0a466f5b70922ddbc91a
# #     # print("total usage of time ", time.time() - t0)
# #     # ss.plot()
# #
# #     pool_crosssec = gen_param_diff_industry()
# #     GA_pool = params_generator(pool_crosssec,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])
# #
# # <<<<<<< HEAD
# #     ss = StrategyPool(GA_pool)
# #     ss.run()
# #
# #     ss.save_strategy_pool('..\\save_fut_strategy','GA_pool')
# #     ss.load_strategy_pool('..\\save_fut_strategy','1226_631')
# #     # test_pool = ss.strategy_return_forward.items()
# #     # print(ss.strategy_return_forward)
# #     # ss.strategy_return_forward = ss.strategy_return_forward.loc[ss.strategy_return_forward.index.year>=2012]
# #     weights = ss.GA_portfolio_optimize(selected_strategy=ss.all_test_name,target="test_")
# #     print(weights)
# #     weights.to_csv('GA_weights.csv')
# >>>>>>> 8bbde757cf3c90fca58ddd03862a4e9e2cc9b705
#     # print(ss.strategy_return_forward)
#     # ss.evaluate()
# # =======
# #     # ss.save_strategy_pool('..\\save_fut_strategy','GA_pool')
# #
# #     # test_pool = ss.strategy_return_forward.items()
# #     # print(ss.strategy_return_forward)
# #     # ss.strategy_return_forward = ss.strategy_return_forward.loc[ss.strategy_return_forward.index.year>=2012]
# #
# #     # print(ss.strategy_return_forward)
# #     # ss.evaluate()
# # >>>>>>> 92583cf35d3e1e4b060e0a466f5b70922ddbc91a
