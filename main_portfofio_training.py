from strategy_pool import StrategyPool
import pandas as pd
import time
from params_gen import *

basic_all_time = {
    'basic':param_basic_alltime,
}

basic_params = {
'basic':param_basic_2005_2017,
}
fast_params = {
'basic':param_basic_fasttest,
}

fast_test = params_generator(fast_params,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10],rev_backtime_=[30])
basic_test = params_generator(basic_params,weight_method_=['equal'],method_=['e_iskew'],qcut_=[10,20],rev_backtime_=[30])


if __name__ == "__main__":
    # ss = StrategyPool(basic_all_time)
    # ss.run()

    ss = StrategyPool(None)

    ss.load_strategy_pool('..\\save_fut_strategy', '1226_631')


    ss.load_portfolio_weights('..\\save_weights\\1226_631')
    ss.portfolio_tst('1226','2017-01-01')
    ss.eval_exp()
    ss.plot_exp()


    ss.evaluate()
    ss.plot()
