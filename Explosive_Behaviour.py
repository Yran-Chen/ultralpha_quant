
from math import floor
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts


class exploBehav:
    def __init__(self, price, cpi = 1 ):

        self.nprice = price
        self.ncpi = cpi
        self.lprice = np.log(self.nprice/self.ncpi)
        self.df = pd.DataFrame(self.lprice)
        self.df.rename(columns = {0: ' lprice'}, inplace = True)



        self.len = self.lprice.shape[0]


        self.acv = np.nan #self.adf1(10000, self.len)
        self.scv = np.nan #self.sadf(10000, self.len)

        self.radfTS = np.nan
        self.fadfTS = np.nan
        self.sadfTS = np.nan
        self.adf1TS = np.nan

        # For convenience here, I directly set this as a fixed value
        self.swindow77 = 77
        self.swindow39 = 39

    def adf1CV(self, m = 10000, t=None):

        ''' estimate the adf critical value with a simulation of m samples with t steps'''

        t = self.len if t is None else t



        y = self.dgp(m, t)  # m steps, t ts(paths)
        deltas = np.zeros(m)

        for i in range(m):

            deltas[i] = ts.adfuller(y[i, :], maxlag=0, regression='c', autolag=None, regresults=False)[0]

        result = {'0.90': np.quantile(deltas, 0.90, axis=0), '0.95': np.quantile(deltas, 0.95, axis=0),
                  '0.99': np.quantile(deltas, 0.99, axis=0), '0.10': np.quantile(deltas, 0.10, axis=0),
                  '0.05': np.quantile(deltas, 0.95, axis=0), '0.01': np.quantile(deltas, 0.01, axis=0)}

        self.acv = result


        self.df.loc[:, 'adf1CV 0.99'] = np.full(self.len, self.acv['0.99'])
        self.df.loc[:, 'adf1CV 0.95'] = np.full(self.len, self.acv['0.95'])
        self.df.loc[:, 'adf1CV 0.90'] = np.full(self.len, self.acv['0.90'])
        self.df.loc[:, 'adf1CV 0.10'] = np.full(self.len, self.acv['0.10'])
        self.df.loc[:, 'adf1CV 0.05'] = np.full(self.len, self.acv['0.05'])
        self.df.loc[:, 'adf1CV 0.01'] = np.full(self.len, self.acv['0.01'])
        return self.acv


    def sadfCV(self, m = 10000, t = None):

        t = self.len if t is None else t

        qe = np.array([0.90, 0.95, 0.99])
        r0 = 0.01 + 1.8 / sqrt(t)
        swindow0 = floor(r0 * t)
        y = self.dgp(t, m)  # genr a txm matrix, with m indepndent time series(paths) data and t steps in each time series
        badfs = np.zeros([t - swindow0 + 1, m])  # t-swindow0 <- interval between swindow0 and t (r0 to 1), m paths
        '''
                     1, 2, 3, ......, m-1, m
        swindow0     .  .  .  .  .  .  .   .
        swindow0+1   .  .  .  .  .  .  .   .
            .        .  .  .  .  .  .  .   .====> this shows the positions for the m ts at each time point between r0(swindow0) and 1(t)
            .        .  .  .  .  .  .  .   .
            .        .  .  .  .  .  .  .   .
            t        .  .  .  .  .  .  .   .
        '''
        sadf = np.empty([m, 1])

        for j in range(0, m):  # for each path
            for i in range(swindow0, t + 1):  # at each time point
                badfs[i - swindow0, j] = ts.adfuller(y[0:i, j], maxlag=0, regression='c', autolag=None, store=False,
                                                     regresults=False)[0]


        sadf[:, 0] = np.amax(badfs, axis=0)
        result = {'0.90': np.quantile(sadf, qe[0], axis=0)[0], '0.95': np.quantile(sadf, qe[1], axis=0)[0],
                  '0.99': np.quantile(sadf, qe[2], axis=0)[0]}

        self.scv = result

        self.df.loc[:, 'sadfCV 0.99'] = np.full(self.len, self.scv['0.99'])
        self.df.loc[:, 'sadfCV 0.95'] = np.full(self.len, self.scv['0.95'])
        self.df.loc[:, 'sadfCV 0.90'] = np.full(self.len, self.scv['0.90'])

        return self.scv



    def radf(self, y = None, fixedlag = 12, regression = 'c', window = 77, autolag = None, regresults = False):
        '''generate rolling adf test statistic with a fixed window size'''

        y = self.lprice if y is None else y

        nrecur = len(y) - window + 1
        slp = np.zeros(nrecur)

        for i in range(0, nrecur):
            slp[i] = ts.adfuller(y.iloc[i:i + window], maxlag=fixedlag, regression=regression, autolag=autolag,
                                 regresults=regresults)[0]
        self.radfTS = slp

        self.df.loc[:, 'radfTS'] = np.concatenate((np.full(window - 1, np.nan), self.radfTS))

        return self.radfTS

    def fadf(self, y = None, fixedlag=12, regression='c', window=39, autolag=None, regresults=False):
        ''' generate a series adf test stastics and return sup adf statistics
            with a growing sample size(forward looking)'''

        y = self.lprice if y is None else y

        nrecur = len(y) - window + 1
        slp = np.zeros(nrecur)

        for i in range(0, nrecur):
            slp[i] = ts.adfuller(y.iloc[0:i + window], maxlag=fixedlag, regression=regression, autolag=autolag,
                                 regresults=regresults)[0]

        self.fadfTS = slp
        self.sadfTS = np.amax(slp)

        self.df.loc[:, 'sadfTS'] = np.full(self.len, self.sadfTS)
        self.df.loc[:, 'fadfTS'] = np.concatenate((np.full(window - 1, np.nan), self.fadfTS))

        return self.fadfTS, self.sadfTS

    def adf1(self, y = None, fixedlag = 12, regression ='c', autolag = None, regresults = False):
        ''' generate the ADF test statistics using the full sample '''

        y = self.lprice if y is None else y

        result = ts.adfuller(y, maxlag = fixedlag, regression = regression, autolag = autolag, regresults = regresults)[0]

        self.adf1TS = result
        self.df.loc[:, 'adf1TS'] = np.full(self.len, self.adf1TS)

        return self.adf1TS


    def dgp(self, n, niter):
        '''generate niter time series with n steps in each time series'''

        u0 = 1 / n
        rn = np.random.normal(0, 1, size=(n, niter))  # generate a n x niter matrix
        z = rn + u0
        y = np.cumsum(z, axis=0)  # each column is an independent observation (time series)
        return y


    def Graph(self, quantile = '0.95'):

        RG = self.df.loc[:, ['lprice', 'adf1CV 0.99', 'adf1CV 0.95', 'adf1CV 0.90', 'sadfCV 0.99', 'sadfCV 0.95', 'sadfCV 0.90',
                         'radfTS', 'sadfTS', 'fadfTS', 'adf1TS']]
        RG.plot()
        plt.show()




        plt.show()
    def summary(self):
        print('''The Sup ADF Test Statistics is: {0}\nThe ADF1 Test Statistics is: {1}\nThe ADF1 Criti Val is: {2}\nThe Sup ADF Crit Val is: {3}
        '''.format(self.sadfTS, self.adf1TS, self.acv, self.scv))



if __name__ == "__main__":
    nominal_nasdaq = pd.read_csv('C:\\Users\\CRZ\\Desktop\\William\\IXIC.csv', index_col=0)['Close']
    cpi = pd.read_csv('C:\\Users\\CRZ\\Desktop\\William\\CPIAUCSL.csv', index_col=0)['CPIAUCSL']
    real_nasdaq = nominal_nasdaq / cpi
    n = np.log(real_nasdaq)

    A = exploBehav(nominal_nasdaq, cpi)
    A.adf1CV(m=2000) # compute the data desired
    A.sadfCV(m=2000)
    A.radf()
    A.fadf()
    A.adf1()
    print('Finish All Computation')
    A.Graph()
    # A.df.to_csv("path_to_save_outputs")


