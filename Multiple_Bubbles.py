from math import floor
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts


class PSY:
    ''' demonstrate the research results of article "Multiple Bubbles", published by PSY with new methods such as BADF,
    BSADF, and GSADF

    BADF ->
    BSADF -> an array of BSADF values for each different r2



    '''

    def __init__(self, price):
        ''' price: inflation adjusted price series'''

        self.nprice = price

        self.df = pd.DataFrame({'pd':self.nprice})


        self.len = self.df.shape[0]


        self.gscv = np.nan #
        self.bscv = np.nan #
        self.scv = np.nan

        self.gsadfTS = np.nan #
        self.badfTS = np.nan #
        self.bsadfTS = np.nan #
        self.sadfTS = np.nan #

        self.swindow0 = self.len * ( 0.01 + 1.8/sqrt(self.len))



    def bsadfCV(self, m = 2000, t = None, maxlag=0, regression='c', autolag = None):
        ''' return the bsadf critical values sequence together with the gsadf critical values
            m: the number of simulation desired'''

        t = self.len if t is None else t

        qe = np.array([0.90, 0.95, 0.99])
        r0 = 0.01 + 1.8 / sqrt(t)
        swindow0 = floor(r0 * t)
        msadfs = np.zeros([m, t - swindow0 + 1])
        gsadf = np.full(m, np.nan)


        y = self.dgp(t, m)

        for j in range(m):

            bsadfs = np.full(t, np.nan)

            for r2 in range(swindow0, t + 1):
                rwadft = np.full(r2 - swindow0 + 1, np.nan)

                for r1 in range(0, r2 - swindow0 + 1):
                    rwadft[r1] = ts.adfuller(y[r1:r2, j], maxlag = maxlag, regression = regression, autolag = autolag)[0]

                bsadfs[r2 - 1] = np.amax(rwadft)

            bsadfs = bsadfs[swindow0 - 1:] # remove nan values

            msadfs[j, :] = bsadfs
            gsadf[j] = np.amax(bsadfs)


        quantile_badfs = np.zeros([len(qe), msadfs.shape[1]])

        for i in range(0, msadfs.shape[1]):
            quantile_badfs[:, i] = np.quantile(msadfs[:, i], qe)

        quantile_gsadf = np.quantile(gsadf, qe)


        result_bsadf = {'0.90': quantile_badfs[0], '0.95': quantile_badfs[1], '0.99': quantile_badfs[2]}
        result_gsadf = {'0.90': quantile_gsadf[0], '0.95': quantile_gsadf[1], '0.99': quantile_gsadf[2]}

        self.bscv, self.gscv = result_bsadf, result_gsadf

        bscv = np.concatenate((np.full([3, swindow0 - 1], np.nan), quantile_badfs), axis = 1)
        gscv90 = np.full(self.len, self.gscv['0.90'])
        gscv95 = np.full(self.len, self.gscv['0.95'])
        gscv99 = np.full(self.len, self.gscv['0.99'])


        self.df.loc[:, 'bscv 0.90'] = bscv[0, :].astype(float)
        self.df.loc[:, 'bscv 0.95'] = bscv[1, :].astype(float)
        self.df.loc[:, 'bscv 0.99'] = bscv[2, :].astype(float)

        self.df.loc[:, 'gscv 0.90'] = gscv90.astype(float)
        self.df.loc[:, 'gscv 0.95'] = gscv95.astype(float)
        self.df.loc[:, 'gscv 0.99'] = gscv99.astype(float)

        # Values -> the bsadfs array for each different r2
        # Quantiles -> the 99%, 95%, 90% quantile values of bsadf from m samples.
        '''
        MSADFS
                r2 = swindow0   r2 = swindow1  ...  r2 = t
        m = 0     sadf|m, r2      sadf|m, r2   ...sadf|m, r2    
        m = 2     sadf|m, r2      sadf|m, r2   ...sadf|m, r2
          .       sadf|m, r2      sadf|m, r2   ...sadf|m, r2  
          .       sadf|m, r2      sadf|m, r2   ...sadf|m, r2  
          .       sadf|m, r2      sadf|m, r2   ...sadf|m, r2  
        m = m     sadf|m, r2      sadf|m, r2   ...sadf|m, r2  

        QUANTILES
                r2 = swindow0   r2 = swindow1  ...  r2 = t

         99%    sadf|m, r2      sadf|m, r2   ...sadf|m, r2

         95%    sadf|m, r2      sadf|m, r2   ...sadf|m, r2

         90%    sadf|m, r2      sadf|m, r2   ...sadf|m, r2
        '''

        return self.bscv, self.gscv


    def sadfCV(self, m = 2000, t = None):
        ''' Return the sadf test statistic value
            m: the number of simulation desired'''

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
                badfs[i - swindow0, j] = ts.adfuller(y[0:i, j], maxlag = 0, regression='c', autolag=None, store=False,
                                                     regresults=False)[0]


        sadf[:, 0] = np.amax(badfs, axis=0)
        result = {'0.90': np.quantile(sadf, qe[0], axis=0)[0], '0.95': np.quantile(sadf, qe[1], axis=0)[0],
                  '0.99': np.quantile(sadf, qe[2], axis=0)[0]}
        self.scv = result

        scv90 = np.full(self.len, self.scv['0.90'])
        scv95 = np.full(self.len, self.scv['0.95'])
        scv99 = np.full(self.len, self.scv['0.99'])

        self.df.loc[:, 'scv 0.90'] = scv90.astype(float)
        self.df.loc[:, 'scv 0.95'] = scv95.astype(float)
        self.df.loc[:, 'scv 0.99'] = scv99.astype(float)

        return self.scv


    def sadf(self, maxlag = 0, y=None, regression='c', autolag='AIC', regresults=False):
        ''' return the SADF (Scalar) and BADF (Sequence)  test statistics '''

        y = self.df['pd'] if y is None else y

        t = self.len
        r0 = 0.01 + 1.8 / sqrt(t)
        swindow0 = floor(r0 * t)
        dim = t - swindow0 + 1
        badfs = np.zeros([t - swindow0 + 1])  # nd.array
        for i in range(swindow0, t + 1):
            badfs[i - swindow0] = \
                ts.adfuller(y.iloc[0:i + 1], maxlag=maxlag, regression=regression, autolag=autolag, regresults=regresults)[0]  # an array of values

        sadf = np.amax(badfs)  # single value

        result = {'badfs': badfs, 'sadf': sadf}
        self.badfTS, self.sadfTS, = result['badfs'], result['sadf']

        badfs = np.concatenate((np.full(swindow0 - 1, np.nan), badfs))
        self.df.loc[:, 'badf TS'] = badfs.astype(float)

        sadf = np.full(self.len, self.sadfTS)
        self.df.loc[:, 'sadf TS'] = sadf.astype(float)

        return self.badfTS, self.sadfTS


    def gsadf(self, maxlag = 0, y = None, regression = 'c', autolag = None, regresults = False):
        ''' Return the GSADF (Sequence) and BSADF (Scalar) test statistics'''

        y = self.df['pd'] if y is None else y
        t = self.len
        swindow0 = floor(t * (0.01 + 1.8/sqrt(t)))

        bsadfs = np.full(t, np.nan)

        for r2 in range(swindow0, t + 1):
            rwadft = np.zeros(r2 - swindow0 + 1)

            for r1 in range(0, r2 - swindow0 + 1): # 0~82
                rwadft[r1] = ts.adfuller(y.iloc[r1:r2], maxlag = maxlag, regression = regression, autolag = autolag, regresults = regresults)[0]

            bsadfs[r2 - 1] = np.amax(rwadft)

        bsadfs = bsadfs[swindow0 - 1:]

        gsadf = np.amax(bsadfs)

        self.bsadfTS, self.gsadfTS = bsadfs, gsadf

        bsadfs = np.concatenate((np.full(swindow0 - 1, np.nan), bsadfs))
        self.df.loc[:, 'bsadf TS'] = bsadfs

        gsadf = np.full(self.len, self.gsadfTS)
        self.df.loc[:, 'gsadf TS'] = gsadf.astype(float)

        return self.bsadfTS, self.gsadfTS


    def dgp(self, n, niter):
        '''generate niter time series with n steps in each time series'''
        np.random.seed(101)
        u0 = 1 / n
        rn = np.random.normal(0, 1, size=(n, niter))  # generate a n x niter matrix
        z = rn + u0
        y = np.cumsum(z, axis=0)  # each column is an independent observation (time series)
        return y

    def locate(self):
        ''' Locate the Start and End time of  Explosive Behaviours'''
        TS = self.df['bsadfTS']
        CV = self.df['bscv 0.95']
        pwindow = np.floor(np.log(self.len))
        isHigher = TS.iloc[self.swindow0:] >= CV.iloc[self.swindow0:]
        cumulative = np.zeros(self.len - self.swindow0).astype(int)

        for i in range(self.len - self.swindow0):
            if isHigher.iloc[i] == True:
                cumulative[i] = cumulative[i - 1] + 1

        idx = self.len - self.swindow0 - 1
        C1 = np.full(self.len - self.swindow0, False)

        while idx >= 0:
            if cumulative[idx] >= pwindow:
                C1[(int(idx - cumulative[idx])): idx] = True
                idx = idx - cumulative[idx]
            else:
                idx = idx - 1

        C1 = np.concatenate((np.full(self.swindow0, np.nan), C1))

        self.df.loc[:, 'isExplosive'] = C1.astype(float)

    def gsadfGraph(self):
        self.df.plot()
        plt.show()


if __name__ == "__main__":
    shiller = pd.read_csv('C:\\Users\\CRZ\\Desktop\\ie_data.csv', index_col=0)
    shiller['pdratio'] = shiller['Price'] / shiller['Dividend']

    A = PSY(shiller['pdratio'])
    A.gsadf(maxlag=0)
    A.sadf(maxlag=0)
    A.bsadfCV(m=1000)
    A.sadfCV(m=1000)
    # A.df.to_csv('Path_to_Save')




