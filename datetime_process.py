from datetime import datetime
from datetime import date
from datetime import timedelta
import pandas as pd
import calendar
import time

monthDict = \
    {	'01':'Janauary',
        '01':'January',
		'02':'February',
		'03':'March',
		'04':'April',
		'05':'May',
		'06':'June',
		'07':'July',
		'08':'August',
		'09':'September',
		'10':'October',
		'11':'November',
		'12':'December'		}


def date_trans(datestr):
    # format - June 29, 2019
    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]
    # print(datestr)
    # print('@@@@@@@@@@@@@@@')
    month,date,year = datestr.split()
    date = date.strip(',')
    month = get_key(monthDict,month)[0]
    return str2date('{}-{}-{}'.format(year,month,date))

def return_date_ofyear(date):
    return

def date2datetime(date,format='%Y-%m-%d'):
    return datetime.strptime(str(date),format)

def date_ranging(beginDate,endDate,freq='D'):
    return pd.date_range(start=beginDate, end=endDate,freq=freq)

def date2str(date, format='%Y-%m-%d'):
    # print(date)
    return pd.to_datetime(date).strftime(format)

def str2date(date,format='%Y-%m-%d'):
    # print(datetime.strptime(date,format))
    # print(date)
    if type(date) == str:
        return datetime.strptime(date,format)
    else:
        return date

def generate_month_end(df):
    trad_days = pd.DataFrame(index=df.index,columns=df.columns)
    datelist = df.index.levels[0]
    pix = multi_dataframe_to_monthly(df)
    datelist_2return = []
    # dx = multiindex_2dateindex(df.copy())
    for year,month in pix:
        _, monthCountDay = calendar.monthrange(year, month)
        min_date = date(year=year, month=month, day=1)
        max_date = date(year=year, month=month, day=monthCountDay)
        pointer = ' date>="{}" and date<="{}" '.format(min_date, max_date)
        to_x = df.query(pointer).dropna()
        if len(to_x)!= 0:
            p = to_x.iloc[-1].name[0]
            datelist_2return.append(p)
        # trad_days = trad_days.append(df.loc[(p,slice(None)),:])
    return datelist_2return

def generate_trading_days(df):
    trad_days = pd.DataFrame()
    datelist = df.index.levels[0]
    pix = multi_dataframe_to_monthly(df)
    for year,month in pix:
        _, monthCountDay = calendar.monthrange(year, month)
        min_date = date(year=year, month=month, day=1)
        max_date = date(year=year, month=month, day=monthCountDay)
        pointer = ' date>="{}" and date<="{}" '.format(min_date, max_date)
        to_x = df.query(pointer).dropna()
        if len(to_x)!= 0:
            p = to_x.iloc[-1].name[0]
            trad_days = trad_days.append(df.loc[(p,slice(None)),:])
    # print(trad_days.dropna())
    return trad_days


def multiindex_2dateindex(df,col='code'):
        return df.reset_index(['date',col]).set_index(['date']).sort_index(axis=0)

def dateindex_2multiindex(df,col='code'):
        return df.reset_index(['date']).set_index(['date',col]).sort_index(axis=0)

def shift_time_(date,delta,format='%Y-%m-%d'):
    if type(date) == str:
        return datetime.strptime(date,format)+timedelta(days=delta)
    else:
        return date+timedelta(days=delta)

def datelist(daterange):
    date_l = [datetime.strftime(x, '%Y-%m-%d') for x in list(daterange)]
    return date_l

def multi_dataframe_to_monthly(datestocks):
    datelist_ = list(datestocks.index.levels[0])
    pix =  date_ranging(min(datelist_), max(datelist_), 'M')
    def year_month(date_):
        return (date_.year,date_.month)
    pix = list(map(year_month,pix))
    return pix

# print(pd.date_range('2015-01-01','2017-03-04','D'))
# print(datelist(date_ranging('2015-01-01','2017-03-04','M')))

def date_pointer(df,year,month,day=None):
    _, monthCountDay = calendar.monthrange(year, month)
    min_date = date(year=year, month=month, day=1)
    max_date = date(year=year, month=month, day=monthCountDay)
    pointer = ' date>="{}" and date<="{}" '.format(min_date,max_date)
    return df.query(pointer)

if __name__=='__main__':
    # pd_index = date_ranging('2015-01-01','2017-03-04','5D')
    # print(pd_indexd)
    # for i in pd_index:
    #     print(i.year,i.month)
    # dt = date(2010, 1, 1)
    # print(date2datetime(dt))
    print(date_trans('June 26, 2019'))



    # print(date2str(shift_time_('2015-01-01',30)))
        # print(i.month)
    # a = date.today()
    # print(a.month)