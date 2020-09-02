import datetime
import calendar
import json
import requests
import math

def get_half_year_ago(year, month):
    month -= 6
    if month <= 0:
        month += 12
        year -= 1
    return year, month

# Input: start daytime and end daytime, format: datetime.date
# Return: a list of existing months between this day time
def get_month_range(start_day,end_day):
  months = (end_day.year - start_day.year)*12 + end_day.month - start_day.month
  month_range = ['%s-%s'%(start_day.year + mon//12,mon%12+1)
                    for mon in range(start_day.month-1,start_day.month + months)]
  return month_range

def get_last_day_per_month(year, month):
    start_date = datetime.date(year, month, 1)
    _, days_in_months = calendar.monthrange(year, month)
    end_date = start_date + datetime.timedelta(days=days_in_months - 1)
    return end_date

def get_next_month(year, month):
    start_date = datetime.date(year, month, 1)
    _, days_in_months = calendar.monthrange(year, month)
    end_date = start_date + datetime.timedelta(days=days_in_months + 1)
    return end_date.year, end_date.month


# 输入当前月的某一天，输出下月月末时间
def get_next_month_end(cur_date):
    cur_y, cur_m = cur_date.year, cur_date.month
    next_y, next_m = get_next_month(cur_y, cur_m)
    start_date = datetime.datetime(next_y, next_m, 1)
    _, days_in_months = calendar.monthrange(next_y, next_m)
    end_date = start_date + datetime.timedelta(days=days_in_months-1)
    return end_date

def get_last_month(year, month):
    start_date = datetime.date(year, month, 1)
    end_date = start_date + datetime.timedelta(-7)
    return end_date.year, end_date.month

# input the year and the month, get the last weekday of this month (not holiday)
# to be solved
# not consistent results, for the same month, there are different possible return
def last_weekday_per_month(year, month):

    # 节假日接口(工作日对应结果为 0, 休息日对应结果为 1, 节假日对应的结果为 2 )
    server_url = "http://www.easybots.cn/api/holiday.php?d="

    start_date = datetime.date(year, month, 1)
    _, days_in_months = calendar.monthrange(year, month)
    end_date = start_date + datetime.timedelta(days = days_in_months-1)
    # print("end day of this month ", end_date)
    d = str(end_date).replace("-", "")
    req = requests.get(server_url + d)
    # print(req.text)

    # 获取data值
    while int(json.loads(req.text)[d]) != 2 :

        end_date = end_date + datetime.timedelta(days= -1)
        # print("current check end date ", end_date)
        d = str(end_date).replace("-", "")
        req = requests.get(server_url + d)
        # print(req.text)

    return end_date

# 例如现在为 19年5月，求6月前，12，24。。。
def cal_n_month_before( year, month, n):
    year -= math.floor(n/12)
    month -= (n%12-1)
    if month <= 0:
        year = year-1
        month = 12+month
    if month == 13:
        month = 1
        year += 1
    return year, month

#year, month = 2019, 4
#n = 2
#print(cal_n_month_before(year, month, n))
#date = datetime.date(2012, 3, 30)
#print(get_next_month_end(date))
