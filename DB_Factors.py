
import psycopg2
from DB_Database import Database
from time_process import get_next_month_end
import pandas as pd
import numpy as np
import time
import os
import datetime
import datetime_process as dtp

class FactorDatabase(Database):
    def __init__(self, db_name, host_name, user_name, pwd, port, start_date='2005-01-01', end_date='2019-06-01'):

        Database.__init__(self,db_name,host_name,user_name,pwd,port)
        start = time.clock()
        self.connnect_db()
        self.start_date = start_date
        self.end_date = end_date
        self.start_date = start_date
        self.end_date = end_date
        self._fut_performance = None
        self._fut_alpha_df = None
        self._alpha_param_df =None
        self._op_code_df = None
        self._op_param_df = None
        self._pool_param_df =None
        # 加载performance_DF
        # self.performance_df = self.get_performance_df(table_name = 'fut_performance')

    def type_trans(self,name):
        if name.endswith('id'):
            return 'varchar'
        elif name.endswith('date'):
            return 'date'
        elif name == 'code' or name == 'instrument':
            return 'varchar'
        elif name.startswith('near') or name.startswith('main') or name == 'method' or name == 'freq_unit' or name.startswith('index'):
            return 'varchar'
        elif name == 'weighted' or name.startswith('is') or name.startswith('if'):
            return 'bool'
        elif name.startswith('fut'):
            return 'varchar'
        else:
            return 'double precision'


    def add_columns(self,table_name,columns_name,datatype):
        columns = self.get_columns_name(table_name)
        # print(columns)
        cursor = self.db.cursor()
        if columns_name in columns:
            print('Columns {} already existed.'.format(columns_name))
            return
        sql = "alter table {} add column {} {}" .format(
                table_name,columns_name,datatype
            )
        print(sql)
        try:
            cursor.execute(sql)
            self.db.commit()
        except:
            self.db.commit()
            print(
                "Failed to add columns to table {}".format(table_name)
            )
        cursor.close()

    def search_keys(self,table_name,key_name,key_values):
        cursor = self.db.cursor()
        sql = "SELECT * FROM {0} where ({1}) = {2}".format(
            table_name,key_name,tuple(key_values.values))
        try:
            print(sql)
            cursor.execute(sql)
            results = cursor.fetchall()
        except:
            raise ValueError(
                "Searching Error."
            )
        cursor.close()
        if not results:
            return 1
        else:
            return 0

    def insert_data(self,table_name,df,keys):
        print('inserting data...')
        def insert_sql(cursor,table_name,col_name,data):
            sql = "INSERT into {0}({1}) VALUES{2}".format(
                table_name,
                col_name,
                tuple(data.values)
            ).replace('nan','null')
            print(sql)
            try:
                cursor.execute(sql)
            except:
                print(
                    "Inserting Error."
                )
        cursor = self.db.cursor()
        col_name = self.list_rename(df.columns.values)
        if keys is not None:
            keys_name = self.list_rename(keys)
        for i in range(0, len(df.index)):
                data = df.iloc[i]
                key_flag = 0
                if keys is not None:
                    key_flag = key_flag | self.search_keys(table_name,keys_name,data[keys])
                    # print(key_flag,data[keys])
                    if key_flag:
                        insert_sql(cursor,table_name,col_name,data)
                        self.db.commit()
                    else:
                        continue
                else:
                    insert_sql(cursor,table_name,col_name,data)
                    self.db.commit()
        cursor.close()
        return

    def update_data(self,table_name,df,keys):
        print('updating data...')
        def update_sql(cursor,table_name,data,key_data,data_name,keys_name):
            sql = "UPDATE {0} set ({1}) = {2} where ({3}) = {4}".format(
                table_name,
                data_name,
                tuple(data.values),
                keys_name,
                tuple(key_data.values),
            ).replace('nan','null')
            print(sql)
            try:
                cursor.execute(sql)
            except:
                print(
                    "Inserting Error."
                )
        cursor = self.db.cursor()
        col_ = list(df.columns.values)
        for key in keys:
            col_.remove(key)
        data_name = self.list_rename(list(df.columns.values))
        col_name = self.list_rename(col_)
        keys_name = self.list_rename(keys)
        # print(col_name)
        # print(keys_name)
        for i in range(0, len(df.index)):
                data = df.iloc[i]
                update_sql(cursor,table_name,
                           data,data[keys],
                           data_name,keys_name)
                self.db.commit()
        cursor.close()

    def add_alpha(self,table_name,df, keys=['date','instrument'],if_duplicate_name = False):
        cursor = self.db.cursor()
        columns = self.get_columns_name(table_name)
        columns_new = df
        columns_2add = list( set(columns_new) - set(columns) )
        if (keys[0] not in columns_new) or (keys[1] not in columns_new):
            print("PRIMARY KEY is required.")
            return
        else:
            columns_diff = list(set(columns_new).intersection(set(columns)))
            for key in keys:
               columns_diff.remove(key)
            if if_duplicate_name:
               if columns_2add:
                   for col in columns_2add:
                       self.add_columns(table_name, col, self.type_trans(col))
               self.update_data(table_name,df,keys)
            else:
                if columns_diff:
                    print('columns:{} already existed.'.format(columns_diff))
                    return
                else:
                    for col in columns_new:
                        self.add_columns(table_name,col,self.type_trans(col))
                    self.update_data(table_name,df,keys)


    def create_table(self,table_name):
        cursor = self.db.cursor()
        sql = "CREATE TABLE {0}()".format(
            table_name)
        try:
            print(sql)
            cursor.execute(sql)
            # results = cursor.fetchall()
        except:
            raise ValueError(
                "Creating Table Error."
            )
        cursor.close()

    def add_data(self, table_name, df, if_new_col = True,keys= None):
        cursor = self.db.cursor()
        columns = self.get_columns_name(table_name)
        columns_new = df.columns
        columns_diff = list(set(columns_new).difference(set(columns)))
        if not columns_diff:
            self.insert_data(table_name,df,keys=keys)
        else:
            if if_new_col:
                for col in columns_diff:
                    self.add_columns(table_name,col,self.type_trans(col))
                self.insert_data(table_name,df,keys=keys)
            else:
                raise ValueError('Columns {} are not in table {}.'.format(columns_diff,table_name))

    def get_performance_df(self,table_name = 'fut_performance'):
        sql = "select * from {}" .format(
                table_name
            )
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            for col in ["date", "start_date", "end_date"]:
                if col not in df.keys():
                    continue
                df[col] = df[col].map(dtp.date2datetime)
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
        cursor.close()


    def get_alpha_df(self,alpha_id = None, table_name='fut_alpha', start_date=None, end_date=None,

                        if_includes_date = True, if_set_time_range = False):
        if (not if_includes_date) or not (if_set_time_range):
            sql = "select date, instrument,{} from {}" .format(
                alpha_id, table_name,
            )
        else:
            sql = "select date,instrument,{} from {} where " \
                  "date between '{}' and '{}' order by date".format(
                alpha_id, table_name,
                start_date, end_date
            )
        cursor = self.db.cursor()
        try:
            # print(sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            # columns = self.get_columns_name(table_name)
            columns = ["date", "instrument", alpha_id]
            df = pd.DataFrame(result, columns=columns)
            for col in ["date", "start_date", "end_date"]:
                if col not in df.keys():
                    continue
                df[col] = df[col].map(dtp.date2datetime)
            if if_includes_date:
                df = df.set_index(['date', "instrument"])
                df = df.sort_index(level='date')
            cursor.close()
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def get_op_code_df(self,date=None,instrument=None, table_name='fut_op_code',if_includes_date=True):
        if date is None and instrument is None:
            sql = "select * from {}".format(table_name)
        elif date is None:
            sql = "select * from {} where instrument = '{}'".format(table_name,instrument)
        elif instrument is None:
            sql = "select * from {} where date = '{}'".format(table_name,date)
        else:
            sql = "select * from {0} where " \
                  "date = '{1}' and instrument = '{2}' ".format(
                table_name,date, instrument)
        # print(sql)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            for col in ["date", "start_date", "end_date"]:
                if col not in df.keys():
                    continue
                df[col] = df[col].map(dtp.date2datetime)
            if if_includes_date:
                df = df.set_index(['date'])
                df = df.sort_index(level='date')
            cursor.close()
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def get_alpha_param_df(self,alpha_id,table_name = 'fut_alpha_param'):
        sql = "select * from {} where alpha_id = '{}' ".format(table_name,alpha_id)
        cursor = self.db.cursor()
        # print(sql)
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            cursor.close()
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def get_op_param_df(self,op_id,table_name = 'fut_op_param'):
        sql = "select * from {} where op_id = '{}' ".format(table_name,op_id)
        cursor = self.db.cursor()
        # print(sql)
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            cursor.close()
            # return df
            pool_param = self.get_pool_param_df(df['fut_pool_id'].values[0])
            # print(pool_param)
            # print(df)
            return pd.merge(df,pool_param,on=['fut_pool_id'])
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def get_pool_param_df (self,fut_pool_id,table_name = 'fut_pool_param'):
        sql = "select * from {} " \
              "where fut_pool_id = '{}' ".format(table_name,fut_pool_id)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = self.get_columns_name(table_name)
            df = pd.DataFrame(result, columns=columns)
            cursor.close()
            return df
        except:
            print(
                "Failed to get data from table {}".format(table_name)
            )
            cursor.close()

    def get_alpha_factor(self, alpha_id):
        return self.fut_alpha_df[[alpha_id]]

    def list_rename(self,list_name):
        return ",".join(list_name)

    def get_op_cond(self, op_id):
        # get condition
        op_condtion = self.op_param_df.loc[op_id]
        # print(op_condtion)
        return op_condtion

    # op dataframe
    # def get_op_df(self):
    #     return
    # @property
    # def fut_performance_df(self):
    #     if self._fut_performance is None:
    #         self._fut_performance = self.get_performance_df()
    #     return self._fut_performance
    # @property
    # def fut_alpha_df(self):
    #     if self._fut_alpha is None:
    #         self._fut_alpha = self.get_alpha_df()
    #     return self._fut_alpha

def func_test():
    print('testing...')


if __name__ == "__main__":
    db_name = 'ultralpha_db'
    host_name = '40.73.102.25'
    user = 'cyr'
    password = 'cyr'
    port = '5432'
    fut_name = 'al'
    start_date = '2010-04-01'
    end_date = '2014-04-01'

    fb = FactorDatabase(db_name = db_name,host_name = host_name, user_name = user, pwd = password,port = port )
    # print('fact0r')

    # fb.add_columns('fut_performance','alpha_id','varchar')
    # print(fb.get_pool_param_df(fut_pool_id='0'))
    # print(fb.get_op_param_df(op_id='op_11'))

    # fb.create_table('    df = pd.read_csv('..\\alpha_data\\op_param_df_new.csv')
    #     df.columns = [i.replace('-','_') for i in df.columns]
    #     fb.add_data('fut_op_param',df)')

    # df = pd.read_csv('..\\alpha_data\\op_code_df.csv')
    # df.columns = [i.replace('-','_') for i in df.columns]
    # fb.add_data('fut_op_code',df)
    #
    # df = pd.read_csv('..\\alpha_data\\op_param_df_new.csv')
    # df.columns = [i.replace('-','_') for i in df.columns]
    # fb.add_data('fut_op_param',df)
    #
    df = pd.read_csv('..\\alpha_data\\fut_pool_param.csv')
    df.columns = [i.replace('-','_') for i in df.columns]
    fb.add_data('fut_pool_param',df)

    # dp = fb.get_alpha_data('fut_alpha',if_set_time_range=False)
    # dp = fb.get_alpha_factor('mom_5')
    # print(dp)




