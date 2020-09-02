import numpy as np
import time
import pandas as pd


class CachePool():

    def __init__(self):
        # self.keys = []
        self.save_pool = {}

    def save_key_values(self, stras_df):
        for key in stras_df.columns:
            self.save_pool[key] = stras_df[key]
        return 0

    def load_key_values(self, key_df):
        keys = self.df2keys(key_df)
        stras_df_ = pd.DataFrame()
        for key in keys:
            stras_df_.insert(0, key, self.save_pool[key])
        return stras_df_

    def find_filter_key(self, key_df):
        keyset = self.df2keys(key_df)
        already_saved_key = []
        for key in keyset:
            if key in self.save_pool.keys():
                already_saved_key.append(key)
            else:
                self.save_pool[key] = None
        if already_saved_key is not None:
            # print('ASK :',already_saved_key)
            for ki in already_saved_key:
                alpha_id, op_id = ki.split('$')
                index_ = key_df.loc[key_df['alpha_id'] == alpha_id].index
                index_ = key_df.loc[index_].loc[key_df['op_id'] == op_id].index
                print(index_)
                key_df.drop(index=index_, axis=0, inplace=True)
        # print(key_df)
        return key_df

    def save_filter_key(self, key_df):
        keyset = self.df2keys(key_df)
        already_saved_key = []
        for key in keyset:
            if key in self.save_pool.keys():
                already_saved_key.append(key)
            else:
                self.save_pool[key] = None
            # self.save_pool[key] = stras_df[key]
        return already_saved_key

    @staticmethod
    def keys2df(keys):
        df = pd.DataFrame(columns=['alpha_id', 'op_id'])
        for key in keys:
            alpha_id, op_id = key.split('$')
            df.insert(value=[alpha_id, op_id])
        print(df)
        return df

    @staticmethod
    def df2keys(df):
        keyset = []
        # print(df)
        for i in df.index:
            # print(i)
            keyset.append('{0}${1}'.format(df.loc[i]['alpha_id'], df.loc[i]['op_id']))
        return keyset


cachePoolDefault = CachePool()
