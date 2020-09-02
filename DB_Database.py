import psycopg2
import pandas as pd

class Database():

    def __init__(self,db_name,host_name,user_name,pwd,port):
        self.db_name = db_name
        self.host_name = host_name
        self.user_name = user_name
        self.pwd = pwd
        self.port = port
        self.db = None

        # db_name = 'ultralpha_db'
        # host_name = '192.168.0.116'
        # user = 'cyr'
        # password = 'cyr'

    def connnect_db(self):
        if self.db is None:
            self.db = psycopg2.connect(database=self.db_name, user=self.user_name, password=self.pwd,
                              host=self.host_name, port=self.port)

    def close_db(self):
        self.db.close()

    def get_columns_name(self,table_name):
        sql = "SELECT COLUMN_NAME FROM information_schema.COLUMNS " \
              "where TABLE_NAME = '{}'".format(table_name)
        cursor = self.db.cursor()
        try:
            cursor.execute(sql)
            columns = cursor.fetchall()
            columns = [x[0] for x in columns]
            return columns
        except:
            print(
                "Failed to get tableColumn."
            )
        cursor.close()

