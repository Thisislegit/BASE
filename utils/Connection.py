import psycopg2
import configparser
import linecache
from itertools import chain
import threading
import json
import os


class Conn:
    def __init__(self):
        conf = configparser.ConfigParser()
        conf.read("./postgresql.conf")
        username = conf.get("postgre", "username")
        password = conf.get("postgre", "password")
        dbname = conf.get("postgre", "dbname")
        self.conn = psycopg2.connect(database=dbname, user=username, password=password, host="localhost", port="5432")
        cursor = self.conn.cursor()
        cursor.execute("select * from pg_stats where schemaname = 'public';")
        self.stats = cursor.fetchall()

    def reconnect(self):
        self.conn.close()
        conf = configparser.ConfigParser()
        conf.read("./postgresql.conf")
        username = conf.get("postgre", "username")
        password = conf.get("postgre", "password")
        dbname = conf.get("postgre", "dbname")
        self.conn = psycopg2.connect(database=dbname, user=username, password=password, host="localhost", port="5432")


class Connection:
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(Connection, "_instance"):
            with Connection._instance_lock:
                if not hasattr(Connection, "_instance"):
                    Connection._instance = object.__new__(cls)
        return Connection._instance

    @staticmethod
    def get_connection():
        conf = configparser.ConfigParser()
        conf.read("./postgresql.conf")
        username = conf.get("postgre", "username")
        password = conf.get("postgre", "password")
        dbname = conf.get("postgre", "dbname")
        conn = psycopg2.connect(database=dbname, user=username, password=password, host="localhost", port="5432")
        return conn

    @staticmethod
    def get_all_attribute_name():
        is_exists = os.path.exists("./all_attribute.json")
        if is_exists:
            return json.load(open('./all_attribute.json', encoding='utf-8'))
        else:
            cur = conn.cursor()
            # sql_command = "select table_name,column_name from information_schema.columns where table_schema='public' and table_name<>'pg_stat_statements';"
            sql_command = "select table_name,column_name from information_schema.columns where table_schema='public';"
            cur.execute(sql_command)
            all_tables = cur.fetchall()
            total_attributes = {}
            simple_attributes = {}
            count = 0
            for i in all_tables:
                total_attributes[i[0] + '.' + i[1]] = count
                # revenue.l_suppkey lineitem.l_suppkey
                simple_attributes[i[1]] = [i[0], count]
                count += 1
            with open('./all_attribute.json', 'w') as file_obj:
                json.dump([total_attributes, simple_attributes], file_obj)
            return [total_attributes, simple_attributes]

    @staticmethod
    def get_all_tables():
        is_exists = os.path.exists('./all_tables.json')
        if is_exists:
            return json.load(open('./all_tables.json', encoding='utf-8'))
        else:
            cur = conn.cursor()
            sql_command = "SELECT tablename FROM pg_tables WHERE tablename NOT LIKE 'pg%' ORDER BY tablename;"
            cur.execute(sql_command)
            all_tables = list(chain.from_iterable(cur.fetchall()))
            all_tables_value = [element for element in range(0, len(all_tables))]
            dict_tables = dict(zip(all_tables, all_tables_value))
            with open('./all_tables.json', 'w') as file_obj:
                json.dump(dict_tables, file_obj)
            return dict_tables

    @staticmethod
    def get_all_tables_rows_num():
        is_exists = os.path.exists('./all_tables_rows_num.json')
        if is_exists:
            return json.load(open('./all_tables_rows_num.json', encoding='utf-8'))
        else:
            cur = conn.cursor()
            sql_command = """
                SELECT relname, reltuples 
                FROM pg_class r JOIN pg_namespace n 
                ON (relnamespace = n.oid) 
                WHERE relkind = 'r' AND n.nspname = 'public';
            """
            cur.execute(sql_command)
            all_tables_rows_num = cur.fetchall()
            all_tables_rows_num_dic = {}
            for i in all_tables_rows_num:
                all_tables_rows_num_dic[i[0]] = i[1]
            with open('./all_tables_rows_num.json', 'w') as file_obj:
                json.dump(all_tables_rows_num_dic, file_obj)
            return all_tables_rows_num_dic


conn = Conn()
# conn = conn.conn
all_attribute_name = Connection.get_all_attribute_name()
# todo: get_all_table中表取多了
all_tables = Connection.get_all_tables()
all_tables_rows_num = Connection.get_all_tables_rows_num()

