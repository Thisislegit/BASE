import sys
import os
import sqlparse
import sql_metadata
import json
import time
from utils.Connection import conn
from itertools import chain
from string import Template
EXPLAIN_FOLDER = './ExplainFiles/'
EXECUTED_EXPLAIN_FOLDER = './test_sql/executed_sql_plan/test_3/'
PREFIX_HINT_EXPLAIN = './hint_explain/'
PREFIX_HINT_ACTION = './hint_action/'
PREFIX_HINT_VECTOR = './hint_vector/'
EXECUTION_MAX_TIME = 90000
SLIDE_NUM = 1000000

def reverse_kv(now, join_table):
    new_one = dict()
    for key, value in now.items():
        new_one[value] = key
    for i in range(len(join_table)):
        if new_one.get(join_table[i][0]):
            join_table[i][0] = new_one.get(join_table[i][0])
        if new_one.get(join_table[i][1]):
            join_table[i][1] = new_one.get(join_table[i][1])
    return join_table


# function to execute SQL and store explain
def loadSQLFolder(conn, path):
    dir_list = os.listdir(path)

    for d in dir_list:
        for f in range(1, 23):
            if d == '.DS_Store':
                continue
            elif f == '.DS_Store':
                continue
            else:
                sql = loadSQL(path+d+'/'+str(f)+'.sql')
                storeSQLExplain(conn, sql, EXPLAIN_FOLDER+d+'/', str(f), 'EXPLAIN (format json) ')


# function to store SQL explain
def storeSQLExplain(conn, sql, foldername, filename, action):
    isExists = os.path.exists(foldername)
    if not isExists:
        os.makedirs(foldername)
    localtime = time.asctime(time.localtime(time.time()))
    print(foldername, filename, localtime)
    cur = conn.conn.cursor()
    with open(foldername+filename + '.json', 'w') as file_obj:
        ## 15需要新建view，单独处理
        # if filename == '15':
        #     sql_list = sql.split(";")
        #     cur.execute(sql_list[0]+";")
        #     sql = sql_list[1]+";"

        sql_command = action + sql
        cur.execute(sql_command)
        query_plan = cur.fetchone()[0][0]
        # if filename == '15':
        #     cur.execute(sql_list[2]+";")
        json.dump(query_plan, file_obj)


# function to load pg_hint_plan
def load_extension():
    cur = conn.conn.cursor()
    sql_command = "load 'pg_hint_plan';"
    cur.execute(sql_command)
    sql_command = "set from_collapse_limit = 10000;"
    cur.execute(sql_command)
    # sql_command = "ALTER SYSTEM SET pg_hint_plan.debug_print = on"
    # cur.execute(sql_command)
    # print(cur.fetchone[0])


# function to set statement timeout
def set_statement_timeout(time):
    cur = conn.conn.cursor()
    sql_command = "load 'pg_hint_plan';"
    cur.execute(sql_command)
    sql_command = "set statement_timeout to " + str(time)+";"
    cur.execute(sql_command)
    sql_command = " set max_parallel_workers_per_gather = 0; "
    cur.execute(sql_command)
    sql = "select * from pg_settings where name like '%statement_timeout%';"
    cur.execute(sql)
    print(cur.fetchone())


# function to get SQL explain from local
def get_SQL_explain(sql, action, filename, conn):
    cur = conn.cursor()
    if filename == '15':
        sql_list = sql.split(";")
        cur.execute(sql_list[0] + ";")
        sql = sql_list[1] + ";"

    sql_command = action + sql

    cur.execute(sql_command)
    query_plan = cur.fetchone()[0][0]
        # store_hint_SQL_explain(action, filename, query_plan, foldername, vector)

    if filename == '15':
        cur.execute(sql_list[2] + ";")
    return query_plan


# function to store vector,hint and explain
def store_hint_SQL_explain(action, filename, explain, foldername, vector):
    is_explain_exists = os.path.exists(PREFIX_HINT_EXPLAIN+foldername+'/'+filename)
    if not is_explain_exists:
        os.makedirs(PREFIX_HINT_EXPLAIN+foldername+'/'+filename)

    is_action_exists = os.path.exists(PREFIX_HINT_ACTION + foldername + '/' + filename)
    if not is_action_exists:
        os.makedirs(PREFIX_HINT_ACTION + foldername + '/' + filename)

    is_vector_exists = os.path.exists(PREFIX_HINT_VECTOR + foldername + '/' + filename)
    if not is_vector_exists:
        os.makedirs(PREFIX_HINT_VECTOR + foldername + '/' + filename)
    now = int(time.time())
    with open(PREFIX_HINT_EXPLAIN+foldername + '/' + filename + '/' + str(now) + '.json', 'w') as file_obj:
        json.dump(explain, file_obj)

    with open(PREFIX_HINT_ACTION+foldername + '/' + filename + '/' + str(now) + '.json', 'w') as file_obj:
        json.dump(action, file_obj)

    with open(PREFIX_HINT_VECTOR+foldername + '/' + filename + '/' + str(now) + '.json', 'w') as file_obj:
        json.dump(vector, file_obj)


# function to add hint and get explain
def get_hint_SQL_explain(sql, action, filename, conn):
    cur = conn.cursor()
    if filename == '15':
        sql_list = sql.split(";")
        cur.execute(sql_list[0] + ";")
        sql = sql_list[1] + ";"
    sql_command = action + sql
    # print(sql_command)
    # exit()
    try:
        cur.execute(sql_command)
        query_plan = cur.fetchone()[0][0]
        # store_hint_SQL_explain(action, filename, query_plan, foldername, vector)
    except Exception:
        conn.rollback()
        set_statement_timeout(EXECUTION_MAX_TIME)
        return -1
    if filename == '15':
        cur.execute(sql_list[2] + ";")
    return query_plan


# function to load explain from local
def loadSQLExplain(path):
    f = open(path, encoding='utf-8')
    content = json.load(f)
    return content


# function to load SQL from local
def loadSQL(path):
    fd = open(path, 'r', encoding='utf-8')
    sql = ''

    for line in fd:
        if line[0] == '-':
            continue
        else:
            sql = sql + line.replace('\n', ' ')

    sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
    return sql

# function to load training plan from tree-LSTM from local
def loadPlan(path='./train_plan/job_train_plan_big_shuf.json'):
    file = []
    count = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            file.append(json.loads(line))
            # count +=1
            # if count > 10000:
            #     break
    return file


# function to execute explain analyze sql
def executeSQL(conn, path):
    dir_list = os.listdir(path)
    number = "0123456789"
    dir_list = [i for i in dir_list if i[0] in number]
    for d in dir_list:
        sql = loadSQL(path + d)
        storeSQLExplain(conn, sql, EXECUTED_EXPLAIN_FOLDER, d.split('.')[0], 'EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json) ')


def create_table_sql():
    cur = conn.cursor()
    table_command = "SELECT tablename FROM pg_tables WHERE   schemaname = 'public' ORDER BY tablename;"
    cur.execute(table_command)
    all_tables = list(chain.from_iterable(cur.fetchall()))
    for i in all_tables:
        table_count_command = "select count(*) from " + i + ";"
        cur.execute(table_count_command)
        now_count = cur.fetchone()[0]
        attribute_command = "select column_name from information_schema.columns where data_type in ('character varying','character') and table_name='"+ i +"';"
        cur.execute(attribute_command)
        all_attributes = list(chain.from_iterable(cur.fetchall()))

        if now_count % SLIDE_NUM == 0:
            slide = now_count//SLIDE_NUM
        else:
            slide = now_count//SLIDE_NUM + 1
        for j in range(slide):
            sql_command = "select " + ','.join(all_attributes) + " from " + i + " offset " + str(j*SLIDE_NUM) + " limit "\
                          + str(SLIDE_NUM) + ";"
            with open('./EachTableSQL/' + i + '-'+ str(j) + '.sql', 'w') as file_obj:
                file_obj.write(sql_command)


def create_join_sql():
    cur = conn.cursor()

    table_count = """
        select relname,reltuples from pg_class r JOIN pg_namespace n 
                ON (relnamespace = n.oid) 
                WHERE relkind = 'r' AND n.nspname = 'public';
    """
    cur.execute(table_count)
    all_tables_count = dict(cur.fetchall())

    table_primary = """
            select col.table_name,col.column_name from
            information_schema.table_constraints tab,
            information_schema.constraint_column_usage col
            where col.constraint_name = tab.constraint_name
            and col.table_name = tab.table_name;
        """
    cur.execute(table_primary)
    all_table_primary = dict(cur.fetchall())

    table_command = """
        SELECT
             tc.constraint_name, tc.table_name, kcu.column_name, 
             ccu.table_name AS foreign_table_name,
             ccu.column_name AS foreign_column_name,
             tc.is_deferrable,tc.initially_deferred
        FROM 
             information_schema.table_constraints AS tc 
             JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
             JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
        where tc.table_name != ccu.table_name;
    """
    cur.execute(table_command)
    all_tables = cur.fetchall()
    name = 0
    for i in all_tables:

        if max(all_tables_count[i[1]]/all_tables_count[i[3]], all_tables_count[i[3]]/all_tables_count[i[1]]) < 100:
            continue
        if all_tables_count[i[1]] > all_tables_count[i[3]]:
            now_count = all_tables_count[i[1]]
            sql_command_template = Template("select * from " + i[3] + " join " + "（ select * from " + i[1] + "where " +
                                            all_table_primary[i[1]] + " offset $off limit" + str(SLIDE_NUM) + " ) mid on" +
                                            i[3] + "." + i[4] + " = mid." + i[2])
        else:
            now_count = all_tables_count[i[3]]
            sql_command_template = Template("select * from " + i[1] + " join " + "（ select * from " + i[3] + "where " +
                                   all_table_primary[i[3]] + " offset $off limit" + str(SLIDE_NUM) + " ) mid on" + i[1] +
                                            "." + i[2] + " = mid." + i[4])

        if now_count % SLIDE_NUM == 0:
            slide = now_count//SLIDE_NUM
        else:
            slide = now_count//SLIDE_NUM + 1

        for j in range(int(slide)):
            sql_command = sql_command_template.substitute(off=str(j*SLIDE_NUM))
                # "select * from " + i[1] + " join " + i[3] + ' on ' + i[1] + "." + i[2] + ' = ' + i[3] \
                #           + "." + i[4] + " offset " + str(j*SLIDE_NUM)+" limit " + str(SLIDE_NUM) + ";"
            with open('./EachJoinTableSQL/' + str(name) + '-' + str(j) + '.sql', 'w') as file_obj:
                file_obj.write(sql_command)
        name += 1


#state:1 test 0 train
def store_log(epoch, state, sql_template, sql_num, reward):
    root_path = './Output_log/'+str(epoch)+'/'+str(sql_template)+'/'

    if not os.path.exists(root_path):
        os.makedirs(root_path)
    # now = int(time.time())
    with open(root_path + str(sql_num) + '.json', 'w') as file_obj:
        json.dump(reward, file_obj)