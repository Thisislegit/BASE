import sys, re
from utils.Connection import *
import gensim
import numpy as np
from torch import nn
# from TreeConvolution.util import prepare_trees
from TreeConvolution import tcnn
import torch
from utils.Connection import all_tables_rows_num
from torch.autograd import Variable
import datetime,time

# OPERATOR_TYPE = ["Seq Scan","Hash","Hash Join","Nested Loop","Index Scan",
#                  "Sort","Aggregate","Gather Merge","Limit","Bitmap Index Scan",
#                  "Bitmap Heap Scan","Unique","Gather","Subquery Scan","Append",
#                  "SetOp","Index Only Scan","Materialize","Merge Join","Group","CTE Scan",
#                  "Result","WindowAgg"]
OPERATOR_TYPE = ["Seq Scan","Hash","Hash Join","Nested Loop",
                 "Sort","Aggregate","Gather Merge","Limit",
                 "Unique","Gather","Append",
                 "SetOp","Materialize","Merge Join","Group",
                 "Result","WindowAgg"]

SCAN_TYPE = ["Seq Scan","Index Scan", "Bitmap Index Scan","Bitmap Heap Scan",
                 "Index Only Scan", "CTE Scan"]
DB_EMBEDDING = './new-wv-nopairs-10.bin'
JOIN_TYPE = ["Hash Join", "Merge Join", "Nested Loop"]
GOAL_FIELD = 'Total Cost'#Actual Total Time，Total Cost
ORIGIN_FIELD = 'Startup Cost'#Startup Cost,Actual Startup Time
W2VEC = './new-wv-nopairs-10-all.bin'
model = gensim.models.Word2Vec.load(W2VEC)
GAMMA = 0.9

# [[Filters],[Join Filter],[Cond], [Relation_alias],[selectivity], [w2c]
def traversal_plan(plan, parent_plan=None):

    query_encode = [[], [], [], {}, {}, [], []]

    if 'Filter' in plan.keys():
        filters = get_filter(plan['Filter'], plan['Relation Name'])
        all_select = {}
        for i in range(len(filters)):
            if '.' not in filters[i]:
                filters[i] = plan['Relation Name'] + '.' + filters[i]
        query_encode[0] = query_encode[0] + filters

        w2c = get_w2c(plan['Filter'])
        query_encode[5] = query_encode[5] + w2c

    if 'Join Filter' in plan.keys():
        filters = get_join_filter(plan['Join Filter'])
        query_encode[1] = query_encode[1] + filters

    if 'Index Cond' in plan.keys():
        # 父节点中有Relation Name? 因为有时候是recheck, 此时需要重复统计
        if not plan.get('Relation Name'):
            assert 'Bitmap Heap Scan' in parent_plan['Node Type'] \
            or 'BitmapAnd' in parent_plan['Node Type'] \
            or 'BitmapOr' in parent_plan['Node Type']
        else:
            conditions = get_join_index(plan['Index Cond'])
            if conditions:
                if plan.get('Alias'):
                    conditions[0] = plan['Alias'] + '.' + conditions[0]
                else:
                    conditions[0] = plan['Relation Name']+'.'+conditions[0]
                query_encode[2].append(conditions)
    elif 'Hash Cond' in plan.keys():
        conditions = get_join(plan['Hash Cond'])
        query_encode[2] += conditions
    elif 'Merge Cond' in plan.keys():
        conditions = get_join(plan['Merge Cond'])
        query_encode[2] += conditions
    elif 'Join Filter' in plan.keys():
        query_encode[2].append(plan['Join Filter'].strip('(').strip(')').split(' = '))

    elif 'Recheck Cond' in plan.keys():
        assert len(plan['Plans']) == 1
        if plan['Plans'][0]['Node Type'] in ['BitmapAnd', 'BitmapOr']:
            for next_plan in plan['Plans'][0]['Plans']:
                conditions = get_join_index(next_plan['Index Cond'])
                # 可能是join关系也可能是filter
                if conditions:
                    if plan.get('Alias'):
                        conditions[0] = plan['Alias'] + '.' + conditions[0]
                    else:
                        conditions[0] = plan['Relation Name'] + '.' + conditions[0]
                    query_encode[2].append(conditions)
                else:
                    filters = get_filter(next_plan['Index Cond'], plan['Relation Name'])
                    for i in range(len(filters)):
                        if '.' not in filters[i]:
                            filters[i] = plan['Relation Name'] + '.' + filters[i]
                    query_encode[0] = query_encode[0] + filters

        else:
            # 可能是join关系也可能是filter
            conditions = get_join_index(plan['Plans'][0]['Index Cond'])
            if conditions:
                if plan.get('Alias'):
                    conditions[0] = plan['Alias'] + '.' + conditions[0]
                else:
                    conditions[0] = plan['Relation Name'] + '.' + conditions[0]
                query_encode[2].append(conditions)
            else:
                filters = get_filter(plan['Plans'][0]['Index Cond'], plan['Relation Name'])
                for i in range(len(filters)):
                    if '.' not in filters[i]:
                        filters[i] = plan['Relation Name'] + '.' + filters[i]
                query_encode[0] = query_encode[0] + filters

    if 'Alias' in plan.keys():
        # print(plan['Alias'],':',plan['Relation Name'])
        if plan['Alias'] != plan['Relation Name']:
            query_encode[3][plan['Alias']] = plan['Relation Name']

    if plan['Node Type'] in SCAN_TYPE:
        if not plan.get('Relation Name'):
            assert 'Bitmap Heap Scan' in parent_plan['Node Type'] \
                   or 'BitmapAnd' in parent_plan['Node Type'] \
                   or 'BitmapOr' in parent_plan['Node Type']
        else:
            query_encode[4][plan['Relation Name']] = plan['Plan Rows'] / all_tables_rows_num[plan['Relation Name']]
        # except:
        #     print('wrong one:', plan)
        # query_encode[4][plan['Alias']] = plan['Plan Rows'] / all_tables_rows_num[plan['Relation Name']]
    # print(query_encode)
    if 'Plans' in plan.keys():
        for i in plan['Plans']:
            query_child_encode = traversal_plan(i, plan)
            query_encode[0] = query_encode[0] + query_child_encode[0]
            query_encode[1] = query_encode[1] + query_child_encode[1]
            query_encode[2] = merge_re(query_encode[2], query_child_encode[2])
            query_encode[3] = merge_dic(query_encode[3], query_child_encode[3])
            query_encode[4] = merge_dic(query_encode[4], query_child_encode[4])
            query_encode[5] = query_encode[5] + query_child_encode[5]
    return query_encode


def merge_re(list1, list2):
    list_re = []
    for i in list1:
        list_re.append(i)
    for i in list2:
        list_re.append(i)
    return list_re


def merge_dic(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def data_check(str):
    filter = str.replace('(', '')
    filter = filter.replace(')', '')
    return filter.strip()


def get_w2c(plan):
    pattern = re.compile(r'\(([^\)]*)\)')
    arr = pattern.findall(plan)
    filters = []
    for i in arr:
        # pattern_date = re.compile(r'(.*?) (>=|<=|=|<>|<|>) \'(.*?)\'::[date|timestamp|bpchar|numeric|text]')
        pattern_date = re.compile(r'(.*?) (>=|<=|=|<>|<|>) \'(.*?)\'::[date|timestamp|bpchar|numeric|text]')
        now = pattern_date.findall(i)
        if len(now) == 1:
            key = now[0][2].lower()
            key = key.lower()
            if model.wv.__contains__(key):
                filters.append([now[0], model.wv.vocab[key].count, model.wv.__getitem__(key) ])
    return filters


# function to get filter from attribute "Filter"
def get_filter(plan, table_name):
    filters = []
    #2:(p_type)::text ~~ '%NICKEL'::text)
    #9:
    #11,18:(sum((partsupp.ps_supplycost * (partsupp.ps_availqty)
    #19：(p_container = ANY ('{"SM CASE","SM BOX","SM PACK","SM PKG"}'::bpchar[]))
    #20：((ps_availqty)::numeric > (SubPlan 1))
    #21：
    # pattern = re.compile(r'\(([^\)]*)\)')
    # arr = pattern.findall(plan)

    arr = re.split(r' AND | OR ', plan)
    # if "AND" in plan or "OR" in plan:
    #     # arr = re.findall(r'\((?:\((?:\([^()]*\)|[^()]*)*\)|[^()])*\)', plan[1:-1])
    # elif
    #
    # else:
    #     arr = re.findall(r'\((?:\((?:\([^()]*\)|[^()]*)*\)|[^()])*\)', plan)
    # print(plan)
    # print('arr len', len(arr))
    for i in arr:
        if " ANY " in i:
            # print(plan)
            now_filter = data_check(re.split(r' = | > | < ', i)[0])
            if "::" in now_filter:
                now_filter = now_filter.split('::')[0]
            filters.append(now_filter)
            continue
            # todo:计算Any的选择率，以及查看any条件下或不会有and or
        elif '~~' in i:
            # todo: 对于类似 ～～ 等like的情况，没有提取选择率，暂时仅提取表名
            #  "((note)::text !~~ '%(Double feature with ''Nosferatu'' (1932)%'::text) AND ((note)::text ~~ '%(200%)%'::text)"
            now_filter = re.split(r' ~~ | !~~ ', i)[0]
            if "::" in now_filter:
                now_filter = now_filter.split('::')[0]
            now_filter = data_check(now_filter)
            if now_filter not in filters:
                filters.append(now_filter)
            continue
        elif ' IS ' in i:
            # TODO: 选择率？
            now_filter = re.split(r' IS ', i)[0]
            now_filter = data_check(now_filter)
            if now_filter not in filters:
                filters.append(now_filter)
            continue

        # pattern_date = re.compile(r'(.*?) (>=|<=|=|<>|<|>) \'(.*?)\'::[date|timestamp|bpchar|numeric|text]')
        pattern_date = re.compile(r'(.*?) (>=|<=|=|<>|<|>) \'(.*?)\'::[date|timestamp|bpchar|numeric|text]')

        now = pattern_date.findall(i)
        if len(now) > 0:
            now[0] = list(now[0])
            if '::' in now[0][0]:
                now[0][0] = now[0][0].split('::')[0]
            if now[0][1] == '=':
                get_equal_selectivity(now[0], 0, table_name)
            elif now[0][1] == '!=' or now[0][1] == '<>' :
                get_equal_selectivity(now[0], 1, table_name)
            else:
                get_scalar_selectivity(now[0], table_name)

            filters.append(data_check(now[0][0]))
        # 同表 列的比较
        pattern_same = re.compile(r'(.*?) (>=|<=|=|<>|<|>) ([A-Za-z_.]+$)')
        now = pattern_same.findall(i)
        if len(now) > 0:
            filters.append(data_check(now[0][0]))
            filters.append(data_check(now[0][2]))
        # selection
        pattern_num = re.compile(r'(.*?) (>=|<=|=|<>|<|>) ((-?\d+)(\.\d+)?|-?\d+$)')
        now = pattern_num.findall(i)
        if len(now) > 0:
            now[0] = list(now[0])
            if '::' in now[0][0]:
                now[0][0] = now[0][0].split('::')[0]
            if now[0][1] == '=':
                get_equal_selectivity(now[0], 0, table_name)
            elif now[0][1] == '!=' or now[0][1] == '<>' :
                get_equal_selectivity(now[0], 1, table_name)
            else:
                get_scalar_selectivity(now[0], table_name)
            filters.append(data_check(now[0][0]))
    return filters




# Changes: add table_name into query
def get_equal_selectivity(cond,state,table_name):
    cur = conn.conn.cursor()
    # cond = list(cond)
    # if '::' in cond[0]:
    #     cond[0] = cond[0].split('::')[0]
    cur.execute("select n_distinct,most_common_vals,most_common_freqs,histogram_bounds,null_frac from pg_stats where attname = '" + data_check(cond[0]) + "'"
                + "and tablename = '" + table_name + "';")
    query = cur.fetchone()
    if cond[2] == "NULL":
        val = float(query[4])
    else:
        if query[0] == -1:
            # 表示唯一列
            val = 1 / (all_tables_rows_num[table_name] * (1 - float(query[4])))
        else:
            # 原来的匹配方式不好，导致query1，query2长度不一致
            # query_1 = query[1].replace("{", "").replace("}", "").replace('"', "").replace(' ', "").split(",")
            pattern = re.compile(r'[\{,]\"([^\"]*?)\"(?=[\},])|[\{,](.*?)(?=[\},])')
            query_1 = [a if a else b for a, b in pattern.findall(query[1])]
            query_2 = list(query[2])
            goal = cond[2].replace(" ","")
            assert len(query_1) == len(query_2)
            if goal in query_1:
                val = query_2[query_1.index(goal)]
            else:
                # val = (1-sum(query_2))/float(query_1[0])
                # 加入字符串的计算
                # selectivity = (1 - sum(mvf) - null_value)/(num_distinct - num_mcv)
                if int(query[0]) > 0:
                    val = (1-sum(query_2) - float(query[4])) / (int(query[0])-len(query_1))
                else:
                    val = (1 - sum(query_2) - float(query[4])) / (round(abs(query[0]) * all_tables_rows_num[table_name]) - len(query_1))

    if state == 1:
        val = 1 - val

    return [cond[0],cond[1],val]

# Changes: add table_name into query
def get_scalar_selectivity(cond, table_name):
    if "::" in cond[0]:
        cond[0] = cond[0].split('::')[0]
        cond[0] = data_check(cond[0])
    if "::" in cond[2]:
        cond[0] = cond[0].split('::')[0]
    cur = conn.conn.cursor()
    cur.execute(
        "select n_distinct,most_common_vals,most_common_freqs,histogram_bounds,null_frac from pg_stats where attname = '" + data_check(
            cond[0]) + "'" + "and tablename = '" + table_name +"';")
    query = cur.fetchone()
    # 加上对MCV/MCF的计算
    # query_1 = query[1].replace("{", "").replace("}", "").replace('"', "").replace(' ', "").split(",")
    # query_2 = list(query[2])
    if query[1] != None:
        pattern = re.compile(r'[\{,]\"([^\"]*?)\"(?=[\},])|[\{,](.*?)(?=[\},])')
        query_1 = [a if a else b for a, b in pattern.findall(query[1])]
        query_2 = list(query[2])
        assert len(query_1) == len(query_2)
        val_MCF = 0
        if cond[1] != 'like':
            for i in range(len(query_1)):
                # check = eval(cond[2] + cond[1] + query_1[i])
                try:
                    check = eval(query_1[i] + cond[1] + cond[2])
                except:
                    check = False
                if check:
                    val_MCF += query_2[i]
        else:
            for i in range(len(query_1)):
                if re.findall(re.compile('^' + cond[2][:-1]), query_1[i]):
                    val_MCF += query_2[i]

    else:
        val_MCF = 1 / all_tables_rows_num[table_name]

    val = 0
    if query[3] is not None:
        # query_3 = query[3].replace("{", "").replace("}", "").split(",")
        pattern = re.compile(r'[\{,]\"([^\"]*?)\"(?=[\},])|[\{,](.*?)(?=[\},])')
        query_3 = [a if a else b for a, b in pattern.findall(query[3])]

        if isVaildDate(query_3[0]):
            now_time = cond[2].split(" ")[0]
            print(now_time)
            assert 0
            now_time = datetime.datetime.strptime(now_time, "%Y-%m-%d")
            for i in range(len(query_3)):
                query_3[i] = datetime.datetime.strptime(query_3[i], "%Y-%m-%d")
            val = 0
            for i in range(len(query_3)-1):
                if query_3[i] < now_time:
                    if query_3[i+1] < now_time:
                        val += 1/len(query_3)
                    else:
                        # print(now_time,query_3[i],query_3[i+1])
                        # print((now_time-query_3[i]),(query_3[i+1]-query_3[i]))
                        val += (now_time-query_3[i])/(query_3[i+1]-query_3[i])/len(query_3)
                        break
                else:
                    break
        else:
            # now_num = float(cond[2])
            if query[2] is None:
                rest_prob = 1 - float(query[4])
            else:
                rest_prob = 1 - sum(query_2) - float(query[4])
            try:
                now_num = float(cond[2])
                for i in range(len(query_3)):
                    query_3[i] = float(query_3[i])
                if now_num >= query_3[i-1]:
                    val = rest_prob
                else:
                    for i in range(len(query_3) - 1):
                        if query_3[i] < now_num:
                            if query_3[i + 1] < now_num:
                                val += rest_prob / (len(query_3) - 1)
                            else:
                                val += rest_prob * (now_num - query_3[i]) / (query_3[i + 1] - query_3[i]) / (len(query_3) - 1)
                                break
                        else:
                            break
                if cond[1] in [">", ">="]:
                    val = rest_prob - val
            except:
                now_num = str(cond[2])
                # for i in range(len(query_3)):
                #     query_3[i] = float(query_3[i])
                for i in range(len(query_3) - 1):
                    if query_3[i] < now_num:
                        if query_3[i + 1] < now_num:
                            val += rest_prob / len(query_3)
                        else:
                            val += rest_prob / len(query_3) / 2
                            break
                    else:
                        break
                if cond[1] in [">", ">="]:
                    val = rest_prob - val
        val += val_MCF
        return [cond[0], cond[1], val]


def isVaildDate(date):
    try:
        time.strptime(date, "%Y-%m-%d")
        return True
    except:
        return False


def get_like(sql):
    sql = sql.lower()
    pattern_num = re.compile(r" (\S*?) like '(.*?)'")
    all = pattern_num.findall(sql)
    all_like = []
    for now in all:
        key = now[1].replace("%", "")
        if model.wv.__contains__(key):
            if now[1][0] == "%":
                all_like.append([1, now[0], model.wv.vocab[key].count, model.wv.__getitem__(key)])
            else:
                all_like.append([0, now[0], model.wv.vocab[key].count, model.wv.__getitem__(key)])
        else:
            continue
    return all_like


def get_join_filter(plan):
    filters = []
    pattern = re.compile(r'\(([^\)]*)\)')
    arr = pattern.findall(plan)
    for i in arr:
        pattern_sub = re.compile(r'([A-Za-z_.]+) (>=|<=|=|<>|<|>) \(SubPlan')
        now = pattern_sub.findall(i)
        if len(now) > 0:
            filters.append(data_check(now[0][0]))
            continue

        pattern_date = re.compile(r'(.*?) (>=|<=|=|<>|<|>) \'(.*?)\'::[date|timestamp|bpchar|numeric|text]')
        now = pattern_date.findall(i)
        if len(now) > 0:
            filters.append(data_check(now[0][0]))
            continue

        pattern_num = re.compile(r'(.*?) (>=|<=|=|<>|<|>) ((-?\d+)(\.\d+)?|-?\d+$)')
        now = pattern_num.findall(i)
        if len(now) > 0:
            filters.append(data_check(now[0][0]))
            continue

        pattern_match = re.compile(r'([A-Za-z_.0-9]+) (>=|<=|=|<>|<|>) ([A-Za-z_.0-9]+)')
        now = pattern_match.findall(i)
        if len(now) > 0:
            filters.append(data_check(now[0][0]))
            if data_check(now[0][2]) != 'ANY':
                filters.append(data_check(now[0][2]))

    return filters


def get_join_index(plan):
    # to avoid：(keyword = 'character-name-in-title'::text)
    # 因为此时等号右边不是表名
    if '::' not in plan:
        pattern_match = re.compile(r'([A-Za-z_.0-9]+) (>=|<=|=|<>|<|>) ([A-Za-z_.0-9]+)')
        now = pattern_match.findall(plan)
        return [now[0][0],now[0][2]]
    return None


def get_join(plan):
    pattern_match = re.compile(r'([A-Za-z_.0-9]+) (>=|<=|=|<>|<|>) ([A-Za-z_.0-9]+)')
    now = pattern_match.findall(plan)
    all = []
    for i in now:
        all.append([i[0],i[2]])
    return all


def get_join_table(join):
    now_join = []
    for i in range(len(join)):
        now_join.append([join[i][0].split(".")[0], join[i][1].split(".")[0]])
        # now_join[i][0] = join[i][0].split(".")[0]
        # now_join[i][1] = join[i][1].split(".")[0]
    return now_join


# function to replace alias
def replace_alias(encode):
    for i in range(len(encode[0])):
        alias = encode[0][i].split(".")
        if alias[0] in encode[3].keys():
            encode[0][i] = encode[3][alias[0]]+'.'+alias[1]
    for i in range(len(encode[1])):
        alias = encode[1][i].split(".")
        if alias[0] in encode[3].keys():
            encode[1][i] = encode[3][alias[0]]+'.'+alias[1]
    for i in range(len(encode[2])):
        alias = encode[2][i][0].split(".")
        if alias[0] in encode[3].keys():
            encode[2][i][0] = encode[3][alias[0]]+'.'+alias[1]
        alias = encode[2][i][1].split(".")
        if alias[0] in encode[3].keys():
            encode[2][i][1] = encode[3][alias[0]] + '.' + alias[1]
    for i in range(len(encode[5])):
        alias = encode[5][i][0][0].split(".")
        if alias[0] in encode[3].keys():
            encode[5][i][0] = list(encode[5][i][0])
            encode[5][i][0][0] = encode[3][alias[0]]+'.'+alias[1]
            encode[5][i][0] = tuple(encode[5][i][0])
    return encode


def check_join_table(join_table, file_num):
    if file_num == 7 or file_num == 8:
        for goal in join_table:
            if goal[0] == 'nation' or goal[1] == 'nation':
                join_table.remove(goal)
        return join_table
    else:
        return join_table


def remove_same_table(join_table, f):
    if f in [7, 8, 18, 20, 21]:
        return [tuple(t) for t in join_table]
    for i in range(len(join_table)):
        join_table[i].sort()
    return list(set([tuple(t) for t in join_table]))


# function to use encode to get all mentioned tables
def get_tables(encode):
    tables = set()
    tables_join = []
    for i in range(len(encode[2])):
        tables.add(encode[2][i][0].split('.')[0])
        tables.add(encode[2][i][1].split('.')[0])
        tables_join.append([encode[2][i][0].split('.')[0], encode[2][i][1].split('.')[0]])
    return [tables_join, list(tables)]


def get_vector(encode):
    #join matrix
    re_matrix = [[0] * len(all_tables) for i in range(len(all_tables))]
    re_matrix_vector = []
    for i in range(len(encode[2])):
        table_left = encode[2][i][0].split('.')
        table_right = encode[2][i][1].split('.')
        m = all_tables[table_left[0]]
        n = all_tables[table_right[0]]
        if m < n:
            re_matrix[m][n] = 1
        else:
            re_matrix[n][m] = 1

    for i in range(0, len(all_tables)):
        re_matrix_vector = re_matrix_vector + re_matrix[i][i+1:]

    #predicate vector
    pre_vector = [0] * len(all_attribute_name[0])
    for k, v in encode[4].items():
        pre_vector[all_attribute_name[0][k]] += v
        # pre_vector[all_attribute_name[0][encode[0][i]]] += 1

    return re_matrix_vector+pre_vector


def get_vector_with_w2c(encode):
    # each 列值 [symobl(one-hot), 匹配数目encode[5][i][1],embedding(encode[5][i][2]),selectivity ]
    symbol = ['=', '<>', 0, 1, ">=", "<="]#like_2 %开头的
    # join matrix
    re_matrix = [[0] * len(all_tables) for i in range(len(all_tables))]
    re_matrix_vector = []
    for i in range(len(encode[2])):
        table_left = encode[2][i][0].split('.')
        table_right = encode[2][i][1].split('.')
        m = all_tables[table_left[0]]
        n = all_tables[table_right[0]]
        if m < n:
            re_matrix[m][n] = 1
        else:
            re_matrix[n][m] = 1

    for i in range(0, len(all_tables)):
        re_matrix_vector = re_matrix_vector + re_matrix[i][i + 1:]

    # predicate vector
    word_vector = [[0] * 18 for _ in range(len(all_attribute_name[0]))]
    if len(encode[5]) > 0:
        for i in encode[5]:
            now_col_vector = []
            if i[0][0].replace("(", "") in all_attribute_name[0]:
                word_vector_index = all_attribute_name[0][i[0][0].replace("(", "")]
            elif i[0][0].replace("(", "") in all_attribute_name[1]:
                word_vector_index = all_attribute_name[1][i[0][0].replace("(", "")][1]
            else:
                continue
            # word_vector_index = all_attribute_name[0][i[0][0].replace("(", "")]
            symbol_now_vector = [0] * len(symbol)
            symbol_now_vector[symbol.index(i[0][1])] = 1
            now_col_vector += symbol_now_vector
            now_col_vector.append(pow(i[1], 1/4))
            now_col_vector += list(i[2])
            now_col_vector.append(1)
            word_vector[word_vector_index] = now_col_vector

    if len(encode[6]) > 0:
        for i in encode[6]:
            now_col_vector = []
            word_vector_index = all_attribute_name[1][i[1]][1]
            symbol_now_vector = [0] * len(symbol)
            symbol_now_vector[symbol.index(i[0])] = 1
            now_col_vector += symbol_now_vector
            now_col_vector.append(pow(i[2], 1 / 4))
            now_col_vector += list(i[3])
            now_col_vector.append(1)
            word_vector[word_vector_index] = now_col_vector
    for i in range(len(encode[0])):
        word_vector[all_attribute_name[0][encode[0][i]]][-1] = 1
    word_vector = list(chain.from_iterable(word_vector))
    return re_matrix_vector + word_vector


# function to traversal plan tree
def traversal_plan_tree(plan):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        return (tuple(vector),)
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        vector_left = traversal_plan_tree(plan['Plans'][0])
        vector_right = traversal_plan_tree(plan['Plans'][1])
        vector = vector or list(vector_left[0])
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        return (tuple(vector) , vector_left , vector_right)
    else:
        child = traversal_plan_tree(plan['Plans'][0])
        return child


def traversal_plan_tree_execute(plan, query_vector, optimizer, model):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        loss = train_net(query_vector, (tuple(vector),), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        count = 1
        return [loss, count, (tuple(vector),)]
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        left = traversal_plan_tree_execute(plan['Plans'][0], query_vector, optimizer, model)
        right = traversal_plan_tree_execute(plan['Plans'][1], query_vector, optimizer, model)
        vector_left = left[2]
        vector_right = right[2]
        vector = vector or list(vector_left[0])
        # vector = vector or vector_right[0]
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        loss = train_net(query_vector, (tuple(vector), vector_left, vector_right), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        return [loss+left[0]+right[0], 1+left[1]+right[1], (tuple(vector), vector_left, vector_right)]
    else:
        node = traversal_plan_tree_execute(plan['Plans'][0], query_vector, optimizer, model)
        return node


def traversal_plan_tree_execute_test(plan, query_vector, optimizer, model):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        loss = test_net(query_vector, (tuple(vector),), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        return [loss[0], 1, (tuple(vector),), loss[1]]
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        left = traversal_plan_tree_execute_test(plan['Plans'][0], query_vector, optimizer, model)
        right = traversal_plan_tree_execute_test(plan['Plans'][1], query_vector, optimizer, model)
        vector_left = left[2]
        vector_right = right[2]
        vector = vector or list(vector_left[0])
        # vector = vector or vector_right[0]
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        loss = test_net(query_vector, (tuple(vector), vector_left, vector_right), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        return [loss[0]+left[0]+right[0], 1+left[1]+right[1], (tuple(vector), vector_left, vector_right), np.sum([loss[1], left[3], right[3]], axis=0)]
    else:
        node = traversal_plan_tree_execute_test(plan['Plans'][0], query_vector, optimizer, model)
        return node


def traversal_plan_tree_execute_reward(plan, query_vector, optimizer, model, reward):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        loss = 0
        count = 0
        return [loss, count, (tuple(vector),)]
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        left = traversal_plan_tree_execute_reward(plan['Plans'][0], query_vector, optimizer, model, reward)
        right = traversal_plan_tree_execute_reward(plan['Plans'][1], query_vector, optimizer, model, reward)
        vector_left = left[2]
        vector_right = right[2]
        vector = vector or list(vector_left[0])
        # vector = vector or vector_right[0]
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        # print(plan['Node Type'],reward)
        reward_now = 0 if len(reward) == 0 else reward.pop(0)
        loss = train_net(query_vector, (tuple(vector), vector_left, vector_right), plan[GOAL_FIELD]-plan[ORIGIN_FIELD] + GAMMA*reward_now, optimizer, model)
        return [loss+left[0]+right[0], 1+left[1]+right[1], (tuple(vector), vector_left, vector_right)]
    else:
        node = traversal_plan_tree_execute_reward(plan['Plans'][0], query_vector, optimizer, model, reward)
        return node


def traversal_plan_tree_execute_reward_test(plan, query_vector, optimizer, model, reward):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        return [0, 1, (tuple(vector),), [0]]
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        left = traversal_plan_tree_execute_reward_test(plan['Plans'][0], query_vector, optimizer, model, reward)
        right = traversal_plan_tree_execute_reward_test(plan['Plans'][1], query_vector, optimizer, model, reward)
        vector_left = left[2]
        vector_right = right[2]
        vector = vector or list(vector_left[0])
        # vector = vector or vector_right[0]
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        reward_now = 0 if len(reward) == 0 else reward.pop(0)
        loss = test_net(query_vector, (tuple(vector), vector_left, vector_right), plan[GOAL_FIELD]-plan[ORIGIN_FIELD] + GAMMA*reward_now, optimizer, model)
        return [loss[0]+left[0]+right[0], 1+left[1]+right[1], (tuple(vector), vector_left, vector_right), np.sum([loss[1], left[3], right[3]], axis=0)]
    else:
        node = traversal_plan_tree_execute_reward_test(plan['Plans'][0], query_vector, optimizer, model, reward)
        return node


def traversal_plan_tree_execute_each_test(plan, query_vector, optimizer, model):
    vector = [0] * (len(all_tables)*2+3)
    if plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Index Scan':
            vector[2 + all_tables[plan['Relation Name']] * 2] = plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        else:
            vector[2 + all_tables[plan['Relation Name']] * 2] = 1
        loss = test_net(query_vector, (tuple(vector),), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        return [loss[0], 1, (tuple(vector),), loss[1]]
    elif plan['Node Type'] in JOIN_TYPE:
        vector[JOIN_TYPE.index(plan['Node Type'])] = 1
        left = traversal_plan_tree_execute_test(plan['Plans'][0], query_vector, optimizer, model)
        right = traversal_plan_tree_execute_test(plan['Plans'][1], query_vector, optimizer, model)
        vector_left = left[2]
        vector_right = right[2]
        vector = vector or list(vector_left[0])
        # vector = vector or vector_right[0]
        for i in range(len(JOIN_TYPE), len(vector)):
            if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                vector[i] = 1
        loss = test_net(query_vector, (tuple(vector), vector_left, vector_right), plan[GOAL_FIELD]-plan[ORIGIN_FIELD], optimizer, model)
        return [loss[0], 1, (tuple(vector), vector_left, vector_right), np.sum([loss[1], left[3], right[3]], axis=0)]
    else:
        node = traversal_plan_tree_execute_test(plan['Plans'][0], query_vector, optimizer, model)
        return node

'''
def traversal_plan_tree_cost(plan, f, query_vector):
    vector = [0] * (len(all_tables)*2+3)
    if 'Plans' in plan.keys() and plan['Node Type'] not in SCAN_TYPE:
        if len(plan['Plans']) > 1:
            if plan['Node Type'] == 'Sort' or plan['Node Type'] == 'Aggregate':
                if f == 11 or f == 17:
                    node = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
                    return node
                elif f == 22:
                    node = traversal_plan_tree_cost(plan['Plans'][1], f, query_vector)
                    return node
            else:
                if plan['Node Type'] in JOIN_TYPE:
                    vector[JOIN_TYPE.index(plan['Node Type'])] = 1
                else:
                    vector[JOIN_TYPE.index('Nested Loop')] = 1
                left = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
                right = traversal_plan_tree_cost(plan['Plans'][1], f, query_vector)
                assert len(left[0]) == 0 or len(right[0]) == 0
                vector_left = left[1]
                vector_right = right[1]
                vector = vector or list(vector_left[0])
                # vector = vector or vector_right[0]
                for i in range(len(JOIN_TYPE), len(vector)):
                    if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                        vector[i] = 1
                cost = []
                cost = cost + left[0]
                cost = cost + right[0]
                # todo: 更改为Total cost, 后续计算差值
                # cost.append(round(pow(plan[GOAL_FIELD] - plan[ORIGIN_FIELD], 1/4), 6))
                cost.append(plan[GOAL_FIELD])
                state = []
                state = state+left[2]
                # assert len(right[2]) == 0
                state = state+right[2]
                state.append([query_vector, (tuple(vector), vector_left, vector_right)])
                return [cost, (tuple(vector), vector_left, vector_right), state]
        else:
            node = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
            return node
    elif plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Heap Scan': #  ? index change to heap
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1 # plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[4 + all_tables[plan['Relation Name']] * 2] = 1
        elif plan['Node Type'] == 'Bitmap Index Scan':
            pass
        else:
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        return [[], (tuple(vector),), []]
    elif 'Plan' in plan.keys():
        node = traversal_plan_tree_cost(plan['Plan'][0], f, query_vector)
        return node
    else:
        return []
'''

def traversal_plan_tree_cost(plan, f, query_vector):
    vector = [0] * (len(all_tables)*2+3)
    if 'Plans' in plan.keys() and plan['Node Type'] not in SCAN_TYPE:
        if len(plan['Plans']) > 1:
            if plan['Node Type'] == 'Sort' or plan['Node Type'] == 'Aggregate':
                if f == 11 or f == 17:
                    node = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
                    return node
                elif f == 22:
                    node = traversal_plan_tree_cost(plan['Plans'][1], f, query_vector)
                    return node
            else:
                if plan['Node Type'] in JOIN_TYPE:
                    vector[JOIN_TYPE.index(plan['Node Type'])] = 1
                else:
                    vector[JOIN_TYPE.index('Nested Loop')] = 1
                left = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
                right = traversal_plan_tree_cost(plan['Plans'][1], f, query_vector)
                assert len(left[0]) == 0 or len(right[0]) == 0
                vector_left = left[1]
                vector_right = right[1]
                vector = vector or list(vector_left[0])
                # vector = vector or vector_right[0]
                for i in range(len(JOIN_TYPE), len(vector)):
                    if vector_left[0][i] == 1 or vector[i] == 1 or vector_right[0][i] == 1:
                        vector[i] = 1
                cost = []
                cost = cost + left[0]
                cost = cost + right[0]
                # todo: 更改为Total cost, 后续计算差值
                # cost.append(round(pow(plan[GOAL_FIELD] - plan[ORIGIN_FIELD], 1/4), 6))

                if vector[:3] == [0, 1, 0] and not(plan['Total Cost'] > plan['Plans'][0]['Total Cost'] and plan['Total Cost'] > plan['Plans'][1]['Total Cost']):
                    # for merge join
                    cost.append((plan[GOAL_FIELD], calcualte_merge_join_cost(plan)))
                else:
                    cost.append(plan[GOAL_FIELD])
                state = []
                state = state+left[2]
                # assert len(right[2]) == 0
                state = state+right[2]
                state.append([query_vector, (tuple(vector), vector_left, vector_right)])
                return [cost, (tuple(vector), vector_left, vector_right), state]
        else:
            node = traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
            return node
    elif plan['Node Type'] in SCAN_TYPE:
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan['Node Type'] == 'Bitmap Heap Scan': #  ? index change to heap
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1 # plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[4 + all_tables[plan['Relation Name']] * 2] = 1
        elif plan['Node Type'] == 'Bitmap Index Scan':
            pass
        else:
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        return [[], (tuple(vector),), []]
    elif 'Plan' in plan.keys():
        node = traversal_plan_tree_cost(plan['Plan'][0], f, query_vector)
        return node
    else:
        return []


def calcualte_merge_join_cost(plan):
    def calculate_sort_cost(plan):
        assert plan['Node Type'] == 'Sort'
        assert len(plan['Plans']) == 1
        if plan['Plans'][0]['Node Type'] in JOIN_TYPE:
            return plan['Total Cost'] - plan['Plans'][0]['Total Cost']
        elif plan['Plans'][0]['Node Type'] in SCAN_TYPE:
            return plan['Total Cost']
        else:
            print(plan)
            assert 0

    assert plan['Node Type'] == 'Merge Join'
    # all cost = current_node + child_1 + child_2
    assert  len(plan['Plans']) == 2
    child_plan_1 = plan['Plans'][0]
    child_plan_2 = plan['Plans'][1]
    if plan['Total Cost'] > child_plan_1['Total Cost'] and plan['Total Cost'] > child_plan_2['Total Cost']:
        print(child_plan_1['Node Type'] , child_plan_2['Node Type'])
        assert child_plan_1.get('Plans') or child_plan_2.get('Plans')
        if child_plan_1.get('Plans') and child_plan_1['Plans'][0]['Node Type'] in JOIN_TYPE:
            return plan['Total Cost'] - child_plan_1['Plans'][0]['Total Cost']
        elif child_plan_2.get('Plans') and child_plan_2['Plans'][0]['Node Type'] in JOIN_TYPE:
            return plan['Total Cost'] - child_plan_2['Plans'][0]['Total Cost']
        else:
            print('Not exist!')
            assert 0
    else:
        # assert child_plan_2['Node Type'] in SCAN_TYPE or child_plan_1['Node Type'] in SCAN_TYPE
        child_cost = 0
        current_cost = plan['Total Cost'] - plan['Startup Cost']
        for child_plan in plan['Plans']:
            if child_plan['Node Type'] == 'Sort':
                child_cost += calculate_sort_cost(child_plan)
            elif child_plan['Node Type'] in SCAN_TYPE:
                child_cost += child_plan['Total Cost']
            elif child_plan['Node Type'] == 'Materialize':
                assert len(child_plan['Plans']) == 1
                if child_plan['Plans'][0]['Node Type'] in SCAN_TYPE:
                    child_cost += child_plan['Total Cost']
                else:
                    child_cost += child_plan['Total Cost'] - child_plan['Plans'][0]['Total Cost']
                    child_cost += child_plan['Plans'][0]['Total Cost'] - child_plan['Plans'][0]['Plans'][0]['Total Cost']

            elif child_plan['Node Type'] in JOIN_TYPE:
                continue
            else:
                print(child_plan['Node Type'])
                assert 0
        return current_cost + child_cost


# function to train value network
def train_net(query_encode, plan_encode, cost, optimizer, net):
    net.train()
    output = net([query_encode, plan_encode])
    optimizer.zero_grad()
    compute_loss = nn.L1Loss()
    loss = compute_loss(output, torch.tensor([round(pow(cost, 1/4), 6)]))
    loss.backward()
    # print(output, [round(pow(cost, 1/4), 6)], cost ,loss)
    # for name, parms in net.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
    #           ' -->grad_value:', parms.grad)
    optimizer.step()
    return loss


# function to test network
def test_net(query_encode, plan_encode, cost, optimizer, net):
    net.eval()
    output = net([query_encode, plan_encode])
    optimizer.zero_grad()
    compute_loss = nn.L1Loss()
    loss = compute_loss(output, torch.tensor([pow(cost, 1/4)]))
    goal = max(output/(pow(cost, 1/4)), pow(cost, 1/4)/output)
    range_1, range_2, range_3 = 0, 0, 0
    if goal < 1.5:
        range_1 += 1
    elif goal > 3:
        range_3 += 1
    else:
        goal += 1
    return [loss, [range_1, range_2, range_3]]



