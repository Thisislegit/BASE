import itertools
import sqlparse

from sqlparse.sql import IdentifierList, Identifier, Where, Comparison, Parenthesis
from sqlparse.tokens import Keyword, DML, Whitespace, Newline

from utils.PlanTree import *


'''
/*
00024  * Note: the default selectivity estimates are not chosen entirely at random.
00025  * We want them to be small enough to ensure that indexscans will be used if
00026  * available, for typical table densities of ~100 tuples/page.  Thus, for
00027  * example, 0.01 is not quite small enough, since that makes it appear that
00028  * nearly all pages will be hit anyway.  Also, since we sometimes estimate
00029  * eqsel as 1/num_distinct, we probably want DEFAULT_NUM_DISTINCT to equal
00030  * 1/DEFAULT_EQ_SEL.
00031  */
00032 
00033 /* default selectivity estimate for equalities such as "A = b" */
00034 #define DEFAULT_EQ_SEL  0.005
00035 
00036 /* default selectivity estimate for inequalities such as "A < b" */
00037 #define DEFAULT_INEQ_SEL  0.3333333333333333
00038 
00039 /* default selectivity estimate for range inequalities "A > b AND A < c" */
00040 #define DEFAULT_RANGE_INEQ_SEL  0.005
00041 
00042 /* default selectivity estimate for pattern-match operators such as LIKE */
00043 #define DEFAULT_MATCH_SEL   0.005
00044 
00045 /* default number of distinct values in a table */
00046 #define DEFAULT_NUM_DISTINCT  200
00047 
00048 /* default selectivity estimate for boolean and null test nodes */
00049 #define DEFAULT_UNK_SEL         0.005
00050 #define DEFAULT_NOT_UNK_SEL     (1.0 - DEFAULT_UNK_SEL)
'''


DEFAULT_EQ_SEL = 0.005
DEFAULT_INEQ_SEL = 0.3333333333333333
DEFAULT_RANGE_INEQ_SEL = 0.005
DEFAULT_MATCH_SEL = 0.005
DEFAULT_NUM_DISTINCT = 200
DEFAULT_UNK_SEL = 0.005
DEFAULT_NOT_UNK_SEL = (1.0 - DEFAULT_UNK_SEL)


def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() == 'SELECT':
            return True
    return False


def extract_from_part(parsed):
    from_seen = False
    for item in parsed.tokens:
        if item.is_group:
            for x in extract_from_part(item):
                yield x
        if from_seen:
            if is_subselect(item):
                for x in extract_from_part(item):
                    yield x
            elif item.ttype is Keyword and item.value.upper() in ['ORDER', 'GROUP', 'BY', 'HAVING', 'GROUP BY']:
                from_seen = False
                StopIteration
            else:
                yield item
        if item.ttype is Keyword and item.value.upper() == 'FROM':
            from_seen = True


def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                value = identifier.value.replace('"', '').lower()
                yield value
        elif isinstance(item, Identifier):
            value = item.value.replace('"', '').lower()
            yield value


def extract_tables(sql):
    # let's handle multiple statements in one sql string
    extracted_tables = []
    statements = list(sqlparse.parse(sql))
    for statement in statements:
        if statement.get_type() != 'UNKNOWN':
            stream = extract_from_part(statement)
            extracted_tables.append(set(list(extract_table_identifiers(stream))))
    return list(itertools.chain(*extracted_tables))


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

# [[Filters],[Join Filter],[Cond], [Relation_alias],[selectivity], [w2c]]
def travesal_SQL(sql):
    def add_val(SQL_encode, t, att, val):
        assert val >= 0
        if SQL_encode[3].get(t):
            t = SQL_encode[3].get(t)

        if SQL_encode[4].get('.'.join([t, att])):
            count = 0
            for name in SQL_encode[3].values():
                if name == t:
                    count += 1
            if count > 1:
                SQL_encode[4]['.'.join([t, att])] = SQL_encode[4]['.'.join([t, att])] + val
            else:
                SQL_encode[4]['.'.join([t, att])] = SQL_encode[4]['.'.join([t, att])] * val
        else:
            SQL_encode[4]['.'.join([t, att])] = val

    SQL_encode = [[], [], [], {}, {}, [], []]
    # all table name and alias
    all_tables = extract_tables(sql)

    for table in all_tables:
        if ' ' in table:
            a, b = table.split(' ')
            SQL_encode[3][b] = a
        else:
            SQL_encode[3][table] = table

    # join relation and selecvtivity
    # 单独处理between 和 IS
    for token in sqlparse.parse(sql)[0].tokens:
        if isinstance(token, Where):
            lst = [one.value for one in token.tokens if (one.ttype is not Whitespace) and (one.value != '\n') ]
            for idx, one in enumerate(lst):
                if 'BETWEEN' == one:
                    result = lst[idx-1:idx+4]
                    t, att = result[0].split('.')
                    result[0] = '.'.join([SQL_encode[3][t], att])
                    add_val(SQL_encode, t, att, _get_selectivity(SQL_encode[3][t], ' '.join(result)))
                    #val1 = get_scalar_selectivity((att, '<', result[2].replace("'", "")), SQL_encode[3][t])[2]
                    #val2 = get_scalar_selectivity((att, '>', result[4].replace("'", "")), SQL_encode[3][t])[2]
                    #add_val(SQL_encode, t, att, (1 - val2)-val1)
                if 'IS' == one:
                    if 'NOT ' not in lst[idx+1]:
                        result = lst[idx-1:idx+2]
                        t, att = result[0].split('.')
                        result[0] = '.'.join([SQL_encode[3][t], att])
                        add_val(SQL_encode, t, att, _get_selectivity(SQL_encode[3][t], ' '.join(result)))
                    else:
                        result = lst[idx - 1:idx + 2]
                        t, att = result[0].split('.')
                        result[0] = '.'.join([SQL_encode[3][t], att])
                        add_val(SQL_encode, t, att, _get_selectivity(SQL_encode[3][t], ' '.join(result)))
                if 'ANY' == one:
                    result = lst[idx-2:idx+2]
                    t, att = result[0].split('.')
                    result[0] = '.'.join([SQL_encode[3][t], att])
                    add_val(SQL_encode, t, att, _get_selectivity(SQL_encode[3][t], ' '.join(result)))


    for token in sqlparse.parse(sql)[0].tokens:
        if isinstance(token, Where):
            for one in token.tokens:
                # print(type(one), one.ttype, one.value)
                if one.value == 'WHERE' or (one.ttype is Whitespace) or (one.ttype is Newline):
                    continue
                if isinstance(one, Comparison):
                    if (' =' in one.value or ' !=' in one.value or ' <>' in one.value) and (' >=' not in one.value) and (' <=' not in one.value):
                        a, b = re.split(' =| !=| <>', one.value)
                        a = a.strip()
                        b = b.strip()
                        if ("'" not in b) and ("." in b):
                            SQL_encode[2].append([a, b])
                        else:
                            t, att = a.split('.')
                            change = one.value.split('.')
                            change[0] = SQL_encode[3][t]
                            add_val(SQL_encode, t, att, _get_selectivity(SQL_encode[3][t], '.'.join(change)))

                    else:

                        one = one.value

                        t, att = one.split(' ')[0].split('.')
                        change = one.split('.')
                        change[0] = SQL_encode[3][t]
                        val = _get_selectivity(SQL_encode[3][t.replace('(', '').strip()], '.'.join(change))
                        add_val(SQL_encode, t, att, val)
                    '''
                    #elif (' >= ' in one.value) or (' <= ' in one.value) or (' < ' in one.value) or (' > ' in one.value):
                    #    a, b = re.split(' >= | <= | < | > ', one.value)
                    #    a = a.strip()
                    #    t, att = a.split('.')
                    #    add_val(SQL_encode, t, att, _get_selectivity(t, one.value))

                    #elif ' like ' in one.value or ' LIKE ' in one.value:
                    #    if ' not like ' in one.value or ' NOT LIKE ' in one.value:
                    #        a, b = re.split(' not like | NOT LIKE ', one.value)
                    #    else:
                    #       a, b = re.split(' like | LIKE ', one.value)

                    #    t, att = a.split('.')
                    #    if "'" in b:
                    #        b = b.replace("'", "")
                    #    if b[0] == '%':
                    #        if ' not like ' in one.value or ' NOT LIKE ' in one.value:
                    #            add_val(SQL_encode, t, att, 1 - DEFAULT_MATCH_SEL)
                    #        else:
                    #            add_val(SQL_encode, t, att, DEFAULT_MATCH_SEL)
                    #    else:
                    #        if ' not like ' in one.value or ' NOT LIKE ' in one.value:
                    #            add_val(SQL_encode, t, att, 1 - get_scalar_selectivity((att, 'like', b), SQL_encode[3][t])[2])
                    #        else:
                    #            add_val(SQL_encode, t, att,
                    #                    get_scalar_selectivity((att, 'like', b), SQL_encode[3][t])[2])

                    elif ' in ' in one.value or ' IN ' in one.value:
                        a, b = re.split(' in | IN ', one.value)
                        t, att = a.split('.')
                        val = 0
                        conds = b.replace(' ', '').replace('\n', '').replace('(', "").replace(')', "").split(',')
                        conds = [i.replace("'", "") for i in conds]
                        for cond in conds:
                            val += get_equal_selectivity((att, '=', cond), 0, SQL_encode[3][t])[2]
                        add_val(SQL_encode, t, att, val)
                '''


                elif isinstance(one, Parenthesis) and (' AND ' in one.value or ' OR ' in one.value):
                    lst = [i.replace('(', '').replace(')', '').replace('\n', '').strip() for i in re.split(' OR | AND ', one.value)]
                    # 增加替換別名
                    for i in range(len(lst)):
                        cur = lst[i]
                        change = cur.split('.')
                        change[0] = SQL_encode[3][change[0]]
                        lst[i] = '.'.join(change)

                    lst_1 = re.findall(' OR | AND ', one.value)
                    title = {}
                    delete = []
                    for i in range(len(lst)):
                        # t_att = lst[i].split(' ')[0]
                        now = lst[i].split(' ')
                        if len(now) == 1:
                            t_att = lst[i].split('=')[0]
                        else:
                            t_att = now[0]
                        if title.get(t_att):
                            title[t_att] = title[t_att] + lst_1[i-1] + lst[i]
                            delete.append(i-1)
                        else:
                            title[t_att] = lst[i]
                    lst = list(title.values())
                    lst_1 = [i for idx, i in enumerate(lst_1) if idx not in delete]


                    for idx, one in enumerate(lst):
                        if (idx == 0):
                            now = one.split(' ')[0]
                            if '=' in now:
                                t, att = now.split('=')[0].split('.')
                            else:
                                t, att = now.split('.')

                            val = _get_selectivity(t.replace('(','').strip(), one)
                            add_val(SQL_encode, t, att, val)
                        else:
                            relationship = lst_1.pop(0)
                            t, att = one.split(' ')[0].split('.')
                            val = _get_selectivity(t.replace('(','').strip(), one)
                            if 'OR' in relationship:
                                if SQL_encode[4].get('.'.join([t, att])):
                                    SQL_encode[4]['.'.join([t, att])] += val
                                else:
                                    SQL_encode[4]['.'.join([t, att])] = val
                            else:
                                if SQL_encode[4].get('.'.join([t, att])):
                                    SQL_encode[4]['.'.join([t, att])] = SQL_encode[4]['.'.join([t, att])] * val
                                else:
                                    SQL_encode[4]['.'.join([t, att])] = val

    return SQL_encode

# {'movie_companies.note': 0.5092998468071146, 'keyword.keyword': 8.943877170753522e-05, 'company_type.kind': 0.75, 'company_name.country_code': 0.9991412812582782, 'company_name.name': 0.01, 'title.production_year': 0.9717}


def process(one, SQL_encode):
    t = att = None
    if ' like ' in one or ' LIKE ' in one:
        if ' not like ' in one or ' NOT LIKE ' in one:
            a, b = re.split(' not like | NOT LIKE ', one)
            if "'" in b:
                b = b.replace("'", "")
            t, att = a.split('.')
            if b[0] == '%':
                val = 1 - DEFAULT_MATCH_SEL
            else:
                val = 1 - get_scalar_selectivity((att, 'like', b), SQL_encode[3][t])[2]
        else:
            a, b = re.split(' like | LIKE ', one)
            if "'" in b:
                b = b.replace("'", "")
            t, att = a.split('.')
            if b[0] == '%':
                val = DEFAULT_MATCH_SEL
            else:
                # print(one)
                val = get_scalar_selectivity((att, 'like', b), SQL_encode[3][t])[2]
                # print(val)

    elif ('=' in one or '!=' in one or '<>' in one) and ('>=' not in one) and ('<=' not in one):
        a, b = re.split('=|!=|<>', one)
        a = a.strip()
        b = b.strip()
        if "'" in b:
            b = b.replace("'", "")
        t, att = a.split('.')
        if '=' in one and '!=' not in one:
            val = get_equal_selectivity((att, '=', b), 0, SQL_encode[3][t])[2]
        else:
            val = get_equal_selectivity((att, '=', b), 1, SQL_encode[3][t])[2]

    elif (' >= ' in one) or (' <= ' in one) or (' < ' in one) or (' > ' in one):
        a, b = re.split(' >= | <= | < | > ', one)
        a = a.strip()
        b = b.strip()
        if "'" in b:
            b = b.replace("'", "")
        t, att = a.split('.')
        op = re.findall('>=|<=|<|>', one)
        assert len(op) == 1
        val = get_scalar_selectivity((att, op[0], b), SQL_encode[3][t])[2]

    elif ' IS ' in one:
        lst = one.split(' ')
        if 'NOT' not in lst:
            t, att = lst[0].split('.')
            val = get_equal_selectivity((att, '=', lst[-1]), 0, SQL_encode[3][t])[2]
        else:
            t, att = lst[0].split('.')
            val = get_equal_selectivity((att, '=', lst[-1]), 1, SQL_encode[3][t])[2]

    elif ' in ' in one or ' IN ' in one:
        a, b = one.split(' in | IN ')
        t, att = a.split('.')
        val = 0
        conds = b.replace(' ', '').replace('\n', '').replace('(', "").replace(')', "").split(',')
        conds = [i.replace("'", "") for i in conds]
        for cond in conds:
            val += get_equal_selectivity((att, '=', cond), 0, SQL_encode[3][t])[2]


    return val, (t, att)

def _get_selectivity(table_name, clause):
    cur = conn.conn.cursor()
    sql_command = 'EXPLAIN (format json) select * from ' + table_name +  ' WHERE ' + clause
    cur.execute(sql_command)
    query_plan = cur.fetchone()[0][0]
    return query_plan['Plan']['Plan Rows'] / all_tables_rows_num[table_name]


if __name__ == '__main__':
    # dir_list = os.listdir("./test_sql/job/")
    # number = "0123456789"
    # dir_list = [i for i in dir_list if i[0] in number]

    # USEABLE_FILE = [101400]
    # USEABLE_FILE = [5]
    # USEABLE_FILE = [101403]

    dirs = os.listdir('./')
    for idx, sql in enumerate(dirs):

        print('idx:', idx)
        sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
        print(sql)

        q = travesal_SQL(sql)[4]
        print(q)
        print(len(q))
        print('*'*50)

                # print(extract_tables(sql))
                # print('*'*50)