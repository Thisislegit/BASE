import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from TreeConvolution import policy_tcnn
from utils.Connection import all_tables
import random
from utils.DNNRLGOO import DNNRLGOO
from utils.PlanTree import *
from utils.Utils import *
# from utils.PlanSQL import travesal_SQL
from Plan_SQL_test import travesal_SQL
from utils.ExperienceBuffer import PrioritizedReplayBuffer
from sklearn.cluster import DBSCAN, KMeans

EXECUTION_MAX_TIME = 90000
GOAL_FIELD = 'Total Cost'  # Actual Total Time，Total Cost
GAMMA = 0.9
NORMALIZATION = 2
TEST_FILE = [84769, 87006, 55149, 25097, 64236, 98753, 48072, 6440, 27378, 48185, 92664, 16325, 84093, 75202, 68153,
             55524, 5121, 83811, 47439, 20268, 92855, 44727, 101212, 84454, 28432, 34103, 37796, 40097, 11349, 22318,
             99675, 44883, 11667, 54974, 47054, 28690, 13770, 458, 54393, 89689, 19780, 17207, 82217, 40672, 58875,
             37306, 83340, 5392, 7751, 17681, 59265, 70580, 58643, 63722, 94566, 40259, 27291, 17928, 43320, 62387,
             70075, 94054, 10893, 1081, 94931, 29891, 63817, 7591, 95293, 43924, 59149, 73094, 10154, 40180, 80204,
             20916, 8101, 3117, 75034, 54907, 4641, 76214, 27851, 73197, 2580, 59027, 69446, 18021, 94306, 3882, 42324,
             3538, 40466, 97383, 36002, 24201, 60625, 52101, 33716, 79841]
TRAIN_JOB = os.listdir('test_sql/job/')


class DNNRLGOO_Pi(DNNRLGOO):
    def __init__(self, model, optimizer, table, query_encode, selectivity, join_table, file_name, table_alias,
                 target_model, test=False, random_num_=0):
        super().__init__(model, optimizer, table, query_encode, selectivity, join_table, file_name, table_alias,
                         target_model, test)
        self.first_state = [self.query_encode, (tuple([0] * (21 * 2 + 3)),)]
        self.choice2vector = {(0, None): 0, (1, None): 1, (0, 0): 2, (1, 0): 3, (0, 1): 4, (1, 1): 5, (0, 2): 6,
                              (1, 2): 7}
        self.vector2choice = dict([val, key] for key, val in self.choice2vector.items())
        self.table2alias = dict([val, key] for key, val in self.table_alias.items())
        self.saved_log_probs = []
        self.first_relation = []
        self.entropy = []
        self.random_num_ = random_num_

    def init_state(self):
        start = [1]
        probs = self.value_net(([self.first_state], [self.tables], start))
        m = Categorical(probs)
        self.entropy.append(float(m.entropy()))
        # action_idx = torch.argmax(probs, dim=1)
        if self.test:
            action_idx = torch.argmax(probs, dim=1)
        else:
            action_idx = m.sample()
            # print(probs)
            # print(probs[0][action_idx])
            # print('*')
        self.saved_log_probs.append(m.log_prob(action_idx))
        table_idx = action_idx // len(self.choice2vector)
        # ["Hash Join", "Merge Join", "Nested Loop"]

        if random.random() < self.random_num_:
            table_availble_ = [all_tables[table] for table in self.tables]
            table_idx = random.choice(table_availble_)

        idx, join_type = self.vector2choice[int(action_idx) % len(self.choice2vector)]
        assert join_type == None
        vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))
        vector_parent[len(self.join_type) + table_idx * 2] = 1
        vector_parent[len(self.join_type) + 1 + table_idx * 2] = idx
        self.total_vector = vector_parent.copy()
        self.collection = (tuple(vector_parent),)
        self.state_list.append((tuple(vector_parent),))
        # self.first_relation.append(list(all_tables.keys())[int(table_idx)])
        alias_name = self.table2alias[list(all_tables.keys())[int(table_idx)]]
        self.first_relation.append(alias_name)
        if idx == 1:
            self.add_hint_index(alias_name)
        self.hint_leading.append(alias_name)
        self.used_table += [alias_name]

        self.check_state()

    def choose_action(self):
        start = [0]
        operator_num = len(self.join_type)
        now_need_join_table = []
        for i in range(len(self.join_table)):
            # delete redundant join relationship
            current = [relation[0] for relation in now_need_join_table]
            if self.join_table[i][0] in self.used_table and \
                    self.join_table[i][1] not in self.used_table:
                if self.join_table[i][1] not in current:
                    now_need_join_table.append([self.join_table[i][1], i])
            if self.join_table[i][1] in self.used_table and \
                    self.join_table[i][0] not in self.used_table:
                if self.join_table[i][0] not in current:
                    now_need_join_table.append([self.join_table[i][0], i])

        # table_availbale = [[i[0] for i in now_need_join_table]]
        table_availbale = [[self.table_alias[i[0]] for i in now_need_join_table]]
        probs = self.value_net(([[self.query_encode, self.state_list[-1]]], table_availbale, start))
        m = Categorical(probs)
        self.entropy.append(float(m.entropy()))
        if self.test:
            action_idx = torch.argmax(probs, dim=1)
        else:
            action_idx = m.sample()
            # print(probs)
            # print(probs[0][action_idx])
            # print('*')
        self.saved_log_probs.append(m.log_prob(action_idx))
        table_idx = int(action_idx) // len(self.choice2vector)
        idx, join_type = self.vector2choice[int(action_idx) % len(self.choice2vector)]

        # ["Hash Join", "Merge Join", "Nested Loop"]
        num_ = random.random()
        if num_ < self.random_num_:
            table_availble_ = [all_tables[table] for table in table_availbale[0]]
            table_idx = random.choice(table_availble_)
            join_type = 0
        vector_right = [0] * (len(all_tables) * 2 + operator_num)
        vector_right[len(self.join_type) + table_idx * 2] = 1
        vector_right[len(self.join_type) + 1 + table_idx * 2] = idx
        tuple_right = (tuple(vector_right),)
        vector_parent = self.total_vector.copy()
        vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
        vector_parent[operator_num + table_idx * 2] = 1
        vector_parent[operator_num + 1 + table_idx * 2] = idx
        vector_parent[join_type] = 1

        self.total_vector = vector_parent
        self.state_list.append((tuple(vector_parent), self.collection, tuple_right))
        self.collection = (tuple(vector_parent), self.collection, tuple_right)
        # alias_name = self.table2alias[list(all_tables.keys())[table_idx]]
        for table, idx in now_need_join_table:
            if self.table_alias[table] == list(all_tables.keys())[table_idx]:
                alias_name = table
        join_index = [i[0] for i in now_need_join_table].index(alias_name)
        if idx == 1:
            self.add_hint_index(alias_name)
        self.add_hint_join(alias_name, join_type)
        self.used_table.append(alias_name)
        self.join_table.pop(now_need_join_table[join_index][1])

        self.check_state()


class _QueryEncoder(nn.Module):
    def __init__(self, input_size=318):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.act = nn.ReLU()


    def forward(self, goal):
        # self.local_device = next(self.hidden1.parameters()).device
        # self.local_device = torch.device('cuda:' + str(0))
        self.local_device = next(self.hidden1.parameters()).device
        encode = []
        for i in goal:
            encode.append(torch.tensor(i, dtype=torch.float32, device=self.local_device).unsqueeze(0))
        x = torch.cat(encode, dim=0)
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.hidden3(x))

        return x


class Discrim_net(nn.Module):
    def __init__(self):
        super(Discrim_net, self).__init__()
        self.encode = policy_tcnn.QueryEncoder(318)
        self.BTC1 = policy_tcnn.BinaryTreeConv(109, 512)
        self.BTC2 = policy_tcnn.BinaryTreeConv(512, 256)
        self.BTC3 = policy_tcnn.BinaryTreeConv(256, 128)
        self.TreeNorm = policy_tcnn.TreeLayerNorm()
        self.TreeAct = policy_tcnn.TreeActivation(nn.ReLU())
        self.DP = policy_tcnn.DynamicPooling()

        self.hidden1 = nn.Linear(128, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.hidden4 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.encode(x)
        x = self.BTC1(x)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.BTC2(x)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.BTC3(x)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.DP(x)

        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        # x = self.activation(self.hidden4(x))
        x = self.hidden4(x)
        return x


class Pi_net(nn.Module):
    def __init__(self):
        super(Pi_net, self).__init__()
        self.encode = policy_tcnn.QueryEncoder(318)
        self.BTC1 = policy_tcnn.BinaryTreeConv(109, 512)
        self.BTC2 = policy_tcnn.BinaryTreeConv(512, 256)
        self.BTC3 = policy_tcnn.BinaryTreeConv(256, 128)
        self.TreeNorm = policy_tcnn.TreeLayerNorm()
        self.TreeAct = policy_tcnn.TreeActivation(nn.ReLU())
        self.DP = policy_tcnn.DynamicPooling()
        self.Pi = Pi(128)

    def forward(self, x):
        now, available_table, start = self.encode(x)
        x = self.BTC1(now)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.BTC2(x)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.BTC3(x)
        x = self.TreeNorm(x)
        x = self.TreeAct(x)
        x = self.DP(x)
        x = self.Pi((x, available_table, start))
        return x


class Pi(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        # table_num * 2 (index) * 2 (join_type) = 21 * (2 * 4) = 168
        self.hidden4 = nn.Linear(32, 168)
        self.activation = nn.ReLU()
        self.out = nn.Softmax(dim=1)
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x):
        x, available_table, start = x
        x = self.drop_layer(x)
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.drop_layer(x)
        x = self.activation(self.hidden3(x))
        x = self.hidden4(x)
        x = self._mask_out(x, available_table, start)
        x = self.out(x)
        return x

    def _mask_out(self, x, table_available, start):
        mask = torch.ones_like(x)

        for idx, one_batch in enumerate(table_available):
            if not start[idx]:
                for one_table in one_batch:
                    start_ = all_tables[one_table]
                    mask[idx, start_ * 8 + 2:(start_ + 1) * 8] = 0
            else:
                for one_table in one_batch:
                    start_ = all_tables[one_table]
                    mask[idx, start_ * 8:start_ * 8 + 2] = 0
        return x + mask.masked_fill(mask == 1, float('-inf'))
        # todo: (1 - mask) * x + mask * mask_fill


def save_train_data():
    def to_one_hot(list):
        choice2vector = {(0, None): 0, (1, None): 1, (0, 0): 2, (1, 0): 3, (0, 1): 4, (1, 1): 5, (0, 2): 6, (1, 2): 7}
        result = []
        for i in list:
            relation_name, index, join_type = i
            one = choice2vector[(index, join_type)]
            current = [0] * (21 * len(choice2vector))
            current[all_tables[relation_name] * len(choice2vector) + one] = 1
            result.append(current)
        return result

    # if not os.path.exists('./train_plan/Imitation_Learning/x_expert_train.npy'):
    x_train = []
    x_train_table = []
    x_train_start = []
    y_train = []
    x_test = []
    x_test_table = []
    x_test_start = []
    y_test = []
    problem_plan = []
    USEABLE_FILE = []
    for file in sorted(os.listdir('./train_plan/index/')):
        with open('./train_plan/index/' + file) as f:
            for idx in f.readlines():
                USEABLE_FILE.append(int(idx))
    USEABLE_FILE.sort()
    # problem = np.load('./train_plan/Neo_baseline/problem_plan.npy').tolist()
    # USEABLE_FILE = [i for i in USEABLE_FILE if i not in problem]
    for num, idx in enumerate(USEABLE_FILE):
        print(idx)
        sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
        sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
        SQL_encode = travesal_SQL(sql)
        query_encode = SQL_encode
        join_table = get_join_table(query_encode[2])
        join_table = reverse_kv(query_encode[3], join_table)
        query_encode = replace_alias(query_encode)

        join_table = remove_same_table(join_table, 7)
        now_tables = get_tables(query_encode)
        query_vector = get_vector(query_encode)

        terminate_plan = get_hint_SQL_explain(sql,
                                              'EXPLAIN (format json)',
                                              str(11),
                                              conn.conn)
        try:
            result_now = _traversal_plan_tree_cost(terminate_plan['Plan'], f, query_vector, None)
        except:
            print('problem plan: {}'.format(idx))
            problem_plan.append(idx)
            with open('{}.json'.format(idx), 'w') as f:
                json.dump(terminate_plan, f)
            assert 0
            continue
        current = []
        order = [one[0] for one in result_now[3]]
        for i in range(len(order)):
            current.append(order[i:])
        assert len(result_now[3]) + 1 == len(result_now[2]) == len(current) + 1
        if idx not in TEST_FILE:
            x_train.extend(result_now[2][:-1])
            x_train_table.extend(current)
            x_train_start.extend([1] + [0] * (len(current) - 1))
            y_train.extend(to_one_hot(result_now[3]))
            # y_train.extend([round(pow(terminate_plan['Plan']['Total Cost'], 1 / NORMALIZATION), 6)] * len(result_now[2]))
        else:
            x_test.extend(result_now[2][:-1])
            x_test_table.extend(current)
            x_test_start.extend([1] + [0] * (len(current) - 1))
            y_test.extend(to_one_hot(result_now[3]))
            # y_test.extend([round(pow(terminate_plan['Plan']['Total Cost'], 1 / NORMALIZATION), 6)] * len(result_now[2]))
    assert 0
    assert len(x_train) == len(y_train) == len(x_train_table) == len(x_train_start)
    assert len(x_test) == len(y_test) == len(x_test_table) == len(x_test_start)
    x_train = np.array(x_train)
    x_train_start = np.array(x_train_start)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_test_start = np.array(x_test_start)
    y_test = np.array(y_test)
    np.save('./train_plan/Imitation_Learning/problem_plan.npy', np.array(problem_plan))
    np.save('./train_plan/Imitation_Learning/x_expert_train.npy', x_train)
    np.save('./train_plan/Imitation_Learning/x_expert_train_start.npy', x_train_start)
    np.save('./train_plan/Imitation_Learning/y_expert_train.npy', y_train)
    np.save('./train_plan/Imitation_Learning/x_expert_test.npy', x_test)
    np.save('./train_plan/Imitation_Learning/x_expert_test_start.npy', x_test_start)
    np.save('./train_plan/Imitation_Learning/y_expert_test.npy', y_test)
    with open('./train_plan/Imitation_Learning/x_expert_train_table.json', 'w') as f:
        json.dump(x_train_table, f)
    with open('./train_plan/Imitation_Learning/x_expert_test_table.json', 'w') as f:
        json.dump(x_test_table, f)

    print('There are {} training expert experiences, all of them are saved!'.format(len(x_train)))
    print('There are {} testing expert experiences, all of them are saved!'.format(len(x_test)))
    assert 0

    # else:
    #    x_train = np.load('./train_plan/Imitation_Learning/x_expert_train.npy', allow_pickle=True)
    #    x_train_table = json.load(open('./train_plan/Imitation_Learning/x_expert_train_table.json'), encoding='utf-8')
    #    x_train_start = np.load('./train_plan/Imitation_Learning/x_expert_train_start.npy')
    #    y_train = np.load('./train_plan/Imitation_Learning/y_expert_train.npy', allow_pickle=True)
    #    x_test = np.load('./train_plan/Imitation_Learning/x_expert_test.npy', allow_pickle=True)
    #    x_test_table = json.load(open('./train_plan/Imitation_Learning/x_expert_test_table.json'), encoding='utf-8')
    #    x_test_start = np.load('./train_plan/Imitation_Learning/x_expert_test_start.npy')
    #    y_test = np.load('./train_plan/Imitation_Learning/y_expert_test.npy', allow_pickle=True)

    return x_train, x_train_table, x_train_start, y_train, x_test, x_test_table, x_test_start, y_test


def _traversal_plan_tree_latency(plan, f, query_vector, first_relation):
    GOAL_FIELD = 'Actual Total Time'  # Actual Total Time，Total Cost
    vector = [0] * (len(all_tables) * 2 + 3)
    if 'Plans' in plan.keys() and plan['Node Type'] not in SCAN_TYPE:
        if len(plan['Plans']) > 1:
            if plan['Node Type'] == 'Sort' or plan['Node Type'] == 'Aggregate':
                if f == 11 or f == 17:
                    node = _traversal_plan_tree_latency(plan['Plans'][0], f, query_vector)
                    return node
                elif f == 22:
                    node = _traversal_plan_tree_latency(plan['Plans'][1], f, query_vector)
                    return node
            else:
                if plan['Node Type'] in JOIN_TYPE:
                    vector[JOIN_TYPE.index(plan['Node Type'])] = 1
                else:
                    vector[JOIN_TYPE.index('Nested Loop')] = 1
                left = _traversal_plan_tree_latency(plan['Plans'][0], f, query_vector, first_relation)
                right = _traversal_plan_tree_latency(plan['Plans'][1], f, query_vector, first_relation)
                assert len(left[2]) == 0 or len(right[2]) == 0
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
                # if vector[:3] == [0, 1, 0] and not(plan['Total Cost'] > plan['Plans'][0]['Total Cost'] and plan['Total Cost'] > plan['Plans'][1]['Total Cost']):
                #     # for merge join
                #     cost.append((plan[GOAL_FIELD], calcualte_merge_join_cost(plan)))
                # else:
                #     cost.append(plan[GOAL_FIELD])
                cost.append(plan[GOAL_FIELD])

                state = []
                state = state + left[2]
                state = state + right[2]
                if len(left[1]) == len(right[1]) == 1:
                    assert len(left[2]) == len(right[2]) == 0
                    state.append([query_vector, (tuple([0] * (21 * 2 + 3)),)])
                    state.append([query_vector, vector_left])
                state.append([query_vector, (tuple(vector), vector_left, vector_right)])

                choice = []
                assert len(left[3]) == 1 or len(right[3]) == 1
                if len(right[3]) == 1:
                    a, b, c = right[3][0]
                    right[3] = [(a, b, vector[:3].index(1))]
                    if len(left[3]) == 1:
                        a, b, c = left[3][0]
                        # 第一个没有join关系的操作看成 None
                        left[3] = [(a, b, None)]
                    choice = choice + left[3]
                    choice = choice + right[3]
                else:
                    a, b, c = left[3][0]
                    left[3] = [(a, b, vector[:3].index(1))]
                    choice = choice + right[3]
                    choice = choice + left[3]
                return [cost, (tuple(vector), vector_left, vector_right), state, choice]
        else:
            node = _traversal_plan_tree_latency(plan['Plans'][0], f, query_vector, first_relation)
            return node
    elif plan['Node Type'] in SCAN_TYPE:
        index = 0
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan[
            'Node Type'] == 'Bitmap Heap Scan':  # ? index change to heap
            vector[3 + all_tables[
                plan['Relation Name']] * 2] = 1  # plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[4 + all_tables[plan['Relation Name']] * 2] = 1
            index = 1
        elif plan['Node Type'] == 'Bitmap Index Scan':
            pass
        else:
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1
        if plan['Alias'] == first_relation:
            cost = plan[GOAL_FIELD]
            return [[cost], (tuple(vector),), [], [(plan['Relation Name'], index, None)]]
        else:
            return [[], (tuple(vector),), [], [(plan['Relation Name'], index, None)]]


    elif 'Plan' in plan.keys():
        node = _traversal_plan_tree_latency(plan['Plan'][0], f, query_vector)
        return node
    else:
        return []


def _traversal_plan_tree_cost(plan, f, query_vector, first_relation):
    vector = [0] * (len(all_tables) * 2 + 3)
    if 'Plans' in plan.keys() and plan['Node Type'] not in SCAN_TYPE:
        if len(plan['Plans']) > 1:
            if plan['Node Type'] == 'Sort' or plan['Node Type'] == 'Aggregate':
                if f == 11 or f == 17:
                    node = _traversal_plan_tree_cost(plan['Plans'][0], f, query_vector)
                    return node
                elif f == 22:
                    node = _traversal_plan_tree_cost(plan['Plans'][1], f, query_vector)
                    return node
            else:
                if plan['Node Type'] in JOIN_TYPE:
                    vector[JOIN_TYPE.index(plan['Node Type'])] = 1
                else:
                    vector[JOIN_TYPE.index('Nested Loop')] = 1
                left = _traversal_plan_tree_cost(plan['Plans'][0], f, query_vector, first_relation)
                right = _traversal_plan_tree_cost(plan['Plans'][1], f, query_vector, first_relation)
                assert len(left[2]) == 0 or len(right[2]) == 0
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
                if vector[:3] == [0, 1, 0] and not (
                        plan['Total Cost'] > plan['Plans'][0]['Total Cost'] and plan['Total Cost'] > plan['Plans'][1][
                    'Total Cost']):
                    # for merge join
                    cost.append((plan[GOAL_FIELD], calcualte_merge_join_cost(plan)))
                else:
                    cost.append(plan[GOAL_FIELD])

                state = []
                state = state + left[2]
                state = state + right[2]
                if len(left[1]) == len(right[1]) == 1:
                    assert len(left[2]) == len(right[2]) == 0
                    state.append([query_vector, (tuple([0] * (21 * 2 + 3)),)])
                    state.append([query_vector, vector_left])
                state.append([query_vector, (tuple(vector), vector_left, vector_right)])

                choice = []
                assert len(left[3]) == 1 or len(right[3]) == 1
                if len(right[3]) == 1:
                    a, b, c = right[3][0]
                    right[3] = [(a, b, vector[:3].index(1))]
                    if len(left[3]) == 1:
                        a, b, c = left[3][0]
                        # 第一个没有join关系的操作看成 None
                        left[3] = [(a, b, None)]
                    choice = choice + left[3]
                    choice = choice + right[3]
                else:
                    a, b, c = left[3][0]
                    left[3] = [(a, b, vector[:3].index(1))]
                    choice = choice + right[3]
                    choice = choice + left[3]
                return [cost, (tuple(vector), vector_left, vector_right), state, choice]
        else:
            node = _traversal_plan_tree_cost(plan['Plans'][0], f, query_vector, first_relation)
            return node
    elif plan['Node Type'] in SCAN_TYPE:
        index = 0
        if plan['Node Type'] == 'Index Scan' or plan['Node Type'] == 'Index Only Scan' or plan[
            'Node Type'] == 'Bitmap Heap Scan':  # ? index change to heap
            vector[3 + all_tables[
                plan['Relation Name']] * 2] = 1  # plan['Plan Rows']/all_tables_rows_num[plan['Relation Name']]
            vector[4 + all_tables[plan['Relation Name']] * 2] = 1
            index = 1
        elif plan['Node Type'] == 'Bitmap Index Scan':
            pass
        else:
            vector[3 + all_tables[plan['Relation Name']] * 2] = 1

        if plan['Alias'] == first_relation:
            cost = plan[GOAL_FIELD]
            return [[cost], (tuple(vector),), [], [(plan['Relation Name'], index, None)]]
        else:
            return [[], (tuple(vector),), [], [(plan['Relation Name'], index, None)]]


    elif 'Plan' in plan.keys():
        node = _traversal_plan_tree_cost(plan['Plan'][0], f, query_vector)
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
    assert len(plan['Plans']) == 2
    child_plan_1 = plan['Plans'][0]
    child_plan_2 = plan['Plans'][1]
    if plan['Total Cost'] > child_plan_1['Total Cost'] and plan['Total Cost'] > child_plan_2['Total Cost']:
        print(child_plan_1['Node Type'], child_plan_2['Node Type'])
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
                    child_cost += child_plan['Plans'][0]['Total Cost'] - child_plan['Plans'][0]['Plans'][0][
                        'Total Cost']

            elif child_plan['Node Type'] in JOIN_TYPE:
                continue
            else:
                print(child_plan['Node Type'])
                assert 0
        return current_cost + child_cost


def train(model, criterion, optimizer, loader, use_gpu, idx, DEVICE):
    model.train()
    epoch_loss = 0
    x, x1, x2, y = loader
    length = 512
    i = 0
    while i < len(x):
        x_, x1_, x2_, y_ = x[idx[i: i + length]], x1[idx[i: i + length]], x2[idx[i: i + length]], y[idx[i: i + length]]
        print(y_[0])
        print(np.where(y_[0] == 1))
        y_ = [int(np.where(i == 1)[0]) for i in y_]
        optimizer.zero_grad()
        output = model((x_, x1_, x2_))
        loss = criterion(output, torch.tensor(y_, device=DEVICE).type(torch.long))
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0) #梯度裁减
        optimizer.step()
        if (use_gpu):
            loss = loss.cpu()
            print(loss)
        epoch_loss += loss.item()

        # print(loss)
        i += length
    return epoch_loss / (len(loader) / length)


def test_accuarcy(model, x, y):
    length = 512
    idx = np.arange(len(x[1]))
    i = 0
    x, x1, x2 = x
    x1 = np.array(x1)
    out = []
    while i < len(x1):
        x_, x1_, x2_ = x[idx[i: i + length]], x1[idx[i: i + length]], x2[idx[i: i + length]]
        y_ = torch.argmax(model((x_, x1_, x2_)), dim=1).tolist()
        out.extend(y_)
        i += length
        print(i)

    gt = np.where(y == 1)[1].tolist()

    count = 0
    for a, b in zip(out, gt):
        if a == b:
            count += 1
    print('accuarcy:', count / len(gt))


def test(model, job=False):
    def execute(sql, job=False):
        SQL_encode = travesal_SQL(sql)
        query_encode = SQL_encode
        join_table = get_join_table(query_encode[2])
        join_table = reverse_kv(query_encode[3], join_table)
        query_encode = replace_alias(query_encode)

        join_table = remove_same_table(join_table, 7)
        now_tables = get_tables(query_encode)
        query_vector = get_vector(query_encode)

        file_name = 1
        rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                         join_table, file_name, query_encode[3], model, test=True)
        # rl.random_num = dqn.epsilon_start
        rl.random_num = 0  # 测试的时候用
        rl.init_state()
        while rl.state:
            rl.choose_action()
        terminate_plan = get_hint_SQL_explain(sql,
                                              '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                              str(11),
                                              conn.conn)
        print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ')
        save_path = 'test_sql/test_result/policy_baseline/test_train/'
        if job:
            save_path = 'test_sql/test_result/policy_baseline/job/'

        with open(save_path + str(idx) + '.json', 'w') as file_obj:
            json.dump(terminate_plan, file_obj)

        print(idx, ' cost:', terminate_plan['Plan']['Total Cost'])

        # save_path = 'test_sql/executed_sql_plan/test_train/'
        terminate_plan = get_hint_SQL_explain(sql,
                                              ' EXPLAIN (format json)',
                                              str(11),
                                              conn.conn)

        save_path_2 = '/data0/chenx/imdb_QO/test_sql/executed_sql_plan/test_train_1/'
        with open(save_path_2 + str(idx) + '.json', 'w') as file_obj:
            json.dump(terminate_plan, file_obj)

        print(idx, ' cost:', terminate_plan['Plan']['Total Cost'])

        print('*' * 50)

        # assert 0

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()
    if job:
        dir_list = os.listdir("./test_sql/job/")
        number = "0123456789"
        dir_list = [i for i in dir_list if i[0] in number]
        for idx in dir_list:
            sql = loadSQL("./test_sql/job/" + idx)
            execute(sql, job=True)

    else:
        # USEABLE_FILE = []
        # for file in sorted(os.listdir('./train_plan/index/')):
        #     with open('./train_plan/index/' + file) as f:
        #         for idx in f.readlines():
        #             USEABLE_FILE.append(int(idx))
        # USEABLE_FILE.sort()

        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            execute(sql)


def interactive_training(model):
    def finish_episode(model, saved_log_probs, rewards, optimizer, batch_loss, batch_size=256):
        R = 0
        policy_loss = []
        returns = []
        gamma = 0.98
        # eps = np.finfo(np.float32).eps.item()
        rewards = [-i for i in rewards]
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=local_device)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        if len(batch_loss) < batch_size:
            batch_loss.extend(policy_loss)
            if len(batch_loss) < batch_size:
                return batch_loss

        batch_loss_ = torch.cat(batch_loss)
        # batch_loss_ = (batch_loss_ - batch_loss_.mean()) / (batch_loss_.std() + eps)
        optimizer.zero_grad()
        # policy_loss = torch.cat(batch_loss).sum()
        policy_loss = batch_loss_.sum()
        # print(float(policy_loss))
        current_loss.append(float(policy_loss))
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        return batch_loss

    def test(model, epoch):
        # save CheckPoint
        checkpoint_path = './Models/Model_Policy_CP/now_{}.pth'.format(epoch)
        state = {'model': model.state_dict()}
        torch.save(state, checkpoint_path)

        path = 'reward_log_policy_NEW.json'
        if os.path.exists(path):
            reward_log = json.load(open(path, encoding='utf-8'))
        else:
            reward_log = []
        all = []
        # TEST_FILE = [16325]
        # TEST_FILE = [7591]
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            # print(idx)

            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            file_name = 1
            rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                             join_table, file_name, query_encode[3], model, test=True)
            # rl.random_num = dqn.epsilon_start
            rl.random_num = 0  # 测试的时候用
            rl.init_state()
            while rl.state:
                rl.choose_action()
            terminate_plan = get_hint_SQL_explain(sql,
                                                  '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                  str(11),
                                                  conn.conn)
            all.append(float(terminate_plan['Plan']['Total Cost']))
        reward_log.append(sum(all) / len(TEST_FILE))
        with open(path, 'w') as file_obj:
            json.dump(reward_log, file_obj)

    USEABLE_FILE = []
    for file in sorted(os.listdir('./train_plan/index/')):
        with open('./train_plan/index/' + file) as f:
            for idx in f.readlines():
                USEABLE_FILE.append(int(idx))
    USEABLE_FILE.sort()
    optimizer = torch.optim.AdamW(params=net.parameters(), lr=0.0005)

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()
    batch_loss = []
    for epoch in range(100, 200):
        print('Current epoch:', epoch)
        all_loss = []
        current_loss = []
        # USEABLE_FILE = [7591]
        # TEST_FILE = []
        for idx in USEABLE_FILE:
            if idx not in TEST_FILE:
                sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
                sql = sqlparse.format(sql, reindent=True, keyword_case="upper")

                SQL_encode = travesal_SQL(sql)
                query_encode = SQL_encode
                join_table = get_join_table(query_encode[2])
                join_table = reverse_kv(query_encode[3], join_table)
                query_encode = replace_alias(query_encode)

                join_table = remove_same_table(join_table, 7)
                now_tables = get_tables(query_encode)
                query_vector = get_vector(query_encode)

                file_name = 1
                rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                                 join_table, file_name, query_encode[3], model, test=False)
                # rl.random_num = dqn.epsilon_start
                rl.random_num = 0  # 测试的时候用
                rl.init_state()
                while rl.state:
                    rl.choose_action()
                terminate_plan = get_hint_SQL_explain(sql,
                                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                      str(11),
                                                      conn.conn)

                # print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)')
                # print(sql)
                result_now = _traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])
                # print(result_now[0])

                if isinstance(result_now[0][0], tuple):
                    rewards = [round(pow(result_now[0][0][1], 1 / NORMALIZATION), 6)]
                else:
                    rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                for i in range(1, len(result_now[0])):
                    if (not isinstance(result_now[0][i], tuple) and (not isinstance(result_now[0][i - 1], tuple))):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                    elif (not isinstance(result_now[0][i], tuple) and (isinstance(result_now[0][i - 1], tuple))):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1][0], 1 / NORMALIZATION), 6))
                    else:
                        rewards.append(round(pow(result_now[0][i][1], 1 / NORMALIZATION), 6))
                result_now[0] = rewards

                batch_size = 256
                batch_loss = finish_episode(model, rl.saved_log_probs, rewards, optimizer, batch_loss, batch_size)
                if len(batch_loss) >= batch_size:
                    batch_loss = []

                if len(current_loss) > 0 and len(current_loss) % 100 == 0:
                    print('loss:', sum(current_loss) / len(current_loss))
                    all_loss.append(sum(current_loss) / len(current_loss))
                    current_loss = []

        # print('Average Loss:', sum(all_loss)/ len(all_loss))
        if epoch % 1 == 0:
            set_statement_timeout(EXECUTION_MAX_TIME)
            load_extension()
            test(model, epoch)


def get_entroy(model):
    cp_path = './Models/Model_Policy_CP/now_'
    path = 'Entropy_log_policy_NEW.json'

    Entropy_log = []
    for epoch in range(-1, 173, 1):
        if epoch == -1:
            cur_cp = torch.load('./Models/Policy_based.pth')
        else:
            cur_path = cp_path + str(epoch) + '.pth'
            cur_cp = torch.load(cur_path)
        model.load_state_dict(cur_cp['model'])
        entropy_log = []
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            # print(idx)
            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            file_name = 1
            rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                             join_table, file_name, query_encode[3], model, test=True)
            # rl.random_num = dqn.epsilon_start
            rl.random_num = 0  # 测试的时候用
            rl.init_state()
            while rl.state:
                rl.choose_action()
            entropy_log.append(sum(rl.entropy) / len(rl.entropy))
        assert len(entropy_log) == len(TEST_FILE)
        print('current epoch {}'.format(epoch), 'entropy is {}'.format(sum(entropy_log) / len(entropy_log)))
        Entropy_log.append(sum(entropy_log) / len(entropy_log))
    with open(path, 'w') as file_obj:
        json.dump(Entropy_log, file_obj)


def latency_test(model, job=False):
    def execute(idx, sql, k, job=False):
        SQL_encode = travesal_SQL(sql)
        query_encode = SQL_encode
        join_table = get_join_table(query_encode[2])
        join_table = reverse_kv(query_encode[3], join_table)
        query_encode = replace_alias(query_encode)

        join_table = remove_same_table(join_table, 7)
        now_tables = get_tables(query_encode)
        query_vector = get_vector(query_encode)

        file_name = 1
        if k == 0:
            random_num_ = 0
        else:
            random_num_ = 1 / 3
        rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                         join_table, file_name, query_encode[3], model,
                         test=False,
                         random_num_=random_num_)
        # rl.random_num = dqn.epsilon_start
        rl.random_num = 0  # 测试的时候用
        rl.init_state()
        while rl.state:
            rl.choose_action()
        terminate_plan = get_hint_SQL_explain(sql,
                                              '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                              str(11),
                                              conn.conn)
        print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ')
        save_path = 'test_sql/test_result/policy_baseline/test_train/'
        if job:
            save_path = 'test_sql/test_result/policy_baseline/job/'
        if terminate_plan != -1:
            # with open(save_path + str(idx) + '.json', 'w') as file_obj:
            #    json.dump(terminate_plan, file_obj)

            print(idx, ' Latency:', terminate_plan['Execution Time'])
            print(idx, ' Cost:', terminate_plan['Plan']['Total Cost'])
            print('Sum of entropy', sum(rl.entropy))
            print('Averrage entropy', sum(rl.entropy) / len(rl.entropy))
            # save_path = 'test_sql/executed_sql_plan/test_train/'
            terminate_plan = get_hint_SQL_explain(sql,
                                                  ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                                  str(11),
                                                  conn.conn)
            print('PG results:')
            print(idx, ' Latency:', terminate_plan['Execution Time'])
            print(idx, ' Cost:', terminate_plan['Plan']['Total Cost'])

            print('*' * 50)
        else:
            print('Plan {} get some problem!'.format(idx))
            print('Save cost!')

            # terminate_plan = get_hint_SQL_explain(sql,
            #                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
            #                                      str(11),
            #                                      conn.conn)
            # save_path = 'test_sql/test_result/policy_baseline/test_train/'
            # if job:
            #    save_path = 'test_sql/test_result/policy_baseline/job/'
            # with open(save_path + str(idx) + '.json', 'w') as file_obj:
            #    json.dump(terminate_plan, file_obj)

            # print('*' * 50)

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()
    if job:
        dir_list = os.listdir("./test_sql/job/")
        number = "0123456789"
        dir_list = [i for i in dir_list if i[0] in number]
        for idx in dir_list:
            sql = loadSQL("./test_sql/job/" + idx)
            execute(idx, sql, 0, job=True)

    else:
        TEST_FILE = [55149]
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            print(TEST_FILE)
            print(sql)
            for k in range(10):
                execute(idx, sql, k)


def PG_latency(job=False):
    def execute(idx, sql, job):
        terminate_plan = get_hint_SQL_explain(sql,
                                              ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                              str(11),
                                              conn.conn)
        print(idx, ' Latency:', terminate_plan['Execution Time'])
        if not job:
            save_path_2 = '/data0/chenx/imdb_QO/test_sql/executed_sql_plan/test_train_1/'
        else:
            save_path_2 = '/data0/chenx/imdb_QO/test_sql/executed_sql_plan/test_1'
        with open(save_path_2 + str(idx) + '.json', 'w') as file_obj:
            json.dump(terminate_plan, file_obj)

    print('*' * 50)
    print('Start to test PG latency')
    if not job:
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            execute(idx, sql, job)
    else:
        dir_list = os.listdir("./test_sql/job/")
        number = "0123456789"
        dir_list = [i for i in dir_list if i[0] in number]
        for idx in dir_list:
            sql = loadSQL("./test_sql/job/" + idx)
            execute(idx, sql, job)


def interactive_training_Transfer(model):
    def finish_episode(model, saved_log_probs, rewards, optimizer, batch_loss, batch_size=256):
        R = 0
        policy_loss = []
        returns = []
        gamma = 0.98
        # eps = np.finfo(np.float32).eps.item()
        rewards = [-i for i in rewards]
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=local_device)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        if len(batch_loss) < batch_size:
            batch_loss.extend(policy_loss)
            if len(batch_loss) < batch_size:
                return batch_loss

        batch_loss_ = torch.cat(batch_loss)
        # batch_loss_ = (batch_loss_ - batch_loss_.mean()) / (batch_loss_.std() + eps)
        optimizer.zero_grad()
        # policy_loss = torch.cat(batch_loss).sum()
        policy_loss = batch_loss_.sum()
        # print(float(policy_loss))
        current_loss.append(float(policy_loss))
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        return batch_loss

    def test(model, epoch):
        # save CheckPoint
        checkpoint_path = './Models/Model_Policy_Transfer/now_{}.pth'.format(epoch)
        state = {'model': model.state_dict()}
        torch.save(state, checkpoint_path)

        path = 'reward_log_policy_Transfer_Execution_Time.json'
        if os.path.exists(path):
            reward_log = json.load(open(path, encoding='utf-8'))
        else:
            reward_log = []
        all = []
        # TEST_FILE = [16325]
        # TEST_FILE = [7591]
        TEST_FILE = [55149]
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            # print(idx)

            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            file_name = 1
            rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                             join_table, file_name, query_encode[3], model, test=True)
            # rl.random_num = dqn.epsilon_start
            rl.random_num = 0  # 测试的时候用
            rl.init_state()
            while rl.state:
                rl.choose_action()
            terminate_plan = get_hint_SQL_explain(sql,
                                                  '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                                  str(11),
                                                  conn.conn)
            all.append(float(terminate_plan['Execution Time']))
        reward_log.append(sum(all) / len(TEST_FILE))
        with open(path, 'w') as file_obj:
            json.dump(reward_log, file_obj)

    # USEABLE_FILE = []
    # for file in sorted(os.listdir('./train_plan/index/')):
    #     with open('./train_plan/index/' + file) as f:
    #         for idx in f.readlines():
    #             USEABLE_FILE.append(int(idx))
    # USEABLE_FILE.sort()
    USEABLE_FILE = [55149]
    local_device = torch.device('cuda:' + str(3))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()
    batch_loss = []
    for epoch in range(0, 10000):
        print('Current epoch:', epoch)
        all_loss = []
        current_loss = []
        # USEABLE_FILE = [7591]
        # TEST_FILE = []
        for idx in USEABLE_FILE:
            # if idx not in TEST_FILE:
            if True:
                sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
                sql = sqlparse.format(sql, reindent=True, keyword_case="upper")

                SQL_encode = travesal_SQL(sql)
                query_encode = SQL_encode
                join_table = get_join_table(query_encode[2])
                join_table = reverse_kv(query_encode[3], join_table)
                query_encode = replace_alias(query_encode)

                join_table = remove_same_table(join_table, 7)
                now_tables = get_tables(query_encode)
                query_vector = get_vector(query_encode)

                file_name = 1
                rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                                 join_table, file_name, query_encode[3], model, test=False, random_num_=0)
                # rl.random_num = dqn.epsilon_start
                rl.random_num = 0  # 测试的时候用
                rl.init_state()
                while rl.state:
                    rl.choose_action()
                terminate_plan = get_hint_SQL_explain(sql,
                                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                                      str(11),
                                                      conn.conn)
                with open('kkk.json', 'w') as f:
                    json.dump(terminate_plan, f)

                # print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)')
                # print(sql)
                result_now = _traversal_plan_tree_latency(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])
                # print(result_now[0])
                NORMALIZATION = 1
                if isinstance(result_now[0][0], tuple):
                    rewards = [round(pow(result_now[0][0][1], 1 / NORMALIZATION), 6)]
                else:
                    rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                for i in range(1, len(result_now[0])):
                    if (not isinstance(result_now[0][i], tuple) and (not isinstance(result_now[0][i - 1], tuple))):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                    elif (not isinstance(result_now[0][i], tuple) and (isinstance(result_now[0][i - 1], tuple))):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1][0], 1 / NORMALIZATION), 6))
                    else:
                        rewards.append(round(pow(result_now[0][i][1], 1 / NORMALIZATION), 6))
                result_now[0] = rewards
                batch_size = 1
                batch_loss = finish_episode(model, rl.saved_log_probs, rewards, optimizer, batch_loss, batch_size)
                if len(batch_loss) >= batch_size:
                    batch_loss = []

                if len(current_loss) > 0 and len(current_loss) % 100 == 0:
                    print('loss:', sum(current_loss) / len(current_loss))
                    all_loss.append(sum(current_loss) / len(current_loss))
                    current_loss = []

        # print('Average Loss:', sum(all_loss)/ len(all_loss))
        if epoch % 100 == 0:
            set_statement_timeout(EXECUTION_MAX_TIME)
            load_extension()
            test(model, epoch)


def train_intialization_discrim_model(x, x_valid):
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    DEVICE = 'CPU'
    use_gpu = False
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:' + str(2))
        use_gpu = True
    path = './Models/Model_Policy_Transfer/Initialization/Initialization.pth'

    model = Discrim_net()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path)['model'])
        print('Load checkpoint!!')

    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0008)
    criterion = nn.MSELoss()
    # model.load_state_dict(cp['model'])

    model.train()

    epoch_loss = 0
    min_valid_loss = 1e5
    length = 512
    i = 0
    for epoch in range(1000):
        count = 0
        i = 0
        while i < len(x):
            x_ = x[idx[i: i + length]]
            optim.zero_grad()
            output = model(x_)
            loss = criterion(output, torch.ones_like(output))
            loss.backward()
            optim.step()
            # if (use_gpu):
            #     loss = loss.cpu()
            #     print('Average Loss:', loss)
            count += 1
            epoch_loss += loss.cpu().item()
            i += length
        print('Epoch {}, Average Loss:{}'.format(epoch, epoch_loss / count))

        # validation
        output_valid = model(x_valid)
        valid_loss = criterion(output_valid, torch.ones_like((output_valid)))
        print('Epoch {}, Average Validation Loss:{}'.format(epoch, valid_loss))
        print('-' * 50)
        if valid_loss < min_valid_loss:
            state = {'model': model.state_dict()}
            torch.save(state, path)
            min_valid_loss = valid_loss


def Inverse_RL_one_sql(policy_net, discrim_net):

    def _execute(sql):
        SQL_encode = travesal_SQL(sql)
        query_encode = SQL_encode
        join_table = get_join_table(query_encode[2])
        join_table = reverse_kv(query_encode[3], join_table)
        query_encode = replace_alias(query_encode)

        join_table = remove_same_table(join_table, 7)
        now_tables = get_tables(query_encode)
        query_vector = get_vector(query_encode)

        file_name = 1
        if k == 0:
            random_num_ = 0
        else:
            random_num_ = 1 / 3
        rl = DNNRLGOO_Pi(policy_net, None, now_tables[1], query_vector, query_encode[4],
                         join_table, file_name, query_encode[3], policy_net,
                         test=False,
                         random_num_=0)
        # rl.random_num = dqn.epsilon_start
        rl.random_num = 0  # 测试的时候用
        rl.init_state()
        while rl.state:
            rl.choose_action()

        return rl

    def _execute_plan(rl):
        terminate_plan = get_hint_SQL_explain(sql,
                                              '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                              str(11),
                                              conn.conn)
        print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ')
        if terminate_plan == -1:
            terminate_plan_new = get_hint_SQL_explain(sql,
                                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                      str(11),
                                                      conn.conn)

            return [rl.query_encode, rl.state_list[-1]], terminate_plan_new['Plan']['Total Cost'], 90000, None

        buffer = []

        result_now_latency = _traversal_plan_tree_latency(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])
        result_now_cost = _traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])

        for idx, num in enumerate(result_now_cost[0]):
            if isinstance(num, tuple):
                result_now_cost[0][idx] = result_now_cost[0][idx - 1] + num[1]
        for now in result_now_cost[0]:
            assert isinstance(now, float)

        try:
            assert len(rl.state_list) == len(result_now_cost[0]) == len(result_now_latency[0])
        except:
            print('iter: {}, Problem Plan!!!!!!!'.format(iter))

            with open(str(idx).split('.')[0] + '_prob.json', 'w') as f:
                json.dump(terminate_plan, f)
            return None, None, None
        for i in range(len(result_now_cost[0])):
            buffer.append(([rl.query_encode, rl.state_list[i]], result_now_cost[0][i], result_now_latency[0][i]))

        return [rl.query_encode, rl.state_list[-1]], terminate_plan['Plan']['Total Cost'], terminate_plan[
            'Execution Time'], buffer

    def execute(sql, policy_net, k):
        SQL_encode = travesal_SQL(sql)
        query_encode = SQL_encode
        join_table = get_join_table(query_encode[2])
        join_table = reverse_kv(query_encode[3], join_table)
        query_encode = replace_alias(query_encode)

        join_table = remove_same_table(join_table, 7)
        now_tables = get_tables(query_encode)
        query_vector = get_vector(query_encode)

        file_name = 1
        if k == 0:
            random_num_ = 0
        else:
            random_num_ = 1 / 3
        rl = DNNRLGOO_Pi(policy_net, None, now_tables[1], query_vector, query_encode[4],
                         join_table, file_name, query_encode[3], policy_net,
                         test=False,
                         random_num_=random_num_)
        # rl.random_num = dqn.epsilon_start
        rl.random_num = 0  # 测试的时候用
        rl.init_state()
        while rl.state:
            rl.choose_action()
        terminate_plan = get_hint_SQL_explain(sql,
                                              '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                              str(11),
                                              conn.conn)
        print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ')
        if terminate_plan == -1:
            terminate_plan_new = get_hint_SQL_explain(sql,
                                                  '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                  str(11),
                                                  conn.conn)

            return [rl.query_encode, rl.state_list[-1]], terminate_plan_new['Plan']['Total Cost'], 90000, None

        buffer = []

        result_now_latency = _traversal_plan_tree_latency(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])
        result_now_cost = _traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])

        for idx, num in enumerate(result_now_cost[0]):
            if isinstance(num, tuple):
                result_now_cost[0][idx] = result_now_cost[0][idx - 1] + num[1]
        for now in result_now_cost[0]:
            assert isinstance(now, float)

        try:
            assert len(rl.state_list) == len(result_now_cost[0]) == len(result_now_latency[0])
        except:
            print('iter: {}, Problem Plan!!!!!!!'.format(iter))

            with open(str(idx).split('.')[0] + '_prob.json', 'w') as f:
                json.dump(terminate_plan, f)
            return None, None, None
        for i in range(len(result_now_cost[0])):
            buffer.append(([rl.query_encode, rl.state_list[i]], result_now_cost[0][i], result_now_latency[0][i]))

        return [rl.query_encode, rl.state_list[-1]], terminate_plan['Plan']['Total Cost'], terminate_plan['Execution Time'], buffer

    def train_discrim(discrim_net, cost_memory, latency_memory, discrim_optim, learn_step_counter, update_num, iter):
        def cal_corr(cost, latency):
            # cost = torch.tensor(cost)
            cost = torch.cat(cost, dim=0).reshape(-1)
            latency = torch.tensor(latency)
            v_cost = cost - torch.mean(cost)
            v_latency = latency - torch.mean(latency)
            corr = torch.sum(v_cost * v_latency) / (torch.sqrt(torch.sum(v_cost ** 2) * torch.sum(v_latency ** 2)) + 1e-7)
            return corr

        def cal_penalty(discrim_net, conflict, lamba):
            # nn.ReLU()
            Penalty_1 = 0
            Penalty_2 = 0
            delta = 0
            # lamba = 0.1
            ratio = iter // 5
            p = 2
            miu = 10**(-3) * (p**ratio)
            batch_size = 256
            if len(conflict.buffer) == 0:
                print('No Conflict!')
                return 0

            batch, importance, indices = conflict.sample(batch_size, (learn_step_counter % 2 == 0) * 0.7)


            # assert len(lamba) == len(conflict)
            error = []
            Penalty_1 = []
            for one in batch:
                small, large = one
                s_state, s_cost = small
                l_state, l_cost = large
                cur = F.relu(-((discrim_net(s_state)+s_cost) - (discrim_net(l_state)+l_cost)) + delta)
                Penalty_1.append(cur**2)
                error.append(float(cur.cpu()))


                # 在transfer_test_13里面没有加上Pentalty_2
                # Penalty_2 += lamba[cur_idx] * cur
                # lamba[cur_idx] += miu * float(cur)
            conflict.set_priorities(indices, error)
            Penalty_1 = torch.tensor(Penalty_1)
            importance = torch.tensor(importance).unsqueeze(1)
            Penalty_1 = torch.mean(importance*Penalty_1)



            # Penalty_2 = Penalty_2 / len(train_penalty)
            # return Penalty_1 + Penalty_2
            return Penalty_1

        # Criterion = nn.LeakyReLU()
        # Criterion = nn.MSELoss()
        Criterion = nn.L1Loss()
        batch_size = 128
        new_exp_batch = 128
        all_loss = 0

        cost_memory = np.array(cost_memory)
        latency_memory = np.array((latency_memory))

        discrim_optim.zero_grad()
        if len(cost_memory) > 0:
            for _ in range(update_num):
                discrim_loss = 0
                # update 增加修改最新的experience以及conflict
                if len(cost_memory) > 2*new_exp_batch:
                    idx = np.arange(0, len(cost_memory) - new_exp_batch)
                    new_exp_idx = np.arange(len(cost_memory) - new_exp_batch, len(cost_memory))
                else:
                    idx = np.arange(0, len(cost_memory))
                    new_exp_idx = np.array([], dtype=np.int64)
                np.random.shuffle(idx)
                # todo:可以改成優先經驗回放池，優先回放矛盾的memory
                if len(cost_memory) >= batch_size:
                    sample_batch = idx[:batch_size]
                else:
                    sample_batch = idx
                sample_batch = np.concatenate([sample_batch, new_exp_idx])
                np.random.shuffle(sample_batch)

                train_cost_batch = cost_memory[sample_batch]
                train_late_batch = latency_memory[sample_batch]
                print('Training batch length:', len(train_cost_batch))
                fixed_cost = []
                for cost, state in train_cost_batch:
                    fixed_cost.append(discrim_net(state) + cost)

                corr = cal_corr(fixed_cost, train_late_batch)
                penalty = cal_penalty(discrim_net, Conflict, lamda)

                loss = Criterion(corr, torch.tensor(1)) + penalty

                #loss = Criterion(corr, torch.tensor(1)) + penalty
                learn_step_counter += 1
                print(loss)
                discrim_loss = loss
                all_loss += discrim_loss
                discrim_loss.backward()
                nn.utils.clip_grad_norm_(discrim_net.parameters(), max_norm=2)
                discrim_optim.step()
            print('Average Loss:', all_loss / update_num)
            checkpoint_path = './Models/Model_Policy_InverseRL/Active_24/discrim_now_{}.pth'.format(iter)
            state = {'model': discrim_net.state_dict()}
            torch.save(state, checkpoint_path)
        else:
            print('Does not find any conflict!!')

    def train_policy(sql, policy_net, policy_optim, discrim_net, policy_update_num):

        for step in range(policy_update_num):
            batch_loss = []
            for idx in TEST_JOB:
                sql = loadSQL("./test_sql/job/" + idx)
                SQL_encode = travesal_SQL(sql)

                query_encode = SQL_encode
                join_table = get_join_table(query_encode[2])
                join_table = reverse_kv(query_encode[3], join_table)
                query_encode = replace_alias(query_encode)

                join_table = remove_same_table(join_table, 7)
                now_tables = get_tables(query_encode)
                query_vector = get_vector(query_encode)

                file_name = 1
                rl = DNNRLGOO_Pi(policy_net, None, now_tables[1], query_vector, query_encode[4],
                                 join_table, file_name, query_encode[3], policy_net, test=False, random_num_=0)
                rl.init_state()
                while rl.state:
                    rl.choose_action()
                terminate_plan = get_hint_SQL_explain(sql,
                                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                      str(11),
                                                      conn.conn)

                # print('/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)')
                # print(sql)
                result_now = _traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector, rl.first_relation[0])
                # print(result_now[0])

                assert len(rl.state_list) == len(result_now[0])
                tune_rate = [float(discrim_net([rl.query_encode, state]).detach().cpu().numpy()) for state in
                             rl.state_list]
                print('change:', tune_rate)
                new_cost = [(a + np.log(b)) if (not isinstance(b, tuple)) else (a + np.log(b[0]), a + np.log(b[1])) for
                            a, b in zip(tune_rate, result_now[0])]
                result_now[0] = new_cost

                NORMALIZATION = 1
                if isinstance(result_now[0][0], tuple):
                    rewards = [round(pow(result_now[0][0][1], 1 / NORMALIZATION), 6)]
                else:
                    rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                for i in range(1, len(result_now[0])):
                    if (not isinstance(result_now[0][i], tuple) and (not isinstance(result_now[0][i - 1], tuple))):
                        try:
                            rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                        except:
                            with open(str(idx).split('.')[0] + '_prob.json', 'w') as f:
                                json.dump(terminate_plan, f)
                            print(
                                '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ')
                            print(result_now[0])
                            print(result_now[0][i], result_now[0][i - 1])
                            assert 0
                    elif (not isinstance(result_now[0][i], tuple) and (isinstance(result_now[0][i - 1], tuple))):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1][0], 1 / NORMALIZATION), 6))
                    else:
                        rewards.append(round(pow(result_now[0][i][1], 1 / NORMALIZATION), 6))

                rewards = [-i for i in rewards]

                for i in rewards:
                    if np.isnan(i):
                        rewards = []
                        break
                print('rewards:', rewards)
                assert len(rewards) == len(tune_rate)
                # for i in range(len(rewards)):
                #     rewards[i] = rewards[i] + (0.98 ** iter) * (1 - tune_rate[i])
                result_now[0] = rewards

                # batch_size = 1
                batch_loss = finish_episode(policy_net, rl.saved_log_probs, rewards, policy_optim, batch_loss, idx)
            # if len(batch_loss) >= batch_size:
            #    batch_loss = []

    def finish_episode(model, saved_log_probs, rewards, optimizer, batch_loss, template):
        R = 0
        policy_loss = []
        returns = []
        gamma = 0.98
        # eps = np.finfo(np.float32).eps.item()
        # rewards = [-i for i in rewards]
        # todo：add curiosity

        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=local_device)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        if int(template[:-5]) < int(TEST_JOB[-1][:-5]):
            batch_loss.extend(policy_loss)
            return batch_loss
        batch_loss.extend(policy_loss)
        # assert len(batch_loss) == 11
        print('batch loss:', len(batch_loss))
        batch_loss_ = torch.cat(batch_loss)

        # batch_loss_ = (batch_loss_ - batch_loss_.mean()) / (batch_loss_.std() + eps)
        optimizer.zero_grad()
        # policy_loss = torch.cat(batch_loss).sum()
        policy_loss = batch_loss_.sum()
        # print('log likelyhodd:', saved_log_probs)
        print('loss:', policy_loss)
        # current_loss.append(float(policy_loss))
        policy_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        return batch_loss

    def test(model, epoch):
        # save CheckPoint
        checkpoint_path = './Models/Model_Policy_InverseRL/Active_24/now_{}.pth'.format(epoch)
        state = {'model': model.state_dict()}
        torch.save(state, checkpoint_path)

        path = 'reward_InverseRL_Curiosity_new_Active_24.json'
        if os.path.exists(path):
            reward_log = json.load(open(path, encoding='utf-8'))
        else:
            reward_log = []
        all = []
        TEST_FILE = TEST_JOB
        for idx in TEST_FILE:
            sql = loadSQL("./test_sql/job/" + idx)

            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            file_name = 1
            rl = DNNRLGOO_Pi(model, None, now_tables[1], query_vector, query_encode[4],
                             join_table, file_name, query_encode[3], model, test=True)
            # rl.random_num = dqn.epsilon_start
            rl.random_num = 0  # 测试的时候用
            rl.init_state()
            while rl.state:
                rl.choose_action()
            terminate_plan = get_hint_SQL_explain(sql,
                                                  '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (ANALYZE,Buffers,TIMING,COSTS,VERBOSE,format json)',
                                                  str(11),
                                                  conn.conn)
            if terminate_plan == -1:
                all.append(90000)
            else:
                all.append(float(terminate_plan['Execution Time']))
        all = np.array(all)
        cur = pow(np.prod(all / PG_late), 1 / len(all))
        print(cur)
        reward_log.append((sum(all) / len(TEST_FILE), cur))

        with open(path, 'w') as file_obj:
            json.dump(reward_log, file_obj)

        # save CheckPoint
        # checkpoint_path = './Models/Model_Policy_Transfer/Stats/now_{}.pth'.format(epoch)
        # state = {'model': model.state_dict(), 'result': sum(all) / len(TEST_FILE)}
        # torch.save(state, checkpoint_path)



    policy_optim = torch.optim.AdamW(params=policy_net.parameters(), lr=0.0005, weight_decay=0.002)
    discrim_optim = torch.optim.AdamW(params=discrim_net.parameters(), lr=0.0008, weight_decay=0.02)

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()

    max_episode = 2000
    batch = 100
    k = 15
    discrim_update_num = 100
    policy_update_num = 20
    train_disrim_flag = True
    total_sample_size = 10

    cost_memory = []
    latency_memory = []
    Trajectories = []
    Conflict = PrioritizedReplayBuffer(20000)
    lamda = []
    count = 0
    learn_step_counter = 0
    for iter in range(1, max_episode):
        print('Current iteration', iter)
        # Active Sampling

        # load the latest encode dict in policy net
        _query_encode.load_state_dict(policy_net.state_dict(), strict=False)
        encode_sql = []
        weights = []
        chosen = []
        policy_net.eval()
        for idx in TEST_JOB:
            sql = loadSQL("./test_sql/job/" + idx)
            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)
            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)
            encode_sql.append(query_vector)
            rl = DNNRLGOO_Pi(policy_net, None, now_tables[1], query_vector, query_encode[4],
                             join_table, 1, query_encode[3], policy_net,
                             test=False,
                             random_num_=0)
            # rl.random_num = dqn.epsilon_start
            rl.random_num = 0  # 测试的时候用
            rl.init_state()
            while rl.state:
                rl.choose_action()
            weights.append(sum(rl.entropy))

        # calculate latent features
        encode_sql = _query_encode(encode_sql)
        encode_sql = encode_sql.cpu().detach().numpy()
        # clustering = DBSCAN(eps=0.12, min_samples=2).fit(encode_sql.cpu().detach().numpy())
        kmeans = KMeans(n_clusters=15, random_state=0, max_iter=1000).fit(encode_sql, sample_weight=weights)

        for i in range(k):
            center = kmeans.cluster_centers_[i]
            lst = np.where(kmeans.labels_ == i)[0].tolist()
            distance = [np.sum((center-encode_sql[idx])**2) for idx in lst]
            chosen.append(lst[np.argmin(distance)])
        chosen = [TEST_JOB[i] for i in chosen]

        # IMPORTANT!!
        policy_net.train()
        for idx in chosen:
            sql = loadSQL("./test_sql/job/" + idx)
            policy_net.train()
            plans = []
            entropys = []
            for _ in range(1, total_sample_size + 1):
                rl = _execute(sql)
                plans.append(rl)
                entropys.append(sum(rl.entropy))
            idx_entro = np.argmax(np.array(entropys))
            final_state, cost_, latency_, buffer = _execute_plan(plans[idx_entro])

                # final_state, cost_, latency_, buffer = execute(sql, policy_net, rand)

            for s, c, l in Trajectories:
                if cost_ < c and latency_ > l:
                    # append small cost to left and large cost to right
                    Conflict.add(((final_state, np.log(cost_)), (s, np.log(c))))
                    # lamda.append(0)
                elif cost_ > c and latency_ < l:
                    Conflict.add(((s, np.log(c)), (final_state, np.log(cost_))))
                    # lamda.append(0)

                elif cost_ < c:
                    Conflict.add(((final_state, np.log(cost_)), (s, np.log(c))))
                elif cost_ > c:
                    Conflict.add(((s, np.log(c)), (final_state, np.log(cost_))))

            Trajectories.append((final_state, cost_, latency_))

            if buffer == None:
                continue

            for state, c, l in buffer:
                cost_memory.append((np.log(c), state))
                latency_memory.append(np.log(l))

                # if latency_ < latency_ex and cost_ > cost_ex:
                #    memory.append((cost_, latency_, T_))

            # memory = json.load(open('/data0/chenx/imdb_QO/Models/Model_Policy_InverseRL/Memory/memory_0.json', encoding='utf-8'))
        # print('len of memory:', len(latency_memory))
        # print('len of conflict:', len(Conflict))

        if train_disrim_flag:
            train_discrim(discrim_net, cost_memory, latency_memory, discrim_optim, learn_step_counter, update_num=discrim_update_num,
                          iter=iter)


        policy_net.eval()
        train_policy(sql, policy_net, policy_optim, discrim_net, policy_update_num=policy_update_num)

        if iter % 5 == 0:
            test(policy_net, iter)






def save_train_data_():
    def to_one_hot(list):
        choice2vector = {(0, None): 0, (1, None): 1, (0, 0): 2, (1, 0): 3, (0, 1): 4, (1, 1): 5, (0, 2): 6, (1, 2): 7}
        result = []
        for i in list:
            relation_name, index, join_type = i
            one = choice2vector[(index, join_type)]
            current = [0] * (21 * len(choice2vector))
            current[all_tables[relation_name] * len(choice2vector) + one] = 1
            result.append(current)
        return result

    if not os.path.exists('./train_plan/Imitation_Learning/x_expert_train.npy'):
        x_train = []
        x_train_table = []
        x_train_start = []
        y_train = []
        x_test = []
        x_test_table = []
        x_test_start = []
        y_test = []
        problem_plan = []
        USEABLE_FILE = []
        for file in sorted(os.listdir('./train_plan/index/')):
            with open('./train_plan/index/' + file) as f:
                for idx in f.readlines():
                    USEABLE_FILE.append(int(idx))
        USEABLE_FILE.sort()
        # problem = np.load('./train_plan/Neo_baseline/problem_plan.npy').tolist()
        # USEABLE_FILE = [i for i in USEABLE_FILE if i not in problem]
        for num, idx in enumerate(USEABLE_FILE):
            print(idx)
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode
            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 7)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            terminate_plan = get_hint_SQL_explain(sql,
                                                  'EXPLAIN (format json)',
                                                  str(11),
                                                  conn.conn)
            try:
                result_now = _traversal_plan_tree_cost(terminate_plan['Plan'], f, query_vector, None)
            except:
                print('problem plan: {}'.format(idx))
                problem_plan.append(idx)
                continue
            current = []
            order = [one[0] for one in result_now[3]]
            for i in range(len(order)):
                current.append(order[i:])
            assert len(result_now[3]) + 1 == len(result_now[2]) == len(current) + 1
            if idx not in TEST_FILE:
                x_train.extend(result_now[2][:-1])
                x_train_table.extend(current)
                x_train_start.extend([1] + [0] * (len(current) - 1))
                y_train.extend(to_one_hot(result_now[3]))
                # y_train.extend([round(pow(terminate_plan['Plan']['Total Cost'], 1 / NORMALIZATION), 6)] * len(result_now[2]))
            else:
                x_test.extend(result_now[2][:-1])
                x_test_table.extend(current)
                x_test_start.extend([1] + [0] * (len(current) - 1))
                y_test.extend(to_one_hot(result_now[3]))
                # y_test.extend([round(pow(terminate_plan['Plan']['Total Cost'], 1 / NORMALIZATION), 6)] * len(result_now[2]))
        assert len(x_train) == len(y_train) == len(x_train_table) == len(x_train_start)
        assert len(x_test) == len(y_test) == len(x_test_table) == len(x_test_start)
        x_train = np.array(x_train)
        x_train_start = np.array(x_train_start)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        x_test_start = np.array(x_test_start)
        y_test = np.array(y_test)
        np.save('./train_plan/Imitation_Learning/problem_plan.npy', np.array(problem_plan))
        np.save('./train_plan/Imitation_Learning/x_expert_train.npy', x_train)
        np.save('./train_plan/Imitation_Learning/x_expert_train_start.npy', x_train_start)
        np.save('./train_plan/Imitation_Learning/y_expert_train.npy', y_train)
        np.save('./train_plan/Imitation_Learning/x_expert_test.npy', x_test)
        np.save('./train_plan/Imitation_Learning/x_expert_test_start.npy', x_test_start)
        np.save('./train_plan/Imitation_Learning/y_expert_test.npy', y_test)
        with open('./train_plan/Imitation_Learning/x_expert_train_table.json', 'w') as f:
            json.dump(x_train_table, f)
        with open('./train_plan/Imitation_Learning/x_expert_test_table.json', 'w') as f:
            json.dump(x_test_table, f)

        print('There are {} training expert experiences, all of them are saved!'.format(len(x_train)))
        print('There are {} testing expert experiences, all of them are saved!'.format(len(x_test)))
    else:
        x_train = np.load('./train_plan/Imitation_Learning/x_expert_train.npy', allow_pickle=True)
        x_train_table = json.load(open('./train_plan/Imitation_Learning/x_expert_train_table.json'), encoding='utf-8')
        x_train_start = np.load('./train_plan/Imitation_Learning/x_expert_train_start.npy')
        y_train = np.load('./train_plan/Imitation_Learning/y_expert_train.npy', allow_pickle=True)
        x_test = np.load('./train_plan/Imitation_Learning/x_expert_test.npy', allow_pickle=True)
        x_test_table = json.load(open('./train_plan/Imitation_Learning/x_expert_test_table.json'), encoding='utf-8')
        x_test_start = np.load('./train_plan/Imitation_Learning/x_expert_test_start.npy')
        y_test = np.load('./train_plan/Imitation_Learning/y_expert_test.npy', allow_pickle=True)

    return x_train, x_train_table, x_train_start, y_train, x_test, x_test_table, x_test_start, y_test




if __name__ == '__main__':
    # x_train, x_train_table, x_train_start, y_train, x_test, x_test_table, x_test_start, y_test = save_train_data_()

    new_cp = json.load(open('reward_log_policy_JOB.json', encoding='utf-8'))
    # chekpoint    
    # min_idx = new_cp.index(18679.564599999998)
    dir_list = os.listdir('./Models/Model_Policy_CP/Job/')
    new_list = [int(i[4:].split('.')[0]) for i in dir_list]
    new_list.sort()
    checkpoint_path = './Models/Model_Policy_CP/Job/now_{}.pth'.format(new_list[min_idx])
    cp = torch.load(checkpoint_path, map_location='cpu')

    policy_net = Pi_net()
    local_device = torch.device('cuda:' + str(0))
    policy_net.to(local_device)
    policy_net.load_state_dict(cp['model'])



    discrim_net = Discrim_net()

    discrim_net.to(local_device)

    _query_encode = _QueryEncoder()
    _query_encode.to(local_device)
    _query_encode.load_state_dict(cp['model'], strict=False)

    print('Load checkpoint!!')
    # 调整reward function 为 F = phi(s', a') - phi(s, a)
    Inverse_RL_one_sql(policy_net, discrim_net)


