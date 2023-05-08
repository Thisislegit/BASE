import os
from torch import nn
import numpy as np
from utils.Connection import all_tables
from utils.PlanTree import OPERATOR_TYPE
import torch
import random
import itertools
from TreeConvolution.util_1 import prepare_trees
from TreeConvolution.tcnn import left_child, right_child


# DNN guided RL Model
class DNNRLGOO(object):
    def __init__(self, model, optimizer, table, query_encode, selectivity, join_table, file_name, table_alias, target_model, test=False, distributed=False):
        self.target_net = target_model
        self.value_net = model
        self.optimizer = optimizer
        self.compute_loss = nn.MSELoss()
        self.tables = table
        self.collection = ()
        self.query_encode = query_encode
        self.plan_encode = []
        self.join_type = ["Hash Join", "Merge Join", "Nested Loop"]
        self.state = True
        # self.latency = 0
        self.sql = ''
        self.hint_index = []
        self.hint_join = []
        self.hint_leading = []
        self.now_stack = []
        self.join_table = join_table
        self.total_vector = [0] * (len(all_tables) * 2 + len(self.join_type))
        # self.latency = 0
        self.loss = 0
        self.count = 0
        self.random_num = 0.5
        self.selectivity = selectivity
        self.used_table = []
        # self.reward = []
        self.state_list = []
        self.file_name = file_name
        self.table_alias = table_alias
        self.Q_values = []
        self.test = test
        self.distributed = distributed

    def add_hint_index(self, table):
        self.hint_index.append('IndexScan( '+table+' )')

    def add_hint_join(self, table, join_type):
        self.hint_leading.append(table)
        type_dic = ['HashJoin', 'MergeJoin', 'NestLoop']
        self.hint_join.append(type_dic[join_type] + '( ' + " ".join(self.hint_leading) + ' )')

    def init_state(self):
        def transformer(x, encode, local_device):
            b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
            if encode:
                return torch.cat((encode, b))
            else:
                return b

        ret = random.random()
        if ret < self.random_num:
            index_0 = random.randint(0, 1)
            index_1 = random.randint(0, 1)
            join_type_index = random.randint(0, len(self.join_type)-1)
            join_index = random.randint(0, len(self.join_table)-1)
            if self.join_table[join_index][0] in self.table_alias:
                left_table = self.table_alias[self.join_table[join_index][0]]
            else:
                left_table = self.join_table[join_index][0]
            if self.join_table[join_index][1] in self.table_alias:
                right_table = self.table_alias[self.join_table[join_index][1]]
            else:
                right_table = self.join_table[join_index][1]
            vector_0 = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_1 = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_0[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
            vector_0[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0

            vector_1[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            vector_1[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1

            vector_parent[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
            vector_parent[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]

            vector_parent[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0
            vector_parent[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1
            self.total_vector = vector_parent.copy()
            vector_parent[join_type_index] = 1
            # min_cost = self.value_net([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
            self.state_list.append((tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),)))
            # self.reward.append(float(min_cost))
            self.collection = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
            if index_0 == 1:
                self.add_hint_index(self.join_table[join_index][0])
            if index_1 == 1:
                self.add_hint_index(self.join_table[join_index][1])
            self.hint_leading.append(self.join_table[join_index][0])
            self.add_hint_join(self.join_table[join_index][1], join_type_index)
            self.used_table += self.join_table[join_index]
            self.join_table.pop(join_index)
        else:
            # min_cost = float("inf")
            # join_index = 0
            i_index = 0
            all_actions = []
            join_indexes = []
            left_index_states = []
            right_index_states = []
            join_type_indexes = []
            for i in self.join_table:
                a, b = i
                for i in [(a, b), (b, a)]:
                    for join in range(3):
                        for index_0 in range(2):
                            for index_1 in range(2):
                                if i[0] in self.table_alias:
                                    left_table = self.table_alias[i[0]]
                                else:
                                    left_table = i[0]
                                if i[1] in self.table_alias:
                                    right_table = self.table_alias[i[1]]
                                else:
                                    right_table = i[1]
                                vector_0 = [0] * (len(all_tables) * 2 + len(self.join_type))
                                vector_1 = [0] * (len(all_tables) * 2 + len(self.join_type))
                                vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))

                                vector_0[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
                                vector_0[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0

                                vector_1[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                                vector_1[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1

                                vector_parent[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
                                vector_parent[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                                vector_parent[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0
                                vector_parent[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1
                                vector_parent[join] = 1

                                # now_value = self.value_net([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                                # if now_value < min_cost:
                                #     node = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
                                #     min_cost = now_value
                                #     join_index = i_index
                                #     self.total_vector = vector_parent.copy()
                                #     # self.total_vector[join] = 0
                                #     left_index_state = index_0
                                #     right_index_state = index_1
                                #     join_type_index = join

                                # become batch training
                                all_actions.append([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                                join_indexes.append(i_index)
                                left_index_states.append(index_0)
                                right_index_states.append(index_1)
                                join_type_indexes.append(join)
                i_index += 1

            # preprocessing!
            if self.distributed:
                encode = []
                trees = []
                local_device = torch.device('cuda:' + str(0))
                for i in all_actions:
                    encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                    trees.append(i[1])
                x = torch.cat(encode, dim=0)
                now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=local_device)
                now_actions = [x, list(now_trees)]
            else:
                now_actions = all_actions

            if not self.test:
                now_value = self.value_net(now_actions)
            else:
                now_value = self.target_net(now_actions)
            min_idx = torch.argmin(now_value, dim=0).item()
            join_index = join_indexes[min_idx]
            node = all_actions[min_idx][1]
            min_cost = now_value[min_idx].item()
            left_index_state = left_index_states[min_idx]
            right_index_state = right_index_states[min_idx]
            join_type_index = join_type_indexes[min_idx]
            self.total_vector = list(node[0])

            self.collection = node
            self.state_list.append(node)
            # self.reward.append(float(min_cost))
            if left_index_state == 1:
                self.add_hint_index(self.join_table[join_index][0])
            if right_index_state == 1:
                self.add_hint_index(self.join_table[join_index][1])
            self.hint_leading.append(self.join_table[join_index][0])
            self.add_hint_join(self.join_table[join_index][1], join_type_index)
            # used table 中也要用别名，对于1个表两个alias，也要当成不同的看待
            self.used_table += self.join_table[join_index]
            self.join_table.pop(join_index)
        self.check_state()


        if not self.test:
            if self.state:
                all_actions = self.getMinQ()
                # print(self.collection)
                self.Q_values.append(([self.query_encode, self.collection], all_actions))
            else:
                self.Q_values.append(([self.query_encode, self.collection], []))

    def choose_action(self):
        def transformer(x, encode, local_device):
            b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
            if encode:
                return torch.cat((encode, b))
            else:
                return b

        operator_num = len(self.join_type)
        ret = random.random()
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
        if ret < self.random_num:
            # vector = [0] * (len(all_tables) * 2 + len(self.join_type))
            random_table = random.randint(0, len(now_need_join_table) - 1)
            random_index = random.randint(0, 1)
            random_join = random.randint(0, 2)

            if now_need_join_table[random_table][0]  in self.table_alias:
                right_table = self.table_alias[now_need_join_table[random_table][0]]
            else:
                right_table = now_need_join_table[random_table][0]
            vector_right = [0] * (len(all_tables) * 2 + operator_num)
            vector_right[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            # vector_right[operator_num + all_tables[right_table] * 2] = 1

            vector_right[operator_num + 1 + all_tables[right_table] * 2] = random_index
            tuple_right = (tuple(vector_right),)
            vector_parent = self.total_vector.copy()
            # 前三项为0！
            vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
            vector_parent[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            # vector_parent[operator_num + all_tables[right_table] * 2] = 1

            vector_parent[operator_num + 1 + all_tables[right_table] * 2] = random_index
            vector_parent[random_join] = 1
            # now_value = self.value_net([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
            # min_cost = now_value
            self.collection = (tuple(vector_parent), self.collection, tuple_right)
            self.state_list.append((tuple(vector_parent), self.collection, tuple_right))
            # self.latency = now_value
            # self.reward.append(float(now_value))
            if random_index == 1:
                self.add_hint_index(now_need_join_table[random_table][0])
            self.add_hint_join(now_need_join_table[random_table][0], random_join)
            # self.total_vector[operator_num + all_tables[right_table] * 2] = self.selectivity[
            #     right_table]
            # self.total_vector[len(self.join_type) + 1 + all_tables[self.tables[random_index]] * 2] = random_index
            self.total_vector = vector_parent.copy()
            self.used_table.append(now_need_join_table[random_table][0])
            self.join_table.pop(now_need_join_table[random_table][1])
        else:
            # node = []
            # min_cost = float("inf")
            # join_index = 0
            # join_type_index = 0
            # right_index_state = 0
            # final_parent_vector = None
            all_actions = []
            join_indexes = []
            right_index_states = []
            join_type_indexes = []

            for i in range(len(now_need_join_table)):
                for join in range(3):
                    for index in range(2):
                        if now_need_join_table[i][0] in self.table_alias:
                            right_table = self.table_alias[now_need_join_table[i][0]]
                        else:
                            right_table = now_need_join_table[i][0]
                        vector_right = [0] * (len(all_tables) * 2 + operator_num)
                        vector_right[operator_num + all_tables[right_table] * 2] = 1 #self.selectivity[right_table]
                        vector_right[operator_num + 1 + all_tables[right_table] * 2] = index
                        tuple_right = (tuple(vector_right),)
                        vector_parent = self.total_vector.copy()
                        vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
                        vector_parent[operator_num + all_tables[right_table] * 2] = 1 #self.selectivity[right_table]
                        vector_parent[
                            operator_num + 1 + all_tables[right_table] * 2] = index
                        vector_parent[join] = 1

                        # now_value = self.value_net(
                        #     [self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                        # print(now_value)
                        # if now_value < min_cost:
                        #     node = (tuple(vector_parent), self.collection, tuple_right)
                        #     min_cost = now_value
                        #     join_index = i
                        #     final_parent_vector = vector_parent
                        #     # self.total_vector[join] = 0
                        #     right_index_state = index
                        #     join_type_index = join

                        # pay attention to batch training
                        all_actions.append([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                        join_indexes.append(i)
                        right_index_states.append(index)
                        join_type_indexes.append(join)

            # now_values = self.value_net(all_actions)

            # preprocessing!
            if self.distributed:
                encode = []
                trees = []
                local_device = torch.device('cuda:' + str(0))
                for i in all_actions:
                    encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                    trees.append(i[1])
                x = torch.cat(encode, dim=0)
                now_trees = prepare_trees(trees, transformer, left_child, right_child, None,
                                          local_device=local_device)
                now_actions = [x, list(now_trees)]
            else:
                now_actions = all_actions

            if not self.test:
                now_values = self.value_net(now_actions)
            else:
                now_values = self.target_net(now_actions)
            min_idx = torch.argmin(now_values, dim=0).item()
            join_index = join_indexes[min_idx]
            node = all_actions[min_idx][1]
            min_cost = now_values[min_idx].item()
            right_index_state = right_index_states[min_idx]
            join_type_index = join_type_indexes[min_idx]
            self.total_vector = list(node[0])
            # print(now_values.shape)
            # print(node)
            # print(min_cost)
            # print(right_index_state)
            # print(join_type_index)
            # print(self.total_vector)
            # print(join_index)
            # print('*' * 50)


            # self.total_vector = final_parent_vector
            self.collection = node
            self.state_list.append(node)
            # self.reward.append(float(min_cost))
            if right_index_state == 1:
                self.add_hint_index(now_need_join_table[join_index][0])
            self.add_hint_join(now_need_join_table[join_index][0], join_type_index)
            self.used_table.append(now_need_join_table[join_index][0])
            self.join_table.pop(now_need_join_table[join_index][1])
        self.check_state()


        if not self.test:
            if self.state:
                all_actions = self.getMinQ()
                self.Q_values.append(([self.query_encode, self.collection], all_actions))
            else:
                self.Q_values.append(([self.query_encode, self.collection], []))

    def getMinQ(self):
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

        node = []
        min_cost = float("inf")
        all_actions = []
        for i in range(len(now_need_join_table)):
            for join in range(3):
                for index in range(2):
                    if now_need_join_table[i][0] in self.table_alias:
                        right_table = self.table_alias[now_need_join_table[i][0]]
                    else:
                        right_table = now_need_join_table[i][0]
                    vector_right = [0] * (len(all_tables) * 2 + operator_num)
                    vector_right[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                    vector_right[operator_num + 1 + all_tables[right_table] * 2] = index
                    tuple_right = (tuple(vector_right),)
                    vector_parent = self.total_vector.copy()
                    vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
                    vector_parent[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                    vector_parent[
                        operator_num + 1 + all_tables[right_table] * 2] = index
                    vector_parent[join] = 1
                    # different!

                    # now_value = self.target_net(
                    #     [self.query_encode, (tuple(vector_parent), self.collection, tuple_right)]).detach()
                    # # print(vector_parent)
                    # # print(now_value)
                    # if now_value < min_cost:
                    #     node = (tuple(vector_parent), self.collection, tuple_right)
                    #     min_cost = now_value
                    #     join_index = i
                    #     final_parent_vector = vector_parent
                    #     # self.total_vector[join] = 0
                    #     right_index_state = index
                    #     join_type_index = join

                    all_actions.append([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])

        return all_actions

    def check_state(self):
        # if self.file_name not in [7, 8, 18, 20, 21]:
        # for every relation
        i = 0
        while i < len(self.join_table):
            if self.join_table[i][0] in self.used_table and self.join_table[i][1] in self.used_table:
                self.join_table.pop(i)
            else:
                i += 1
        if len(self.join_table) == 0:
            self.state = False

    def get_hint_leading(self):
        if len(self.hint_leading) < 2:
            return ''
        else:
            return 'Leading('+" ".join(self.hint_leading) + ')'

    def get_hint_index(self):
        return " ".join(self.hint_index)

    def get_hint_join(self):
        return " ".join(self.hint_join)

class DNNRLGOO_latency(object):
    def __init__(self, model, optimizer, table, query_encode, selectivity, join_table, file_name, table_alias, target_model, test=False, distributed=False):
        self.target_net = target_model
        self.value_net = model
        self.optimizer = optimizer
        self.compute_loss = nn.MSELoss()
        self.tables = table
        self.collection = ()
        self.query_encode = query_encode
        self.plan_encode = []
        self.join_type = ["Hash Join", "Merge Join", "Nested Loop"]
        self.state = True
        # self.latency = 0
        self.sql = ''
        self.hint_index = []
        self.hint_join = []
        self.hint_leading = []
        self.now_stack = []
        self.join_table = join_table
        self.total_vector = [0] * (len(all_tables) * 2 + len(self.join_type))
        # self.latency = 0
        self.loss = 0
        self.count = 0
        self.random_num = 0.5
        self.selectivity = selectivity
        self.used_table = []
        # self.reward = []
        self.state_list = []
        self.file_name = file_name
        self.table_alias = table_alias
        self.Q_values = []
        self.test = test
        self.distributed = distributed

    def add_hint_index(self, table):
        self.hint_index.append('IndexScan( '+table+' )')

    def add_hint_join(self, table, join_type):
        self.hint_leading.append(table)
        type_dic = ['HashJoin', 'MergeJoin', 'NestLoop']
        self.hint_join.append(type_dic[join_type] + '( ' + " ".join(self.hint_leading) + ' )')

    def init_state(self):
        def transformer(x, encode, local_device):
            b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
            if encode:
                return torch.cat((encode, b))
            else:
                return b

        ret = random.random()
        if ret < self.random_num:
            index_0 = random.randint(0, 1)
            index_1 = random.randint(0, 1)
            join_type_index = random.randint(0, len(self.join_type)-1)
            join_index = random.randint(0, len(self.join_table)-1)
            if self.join_table[join_index][0] in self.table_alias:
                left_table = self.table_alias[self.join_table[join_index][0]]
            else:
                left_table = self.join_table[join_index][0]
            if self.join_table[join_index][1] in self.table_alias:
                right_table = self.table_alias[self.join_table[join_index][1]]
            else:
                right_table = self.join_table[join_index][1]
            vector_0 = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_1 = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))
            vector_0[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
            vector_0[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0

            vector_1[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            vector_1[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1

            vector_parent[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
            vector_parent[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]

            vector_parent[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0
            vector_parent[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1
            self.total_vector = vector_parent.copy()
            vector_parent[join_type_index] = 1
            # min_cost = self.value_net([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
            self.state_list.append((tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),)))
            # self.reward.append(float(min_cost))
            self.collection = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
            if index_0 == 1:
                self.add_hint_index(self.join_table[join_index][0])
            if index_1 == 1:
                self.add_hint_index(self.join_table[join_index][1])
            self.hint_leading.append(self.join_table[join_index][0])
            self.add_hint_join(self.join_table[join_index][1], join_type_index)
            self.used_table += self.join_table[join_index]
            self.join_table.pop(join_index)
        else:
            # min_cost = float("inf")
            # join_index = 0
            i_index = 0
            all_actions = []
            join_indexes = []
            left_index_states = []
            right_index_states = []
            join_type_indexes = []
            for i in self.join_table:
                a, b = i
                for i in [(a, b), (b, a)]:
                    for join in range(3):
                        for index_0 in range(2):
                            for index_1 in range(2):
                                if i[0] in self.table_alias:
                                    left_table = self.table_alias[i[0]]
                                else:
                                    left_table = i[0]
                                if i[1] in self.table_alias:
                                    right_table = self.table_alias[i[1]]
                                else:
                                    right_table = i[1]
                                vector_0 = [0] * (len(all_tables) * 2 + len(self.join_type))
                                vector_1 = [0] * (len(all_tables) * 2 + len(self.join_type))
                                vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))

                                vector_0[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
                                vector_0[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0

                                vector_1[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                                vector_1[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1

                                vector_parent[len(self.join_type) + all_tables[left_table] * 2] = 1 # self.selectivity[left_table]
                                vector_parent[len(self.join_type) + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                                vector_parent[len(self.join_type) + 1 + all_tables[left_table] * 2] = index_0
                                vector_parent[len(self.join_type) + 1 + all_tables[right_table] * 2] = index_1
                                vector_parent[join] = 1

                                # now_value = self.value_net([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                                # if now_value < min_cost:
                                #     node = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
                                #     min_cost = now_value
                                #     join_index = i_index
                                #     self.total_vector = vector_parent.copy()
                                #     # self.total_vector[join] = 0
                                #     left_index_state = index_0
                                #     right_index_state = index_1
                                #     join_type_index = join

                                # become batch training
                                all_actions.append([self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                                join_indexes.append(i_index)
                                left_index_states.append(index_0)
                                right_index_states.append(index_1)
                                join_type_indexes.append(join)
                i_index += 1

            # preprocessing!
            if self.distributed:
                encode = []
                trees = []
                local_device = torch.device('cuda:' + str(0))
                for i in all_actions:
                    encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                    trees.append(i[1])
                x = torch.cat(encode, dim=0)
                now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=local_device)
                now_actions = [x, list(now_trees)]
            else:
                now_actions = all_actions

            if not self.test:
                now_value,_ = self.value_net(now_actions)
            else:
                now_value,_ = self.target_net(now_actions)
            min_idx = torch.argmin(now_value, dim=0).item()
            join_index = join_indexes[min_idx]
            node = all_actions[min_idx][1]
            min_cost = now_value[min_idx].item()
            left_index_state = left_index_states[min_idx]
            right_index_state = right_index_states[min_idx]
            join_type_index = join_type_indexes[min_idx]
            self.total_vector = list(node[0])

            self.collection = node
            self.state_list.append(node)
            # self.reward.append(float(min_cost))
            if left_index_state == 1:
                self.add_hint_index(self.join_table[join_index][0])
            if right_index_state == 1:
                self.add_hint_index(self.join_table[join_index][1])
            self.hint_leading.append(self.join_table[join_index][0])
            self.add_hint_join(self.join_table[join_index][1], join_type_index)
            # used table 中也要用别名，对于1个表两个alias，也要当成不同的看待
            self.used_table += self.join_table[join_index]
            self.join_table.pop(join_index)
        self.check_state()


        if not self.test:
            if self.state:
                all_actions = self.getMinQ()
                # print(self.collection)
                self.Q_values.append(([self.query_encode, self.collection], all_actions))
            else:
                self.Q_values.append(([self.query_encode, self.collection], []))

    def choose_action(self):
        def transformer(x, encode, local_device):
            b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
            if encode:
                return torch.cat((encode, b))
            else:
                return b

        operator_num = len(self.join_type)
        ret = random.random()
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
        if ret < self.random_num:
            # vector = [0] * (len(all_tables) * 2 + len(self.join_type))
            random_table = random.randint(0, len(now_need_join_table) - 1)
            random_index = random.randint(0, 1)
            random_join = random.randint(0, 2)

            if now_need_join_table[random_table][0]  in self.table_alias:
                right_table = self.table_alias[now_need_join_table[random_table][0]]
            else:
                right_table = now_need_join_table[random_table][0]
            vector_right = [0] * (len(all_tables) * 2 + operator_num)
            vector_right[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            # vector_right[operator_num + all_tables[right_table] * 2] = 1

            vector_right[operator_num + 1 + all_tables[right_table] * 2] = random_index
            tuple_right = (tuple(vector_right),)
            vector_parent = self.total_vector.copy()
            # 前三项为0！
            vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
            vector_parent[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
            # vector_parent[operator_num + all_tables[right_table] * 2] = 1

            vector_parent[operator_num + 1 + all_tables[right_table] * 2] = random_index
            vector_parent[random_join] = 1
            # now_value = self.value_net([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
            # min_cost = now_value
            self.collection = (tuple(vector_parent), self.collection, tuple_right)
            self.state_list.append((tuple(vector_parent), self.collection, tuple_right))
            # self.latency = now_value
            # self.reward.append(float(now_value))
            if random_index == 1:
                self.add_hint_index(now_need_join_table[random_table][0])
            self.add_hint_join(now_need_join_table[random_table][0], random_join)
            # self.total_vector[operator_num + all_tables[right_table] * 2] = self.selectivity[
            #     right_table]
            # self.total_vector[len(self.join_type) + 1 + all_tables[self.tables[random_index]] * 2] = random_index
            self.total_vector = vector_parent.copy()
            self.used_table.append(now_need_join_table[random_table][0])
            self.join_table.pop(now_need_join_table[random_table][1])
        else:
            # node = []
            # min_cost = float("inf")
            # join_index = 0
            # join_type_index = 0
            # right_index_state = 0
            # final_parent_vector = None
            all_actions = []
            join_indexes = []
            right_index_states = []
            join_type_indexes = []

            for i in range(len(now_need_join_table)):
                for join in range(3):
                    for index in range(2):
                        if now_need_join_table[i][0] in self.table_alias:
                            right_table = self.table_alias[now_need_join_table[i][0]]
                        else:
                            right_table = now_need_join_table[i][0]
                        vector_right = [0] * (len(all_tables) * 2 + operator_num)
                        vector_right[operator_num + all_tables[right_table] * 2] = 1 #self.selectivity[right_table]
                        vector_right[operator_num + 1 + all_tables[right_table] * 2] = index
                        tuple_right = (tuple(vector_right),)
                        vector_parent = self.total_vector.copy()
                        vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
                        vector_parent[operator_num + all_tables[right_table] * 2] = 1 #self.selectivity[right_table]
                        vector_parent[
                            operator_num + 1 + all_tables[right_table] * 2] = index
                        vector_parent[join] = 1

                        # now_value = self.value_net(
                        #     [self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                        # print(now_value)
                        # if now_value < min_cost:
                        #     node = (tuple(vector_parent), self.collection, tuple_right)
                        #     min_cost = now_value
                        #     join_index = i
                        #     final_parent_vector = vector_parent
                        #     # self.total_vector[join] = 0
                        #     right_index_state = index
                        #     join_type_index = join

                        # pay attention to batch training
                        all_actions.append([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                        join_indexes.append(i)
                        right_index_states.append(index)
                        join_type_indexes.append(join)

            # now_values = self.value_net(all_actions)

            # preprocessing!
            if self.distributed:
                encode = []
                trees = []
                local_device = torch.device('cuda:' + str(0))
                for i in all_actions:
                    encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                    trees.append(i[1])
                x = torch.cat(encode, dim=0)
                now_trees = prepare_trees(trees, transformer, left_child, right_child, None,
                                          local_device=local_device)
                now_actions = [x, list(now_trees)]
            else:
                now_actions = all_actions

            if not self.test:
                now_values,_ = self.value_net(now_actions)
            else:
                now_values,_ = self.target_net(now_actions)
            min_idx = torch.argmin(now_values, dim=0).item()
            join_index = join_indexes[min_idx]
            node = all_actions[min_idx][1]
            min_cost = now_values[min_idx].item()
            right_index_state = right_index_states[min_idx]
            join_type_index = join_type_indexes[min_idx]
            self.total_vector = list(node[0])
            # print(now_values.shape)
            # print(node)
            # print(min_cost)
            # print(right_index_state)
            # print(join_type_index)
            # print(self.total_vector)
            # print(join_index)
            # print('*' * 50)


            # self.total_vector = final_parent_vector
            self.collection = node
            self.state_list.append(node)
            # self.reward.append(float(min_cost))
            if right_index_state == 1:
                self.add_hint_index(now_need_join_table[join_index][0])
            self.add_hint_join(now_need_join_table[join_index][0], join_type_index)
            self.used_table.append(now_need_join_table[join_index][0])
            self.join_table.pop(now_need_join_table[join_index][1])
        self.check_state()


        if not self.test:
            if self.state:
                all_actions = self.getMinQ()
                self.Q_values.append(([self.query_encode, self.collection], all_actions))
            else:
                self.Q_values.append(([self.query_encode, self.collection], []))

    def getMinQ(self):
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

        node = []
        min_cost = float("inf")
        all_actions = []
        for i in range(len(now_need_join_table)):
            for join in range(3):
                for index in range(2):
                    if now_need_join_table[i][0] in self.table_alias:
                        right_table = self.table_alias[now_need_join_table[i][0]]
                    else:
                        right_table = now_need_join_table[i][0]
                    vector_right = [0] * (len(all_tables) * 2 + operator_num)
                    vector_right[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                    vector_right[operator_num + 1 + all_tables[right_table] * 2] = index
                    tuple_right = (tuple(vector_right),)
                    vector_parent = self.total_vector.copy()
                    vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
                    vector_parent[operator_num + all_tables[right_table] * 2] = 1 # self.selectivity[right_table]
                    vector_parent[
                        operator_num + 1 + all_tables[right_table] * 2] = index
                    vector_parent[join] = 1
                    # different!

                    # now_value = self.target_net(
                    #     [self.query_encode, (tuple(vector_parent), self.collection, tuple_right)]).detach()
                    # # print(vector_parent)
                    # # print(now_value)
                    # if now_value < min_cost:
                    #     node = (tuple(vector_parent), self.collection, tuple_right)
                    #     min_cost = now_value
                    #     join_index = i
                    #     final_parent_vector = vector_parent
                    #     # self.total_vector[join] = 0
                    #     right_index_state = index
                    #     join_type_index = join

                    all_actions.append([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])

        return all_actions

    def check_state(self):
        # if self.file_name not in [7, 8, 18, 20, 21]:
        # for every relation
        i = 0
        while i < len(self.join_table):
            if self.join_table[i][0] in self.used_table and self.join_table[i][1] in self.used_table:
                self.join_table.pop(i)
            else:
                i += 1
        if len(self.join_table) == 0:
            self.state = False

    def get_hint_leading(self):
        if len(self.hint_leading) < 2:
            return ''
        else:
            return 'Leading('+" ".join(self.hint_leading) + ')'

    def get_hint_index(self):
        return " ".join(self.hint_index)

    def get_hint_join(self):
        return " ".join(self.hint_join)