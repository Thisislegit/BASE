import os
from torch import nn
import numpy as np
from utils.Connection import all_tables
from utils.PlanTree import OPERATOR_TYPE
import torch
import random
import itertools


class ChooseStepDNNRL(object):
    def __init__(self, model, optimizer, table, query_encode, name, selectivity, join_table, file_name, table_alias, nd_step):
        self.target_net = model
        self.value_net = model
        self.optimizer = optimizer
        self.compute_loss = nn.MSELoss()
        self.tables = table
        self.collection = ()
        self.query_encode = query_encode
        self.plan_encode = []
        self.join_type = ["Hash Join", "Merge Join", "Nested Loop"]
        self.state = True
        self.latency = 0
        self.sql = ''
        self.hint_index = []
        self.hint_join = []
        self.hint_leading = []
        self.now_stack = []
        self.join_table = join_table
        self.total_vector = [0] * (len(all_tables) * 2 + len(self.join_type))
        self.latency = 0
        self.loss = 0
        self.count = 0
        self.random_num = 0.5
        self.name = name
        self.selectivity = selectivity
        self.used_table = []
        self.reward = []
        self.state_list = []
        self.file_name = file_name
        self.table_alias = table_alias
        self.goal_step = nd_step

    def add_hint_index(self, table):
        self.hint_index.append('IndexScan( '+table+' )')

    def add_hint_join(self, table, join_type):
        self.hint_leading.append(table)
        type_dic = ['HashJoin', 'MergeJoin', 'NestLoop']
        #self.hint_join.append(type_dic[join_type] + '( ' + " ".join(self.hint_leading) + ' )')
        self.hint_join.append('HashJoin( ' + " ".join(self.hint_leading) + ' )')

    def init_state(self):
        ret = random.random()
        min_cost = float("inf")
        join_index = 0
        i_index = 0
        join_type_index = 0
        left_index_state = 0
        right_index_state = 0

        all_results = []
        for i in self.join_table:
            for join in range(0,1):
                for index_0 in range(2):
                    for index_1 in range(1, 2):
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

                        vector_0[len(self.join_type) - 1 + all_tables[left_table] * 2] = self.selectivity[left_table]
                        vector_0[len(self.join_type) + all_tables[left_table] * 2] = index_0

                        vector_1[len(self.join_type) - 1 + all_tables[right_table] * 2] = self.selectivity[right_table]
                        vector_1[len(self.join_type) + all_tables[right_table] * 2] = index_1

                        vector_parent[len(self.join_type) - 1 + all_tables[left_table] * 2] = self.selectivity[
                            left_table]
                        vector_parent[len(self.join_type) - 1 + all_tables[right_table] * 2] = self.selectivity[
                            right_table]
                        vector_parent[len(self.join_type) + all_tables[left_table] * 2] = index_0
                        vector_parent[len(self.join_type) + all_tables[right_table] * 2] = index_1
                        vector_parent[join] = 1

                        now_value = self.value_net(
                            [self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                        node = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
                        all_results.append([now_value, node, i_index, vector_parent, index_0, index_1, join])
            i_index += 1
        all_results.sort(key=lambda u: (u[0]))
        if self.goal_step == 0:
            goal_result = all_results[1]
        else:
            goal_result = all_results[0]
        self.total_vector = goal_result[3]
        self.total_vector[goal_result[6]] = 0
        self.collection = goal_result[1]
        self.state_list.append(goal_result[1])
        self.reward.append(float(goal_result[0]))
        if goal_result[4] == 1:
            self.add_hint_index(self.join_table[goal_result[2]][0])
        if goal_result[5] == 1:
            self.add_hint_index(self.join_table[goal_result[2]][1])
        self.hint_leading.append(self.join_table[goal_result[2]][0])
        self.add_hint_join(self.join_table[goal_result[2]][1], goal_result[6])
        self.used_table += self.join_table[goal_result[2]]
        self.join_table.pop(goal_result[2])

        self.check_state()

    def choose_action(self):
        operator_num = len(self.join_type)
        ret = random.random()
        now_need_join_table = []

        for i in range(len(self.join_table)):
            if self.join_table[i][0] in self.used_table:
                now_need_join_table.append([self.join_table[i][1], i])
            if self.join_table[i][1] in self.used_table:
                now_need_join_table.append([self.join_table[i][0], i])

        node = []
        min_cost = float("inf")
        join_index = 0
        join_type_index = 0
        right_index_state = 0
        all_results = []
        for i in range(len(now_need_join_table)):
            for join in range(0,1):
                for index in range(1, 2):
                    if now_need_join_table[i][0] in self.table_alias:
                        right_table = self.table_alias[now_need_join_table[i][0]]
                    else:
                        right_table = now_need_join_table[i][0]
                    vector_right = [0] * (len(all_tables) * 2 + operator_num)
                    vector_right[operator_num - 1 + all_tables[right_table] * 2] = self.selectivity[
                        right_table]
                    vector_right[operator_num + all_tables[right_table] * 2] = index
                    tuple_right = (tuple(vector_right),)
                    vector_parent = self.total_vector.copy()
                    vector_parent[operator_num - 1 + all_tables[right_table] * 2] = self.selectivity[right_table]
                    vector_parent[
                        operator_num + all_tables[right_table] * 2] = index
                    vector_parent[join] = 1
                    now_value = self.value_net(
                        [self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                    node = (tuple(vector_parent), self.collection, tuple_right)
                    all_results.append([now_value, node, i, vector_parent, index, join])
        all_results.sort(key=lambda u: (u[0]))
        if self.goal_step == len(self.used_table)-1:
            goal_result = all_results[1]
        else:
            goal_result = all_results[0]
        self.total_vector = goal_result[3]
        self.total_vector[goal_result[5]] = 0
        self.collection = goal_result[1]
        self.state_list.append(goal_result[1])
        self.reward.append(float(goal_result[0]))
        if goal_result[4] == 1:
            self.add_hint_index(now_need_join_table[goal_result[2]][0])
        self.add_hint_join(now_need_join_table[goal_result[2]][0], goal_result[5])
        self.used_table.append(now_need_join_table[goal_result[2]][0])
        self.join_table.pop(now_need_join_table[goal_result[2]][1])
        self.check_state()

    def check_state(self):
        if self.file_name not in [7, 8, 18, 20, 21]:
            for i in range(len(self.join_table)):
                if self.join_table[i][0] in self.used_table and self.join_table[i][1] in self.used_table:
                    self.join_table.pop(i)
                    break
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
