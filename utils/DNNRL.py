import os
from torch import nn
import numpy as np
from utils.Connection import all_tables
from utils.PlanTree import OPERATOR_TYPE
import torch
import random


# DNN guided RL Model
class DNNRL(object):
    def __init__(self, model, optimizer, table, query_encode, name, selectivity):
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
        self.total_vector = [0] * (len(all_tables) * 2 + len(self.join_type))
        self.latency = 0
        self.loss = 0
        self.count = 0
        self.random_num = 0.05
        self.name = name
        self.selectivity = selectivity

    def add_hint_index(self, table):
        self.hint_index.append('IndexScan( '+table+' )')

    def add_hint_join(self, table, join_type):
        self.hint_leading.append(table)
        type_dic = ['HashJoin', 'MergeJoin', 'NestLoop']
        self.hint_join.append(type_dic[join_type] + '( ' + " ".join(self.hint_leading) + ' )')

    # function to choose first table
    def init_state(self):
        node = ()
        min_cost = float("inf")
        table_index = -1
        index_state = -1
        ret = random.random()
        if ret < self.random_num:
            vector = [0] * (len(all_tables) * 2 + len(self.join_type))
            random_table = random.randint(0, len(self.tables)-1)
            random_index = random.randint(0, 1)
            vector[len(self.join_type) - 1 + all_tables[self.tables[random_table]] * 2] = self.selectivity[self.tables[random_table]]
            vector[len(self.join_type) + all_tables[self.tables[random_table]] * 2] = random_index
            min_cost = self.value_net([self.query_encode, (tuple(vector),)])
            node = (tuple(vector),)
            table_index = random_table
            index_state = random_index
        else:
            for i in range(len(self.tables)):
                for index in range(2):
                    vector = [0] * (len(all_tables) * 2 + len(self.join_type))
                    vector[len(self.join_type) - 1 + all_tables[self.tables[i]] * 2] = self.selectivity[self.tables[i]]
                    vector[len(self.join_type) + all_tables[self.tables[i]] * 2] = index
                    now_value = self.value_net([self.query_encode, (tuple(vector),)])
                    if now_value < min_cost:
                        node = (tuple(vector),)
                        min_cost = now_value
                        table_index = i
                        index_state = index
        if index_state == 1:
            self.add_hint_index(self.tables[table_index])
        self.hint_leading.append(self.tables[table_index])#join order 添加
        self.total_vector[len(self.join_type) - 1 + all_tables[self.tables[table_index]] * 2] = 1
        self.total_vector[len(self.join_type) + all_tables[self.tables[table_index]] * 2] = index_state
        self.tables.pop(table_index)
        self.collection = node
        self.latency = min_cost

    def choose_action(self):
        operator_num = len(self.join_type)
        min_cost = float("inf")
        table_index = -1
        index_state = -1
        join_type = -1
        node = ()
        ret = random.random()
        if ret < self.random_num:
            vector = [0] * (len(all_tables) * 2 + len(self.join_type))
            random_table = random.randint(0, len(self.tables) - 1)
            random_index = random.randint(0, 1)
            random_join = random.randint(0, 2)
            vector_right = [0] * (len(all_tables) * 2 + operator_num)
            vector_right[operator_num - 1 + all_tables[self.tables[random_table]] * 2] = self.selectivity[self.tables[random_table]]
            vector_right[operator_num + all_tables[self.tables[random_table]] * 2] = random_index
            tuple_right = (tuple(vector_right),)
            vector_parent = self.total_vector.copy()
            vector_parent[operator_num - 1 + all_tables[self.tables[random_table]] * 2] = self.selectivity[self.tables[random_table]]
            vector_parent[operator_num + all_tables[self.tables[random_table]] * 2] = random_index
            vector_parent[random_join] = 1
            now_value = self.value_net([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
            node = (tuple(vector_parent), self.collection, tuple_right)
            min_cost = now_value
            table_index = random_table
            index_state = random_index
            join_type = random_join
        else:
            for i in range(len(self.tables)):
                for join in range(3):
                    for index in range(2):
                        vector_right = [0] * (len(all_tables) * 2 + operator_num)
                        vector_right[operator_num - 1 + all_tables[self.tables[i]] * 2] = self.selectivity[self.tables[i]]
                        vector_right[operator_num + all_tables[self.tables[i]] * 2] = index
                        tuple_right = (tuple(vector_right),)
                        vector_parent = self.total_vector.copy()
                        vector_parent[operator_num - 1 + all_tables[self.tables[i]] * 2] = self.selectivity[self.tables[i]]
                        vector_parent[operator_num + all_tables[self.tables[i]] * 2] = index
                        vector_parent[join] = 1
                        now_value = self.value_net([self.query_encode, (tuple(vector_parent), self.collection, tuple_right)])
                        if now_value < min_cost:
                            node = (tuple(vector_parent), self.collection, tuple_right)
                            min_cost = now_value
                            table_index = i
                            index_state = index
                            join_type = join
        if index_state == 1:
            self.add_hint_index(self.tables[table_index])
        self.collection = node
        self.latency = min_cost
        self.add_hint_join(self.tables[table_index], join_type)
        self.total_vector[len(self.join_type) - 1 + all_tables[self.tables[table_index]] * 2] = 1
        self.total_vector[len(self.join_type) + all_tables[self.tables[table_index]] * 2] = index_state
        self.tables.pop(table_index)
        self.check_state()
        return min_cost

    def learn(self, cost):
        self.value_net.train()
        output = self.value_net([self.query_encode, self.collection])
        self.optimizer.zero_grad()
        compute_loss = nn.L1Loss()
        loss_now = compute_loss(output, torch.tensor([round(pow(cost, 1/4), 6)]))
        loss_now.backward()
        self.optimizer.step()
        self.loss += loss_now
        self.count += 1

    def check_state(self):
        if len(self.tables) == 0:
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
