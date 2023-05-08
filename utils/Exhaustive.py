import os
from torch import nn
import numpy as np
from utils.Connection import all_tables
from utils.PlanTree import OPERATOR_TYPE
import torch
import random
import itertools



class Exhaustive(object):
    def __init__(self, model, optimizer, table, query_encode, name, selectivity, join_table, file_name, table_alias):
        self.target_net = model
        self.value_net = model
        self.optimizer = optimizer
        self.compute_loss = nn.MSELoss()
        self.tables = table
        self.collection = ()
        self.query_encode = query_encode
        self.plan_encode = []
        self.join_type = ["Hash Join", "Merge Join", "Nested Loop"]
        self.type_dic = ['HashJoin', 'MergeJoin', 'NestLoop']
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
        self.random_num = 0.1
        self.name = name
        self.selectivity = selectivity
        self.used_table = []
        self.reward = []
        self.state_list = []
        self.file_name = file_name
        self.table_alias = table_alias

    def choose_nd(self, num):
        def permutation_join(vector, tables_array, now_used_tables):
            if len(tables_array) == 0:
                vector.sort(key=lambda u: (u[2]))
                return [vector[0]]
            if self.file_name not in [7, 8, 18, 21]:
                for i_tables_array in range(len(tables_array)):
                    if tables_array[i_tables_array][0] in now_used_tables and tables_array[i_tables_array][1] in now_used_tables:
                        tables_array.pop(i_tables_array)
                        break
            all_vector = []
            for vector_i in range(0, len(vector)):
                for tables_array_index in range(0, len(tables_array)):
                    new_vector = []
                    new_tables_array = tables_array.copy()
                    new_now_used_tables = now_used_tables.copy()
                    if tables_array[tables_array_index][0] in now_used_tables:
                        right_table_name = tables_array[tables_array_index][1]
                        new_tables_array.pop(tables_array_index)
                    elif tables_array[tables_array_index][1] in now_used_tables:
                        right_table_name = tables_array[tables_array_index][0]
                        new_tables_array.pop(tables_array_index)
                    else:
                        continue
                    if right_table_name in self.table_alias:
                        right_table = self.table_alias[right_table_name]
                    else:
                        right_table = right_table_name
                    new_now_used_tables.append(right_table_name)
                    for now_join in range(3):
                        for now_index in range(2):
                            vector_right = [0] * (len(all_tables) * 2 + len(self.join_type))
                            vector_right[len(self.join_type) - 1 + all_tables[right_table] * 2] = self.selectivity[
                                right_table]
                            vector_right[len(self.join_type) + all_tables[right_table] * 2] = now_index
                            tuple_right = (tuple(vector_right),)
                            now_vector_parent = vector[vector_i][0].copy()
                            now_vector_parent[len(self.join_type) - 1 + all_tables[right_table] * 2] = self.selectivity[
                                right_table]
                            now_vector_parent[
                                len(self.join_type) + all_tables[right_table] * 2] = now_index
                            now_vector_parent[now_join] = 1
                            now_value = self.value_net(
                                [self.query_encode, (tuple(now_vector_parent), vector[vector_i][1], tuple_right)])
                            now_v_4 = vector[vector_i][4].copy()
                            now_v_4.append(self.type_dic[now_join] + '( ' + " ".join(new_now_used_tables) + ' )')
                            now_v_5 = vector[vector_i][5].copy()
                            if now_index == 1:
                                now_v_5.append('IndexScan( ' + right_table_name + ' )')
                            new_vector.append([now_vector_parent, (tuple(now_vector_parent), vector[vector_i][1], tuple_right), now_value+vector[vector_i][2], new_now_used_tables, now_v_4, now_v_5])
                    mid_goal = permutation_join(new_vector, new_tables_array, new_now_used_tables)
                    for mid_goal_value in mid_goal:
                        all_vector.append(mid_goal_value)
            # print(len(all_vector))
            # print(all_vector[0][2])
            all_vector.sort(key=lambda u: (u[2]))
            # print(all_vector[0][2])
            return [all_vector[0]]
        rank_vector = []
        print(self.join_table.copy())
        for i in range(0, len(self.join_table)):
            now_all_tables = self.join_table.copy()
            first_need_tablse = now_all_tables[i]
            now_tables = list(now_all_tables[i])
            if now_tables[0] in self.table_alias:
                now_tables[0] = self.table_alias[now_tables[0]]
            if now_tables[1] in self.table_alias:
                now_tables[1] = self.table_alias[now_tables[1]]
            now_all_tables.pop(i)
            for join in range(3):
                for index_0 in range(2):
                    for index_1 in range(2):
                        # permutation_join([], now_all_tables)
                        index_array = []
                        join_array = []
                        if index_0 == 1:
                            index_array.append('IndexScan( '+first_need_tablse[0]+' )')
                        if index_1 == 1:
                            index_array.append('IndexScan( '+first_need_tablse[1]+' )')
                        join_array.append(self.type_dic[join] + '( ' + " ".join(list(now_tables)) + ' )')

                        vector_0 = [0] * (len(all_tables) * 2 + len(self.join_type))
                        vector_1 = [0] * (len(all_tables) * 2 + len(self.join_type))
                        vector_parent = [0] * (len(all_tables) * 2 + len(self.join_type))

                        vector_0[len(self.join_type) - 1 + all_tables[now_tables[0]] * 2] = self.selectivity[now_tables[0]]
                        vector_0[len(self.join_type) + all_tables[now_tables[0]] * 2] = index_0
                        vector_1[len(self.join_type) - 1 + all_tables[now_tables[1]] * 2] = self.selectivity[now_tables[1]]
                        vector_1[len(self.join_type) + all_tables[now_tables[1]] * 2] = index_1

                        vector_parent[len(self.join_type) - 1 + all_tables[now_tables[1]] * 2] = self.selectivity[now_tables[1]]
                        vector_parent[len(self.join_type) - 1 + all_tables[now_tables[1]] * 2] = self.selectivity[now_tables[1]]
                        vector_parent[len(self.join_type) + all_tables[now_tables[0]] * 2] = index_0
                        vector_parent[len(self.join_type) + all_tables[now_tables[1]] * 2] = index_1
                        vector_parent[join] = 1

                        vector_collection = (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))
                        now_value = self.value_net(
                            [self.query_encode, (tuple(vector_parent), (tuple(vector_0),), (tuple(vector_1),))])
                        goal = permutation_join([[vector_parent, vector_collection, now_value, list(first_need_tablse), join_array, index_array]], now_all_tables, list(first_need_tablse), )
                        for goal_val in goal:
                            self.get_all_hint(goal_val)
                            rank_vector.append(goal_val)
        rank_vector.sort(key=lambda u:(u[2]))
        print(rank_vector[0][2],self.get_all_hint(rank_vector[0]))
        print(rank_vector[1][2],self.get_all_hint(rank_vector[1]))
        print(len(rank_vector))
        exit()

    def get_all_hint(self, vector):
        str = '/*+ ' + 'Leading('+" ".join(vector[3]) + ')'+ ' ' + " ".join(vector[4]) + ' ' + " ".join(vector[5]) + ' */'
        return str
