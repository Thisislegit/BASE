import sys
from utils.Connection import *
from utils.Utils import *
from utils.PlanTree import *
from torch import optim
from utils.Net import Net
import torch
import numpy as np
from torch import nn
from Environment import postgresql
# from utils.DNNRLGOO import DNNRLGOO
from utils.DNNRLGOO_Merge import DNNRLGOO
from utils.ChooseStepDNNRL import ChooseStepDNNRL
from utils.Exhaustive import Exhaustive
import hashlib
from torch.optim import lr_scheduler
import random
# random.seed(2021)
import collections
from utils.DQN import DQN
# from utils.PlanSQL import travesal_SQL
from Plan_SQL_test import travesal_SQL
import linecache
from utils.Arguments import parser
args = parser.parse_args()


EXECUTION_MAX_TIME = 90000
GOAL_FIELD = 'Total Cost'#Actual Total Time，Total Cost
GAMMA = 0.9
NORMALIZATION = 2

USEABLE_FILE = []
for file in sorted(os.listdir('./train_plan/index/')):
    with open('./train_plan/index/'+file) as f:
        for idx in f.readlines():
            USEABLE_FILE.append(int(idx))
USEABLE_FILE.sort()

TEST_FILE = [84769, 87006, 55149, 25097, 64236, 98753, 48072, 6440, 27378, 48185, 92664, 16325, 84093, 75202, 68153, 55524, 5121, 83811, 47439, 20268, 92855, 44727, 101212, 84454, 28432, 34103, 37796, 40097, 11349, 22318, 99675, 44883, 11667, 54974, 47054, 28690, 13770, 458, 54393, 89689, 19780, 17207, 82217, 40672, 58875, 37306, 83340, 5392, 7751, 17681, 59265, 70580, 58643, 63722, 94566, 40259, 27291, 17928, 43320, 62387, 70075, 94054, 10893, 1081, 94931, 29891, 63817, 7591, 95293, 43924, 59149, 73094, 10154, 40180, 80204, 20916, 8101, 3117, 75034, 54907, 4641, 76214, 27851, 73197, 2580, 59027, 69446, 18021, 94306, 3882, 42324, 3538, 40466, 97383, 36002, 24201, 60625, 52101, 33716, 79841]


def main():
    def test():
        # save CheckPoint
        checkpoint_path = './Models/Model_checkpoint/now_{}.pth'.format(dqn.cur_epoch)
        state = {'model': dqn.net.state_dict()}
        torch.save(state, checkpoint_path)

        path = 'reward_log_NEW.json'
        if os.path.exists(path):
            reward_log = json.load(open(path, encoding='utf-8'))
        else:
            reward_log = []
        all = []
        for idx in TEST_FILE:
            sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
            sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
            SQL_encode = travesal_SQL(sql)
            query_encode = SQL_encode

            join_table = get_join_table(query_encode[2])
            join_table = reverse_kv(query_encode[3], join_table)
            query_encode = replace_alias(query_encode)

            join_table = remove_same_table(join_table, 1)
            now_tables = get_tables(query_encode)
            query_vector = get_vector(query_encode)

            file_name = 1
            rl = DNNRLGOO(dqn.net, dqn.optimizer, now_tables[1], query_vector, query_encode[4],
                          join_table, file_name, query_encode[3], dqn.target_net, test=True)
            rl.random_num = dqn.epsilon_start
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


    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()
    dqn = DQN(Net(path=args.path, learning_rate=args.learning_rate, max_lr=args.max_lr, steps_per_epoch=args.steps_per_epoch, epoch=args.epoch_end-args.epoch_start),
              Net(path=args.path, learning_rate=args.learning_rate, max_lr=args.max_lr, steps_per_epoch=args.steps_per_epoch, epoch=args.epoch_end-args.epoch_start),
              epsilon_start=args.epsilon_start,
                epsilon_decay=args.epsilon_decay,
                epsilon_end=args.epsilon_end,
                path=args.path,
                batch_size=args.batch_size,
                sync_batch_size=args.sync_batch_size,
                capacity=args.capacity)
    reward_log = []
    for epoch in range(args.epoch_start, args.epoch_end):
        # random.shuffle(d_list)

        print('episode:', epoch)
        # 初始化记录loss为空表格
        dqn.all_loss = []
        dqn.cur_epoch = epoch

        for num, idx in enumerate(USEABLE_FILE):
            if idx not in TEST_FILE:
                # print(idx)
                sql = linecache.getline('./train_plan/query_multi.sql', idx+1)
                sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
                # 加入 sql encode
                # [[Filters],[Join Filter],[Cond], [Relation_alias],[selectivity], [w2c]]
                SQL_encode = travesal_SQL(sql)
                query_encode = SQL_encode
                join_table = get_join_table(query_encode[2])
                join_table = reverse_kv(query_encode[3], join_table)
                query_encode = replace_alias(query_encode)

                join_table = remove_same_table(join_table, 7)
                now_tables = get_tables(query_encode)
                query_vector = get_vector(query_encode)

                file_name = 1

                rl = DNNRLGOO(dqn.net, dqn.optimizer, now_tables[1], query_vector, query_encode[4],
                              join_table, file_name, query_encode[3], dqn.target_net)
                rl.random_num = dqn.epsilon_start
                # rl.random_num = 0 # 测试的时候用
                rl.init_state()
                while rl.state:
                    rl.choose_action()

                terminate_plan = get_hint_SQL_explain(sql,
                                                      '/*+ ' + rl.get_hint_leading() + ' ' + rl.get_hint_join() + ' ' + rl.get_hint_index() + ' */ ' + ' EXPLAIN (format json)',
                                                      str(11),
                                                      conn.conn)
                # print(terminate_plan)
                f = 1
                if terminate_plan == -1:
                    continue
                else:
                    # print(terminate_plan['Plan']['Actual Total Time'])
                    result_now = traversal_plan_tree_cost(terminate_plan['Plan'], f, query_vector)
                    #rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                    #for i in range(1, len(result_now[0])):
                    #    rewards.append(round(pow(result_now[0][i] - result_now[0][i-1], 1 / NORMALIZATION), 6))
                    #result_now[0] = rewards
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

                    # print(result_now[1])
                    assert len(result_now[0]) == len(rl.Q_values)
                    if result_now is not None:
                        for i in range(len(result_now[0])):
                            # todo: 改成真实的 now state
                            now_state = result_now[2][i]
                            # now_state = result_now[2][i] # now_state yong zhen shi de
                            state = 1 if i != len(result_now[2])-1 else 0

                            # possible next state
                            next_possible = []
                            if state:
                                all_right = set([i[1][2] for i in rl.Q_values[i][1]])
                                encode = rl.Q_values[i][1][0][0]
                                for tuple_right in all_right:
                                    assert len(tuple_right) == 1
                                    collection = now_state[1]
                                    for join in [0, 2]:
                                        vector_parent = list(collection[0])
                                        vector_parent[0] = vector_parent[1] = vector_parent[2] = 0
                                        vector_parent[join] = 1
                                        for idx, pos in enumerate(tuple_right[0]):
                                            if pos != 0:
                                                vector_parent[idx] = 1
                                        next_possible.append([encode, (tuple(vector_parent), collection, tuple_right)])
                            next_state = next_possible
                            reward = result_now[0][i]

                            dqn.buffer.add([reward, now_state, next_state, state])




                # add optimal plan
                if num % 5000 == 0:
                    sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
                    sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
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
                    result_now = traversal_plan_tree_cost(terminate_plan['Plan'], f, query_vector)
                    #rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                    #for i in range(1, len(result_now[0])):
                    #    rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                    #result_now[0] = rewards
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

                    for i in range(len(result_now[0])):
                        now_state = result_now[2][i]
                        state = 1 if i != len(result_now[2])-1 else 0
                        next_state = [result_now[2][i+1]] if i != len(result_now[2])-1 else []
                        reward = result_now[0][i]

                        dqn.buffer.add([reward, now_state, next_state, state])



            if num % 10 == 0:
                dqn.learn()
        print('avergae loss:', sum(dqn.all_loss) / len(dqn.all_loss) if len(dqn.all_loss) != 0 else 0)

        dqn.update_epsilon()
        conn.reconnect()
        set_statement_timeout(EXECUTION_MAX_TIME)
        load_extension()
        test()






if __name__ == '__main__':
    main()