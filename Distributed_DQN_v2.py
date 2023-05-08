import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from utils.Net import Net
from utils.DQN import Distibut_DQN_v2
from utils.Arguments import parser
import math
# from utils.DNNRLGOO import DNNRLGOO
from utils.DNNRLGOO_Merge import DNNRLGOO
from utils.Utils import *
from utils.PlanTree import *
from utils.Connection import Conn
#from utils.PlanSQL import travesal_SQL
from Plan_SQL_test import travesal_SQL
import random
import time
from TreeConvolution.util_1 import prepare_trees
from TreeConvolution.tcnn import left_child, right_child
from collections import OrderedDict
# python -u RLGOOTest_NEW.py --path ./Models/now.pth --epoch_start 1 --epoch_end 100 --epsilon_decay 0.95 --epsilon_end 0.02 --capacity 60000 --batch_size 512 --sync_batch_size 50 --steps_per_epoch 1000 --max_lr 0.0008 --learning_rate 0.0003 > log_NEW.txt 2>&1

EXECUTION_MAX_TIME = 90000
GOAL_FIELD = 'Total Cost'#Actual Total Time，Total Cost
GAMMA = 0.9
NORMALIZATION = 2

USEABLE_FILE = []
for file in sorted(os.listdir('./train_plan/index/')):
    with open('./train_plan/index/'+file) as f:
        for idx in f.readlines():
            USEABLE_FILE.append(int(idx))
# USEABLE_FILE.sort()
random.shuffle(USEABLE_FILE)

TEST_FILE = [84769, 87006, 55149, 25097, 64236, 98753, 48072, 6440, 27378, 48185, 92664, 16325, 84093, 75202, 68153, 55524, 5121, 83811, 47439, 20268, 92855, 44727, 101212, 84454, 28432, 34103, 37796, 40097, 11349, 22318, 99675, 44883, 11667, 54974, 47054, 28690, 13770, 458, 54393, 89689, 19780, 17207, 82217, 40672, 58875, 37306, 83340, 5392, 7751, 17681, 59265, 70580, 58643, 63722, 94566, 40259, 27291, 17928, 43320, 62387, 70075, 94054, 10893, 1081, 94931, 29891, 63817, 7591, 95293, 43924, 59149, 73094, 10154, 40180, 80204, 20916, 8101, 3117, 75034, 54907, 4641, 76214, 27851, 73197, 2580, 59027, 69446, 18021, 94306, 3882, 42324, 3538, 40466, 97383, 36002, 24201, 60625, 52101, 33716, 79841]



def learner(args, batch_q, param_q, lock, d, num_processes, experience_, experience_for_train_):
    def transformer(x, encode, local_device):
        b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
        if encode:
            return torch.cat((encode, b))
        else:
            return b

    def test(test_count):
        # save CheckPoint
        checkpoint_path = './Models/Distributed_v2/now_{}.pth'.format(test_count)
        state = {'model': dqn.net.state_dict()}
        torch.save(state, checkpoint_path)

        path = 'reward_log_Distributed_v2.json'
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
                          join_table, file_name, query_encode[3], dqn.target_net, test=True, distributed=True)
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


    print("---------------------------->",  "learner")
    local_device = torch.device('cuda:' + str(d))
    local_target_model = nn.DataParallel(create_model())
    local_target_model.to(local_device)
    local_eval_model = nn.DataParallel(create_model())
    local_eval_model.to(local_device)
    test_count = 0
    if os.path.exists(args.path):
        print('load existing parameters!')
        local_eval_model.load_state_dict(torch.load(args.path)['model'])
    local_target_model.load_state_dict(local_eval_model.state_dict())
    # update_target_model(local_eval_model, local_target_model)
    for _ in range(num_processes):
        param_q.put(local_eval_model.state_dict())



    local_optimazor = optim.AdamW(local_eval_model.parameters(), lr=args.learning_rate)


    dqn = Distibut_DQN_v2(local_target_model,
                       local_eval_model,
                       local_optimazor,
                      epsilon_start=args.epsilon_start,
                        epsilon_decay=args.epsilon_decay,
                        epsilon_end=args.epsilon_end,
                        path=args.path,
                        batch_size=args.batch_size,
                        sync_batch_size=args.sync_batch_size,
                        capacity=args.capacity)


    local_eval_model.train()
    learn_step = 0
    while True:
        print('train_experiance size:', batch_q.qsize())
        if not batch_q.empty():
            # print(dqn.count)
            dqn.count += 1
            learn_step += 1
            dqn.update_net()
            if dqn.count % 20 == 0:
                param_q.put(local_eval_model.state_dict())
            if learn_step % 500 == 0:
                test(test_count)
                test_count += 1

            batch = batch_q.get()
            zero = [[0] * 318, (tuple([0] * (len(all_tables) * 2 + 3)),)]
            zero_batches = []
            # now state
            now_states = batch[1]

            next_states = []
            # next state

            all_action = batch[2][:2]
            now_values = local_eval_model(all_action)

            cur_idx = 0
            for idx, i in enumerate(batch[2][2]):
                if i:
                    length = len(i)
                    min_idx = torch.argmin(now_values[cur_idx:cur_idx + length], dim=0).item()
                    next_states.append(i[min_idx])
                    cur_idx += length
                else:
                    assert batch[3][idx] == 0
                    next_states.append(zero)
                    zero_batches.append(idx)



            local_eval_model.train()
            q_eval = local_eval_model(now_states)

            encode = []
            trees = []
            for i in next_states:
                encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                trees.append(i[1])
            x = torch.cat(encode, dim=0)
            now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=local_device)
            next_v = local_target_model([x, list(now_trees)]).detach()
            for idx in zero_batches:
                next_v[idx] = 0
            q_target = torch.tensor(batch[0], device=local_device).unsqueeze(1) + next_v * dqn.gamma

            dqn.optimizer.zero_grad()

            loss = dqn.compute_loss(q_eval, q_target)
            loss.backward()

            #a = list(local_eval_model.named_parameters())
            #print(a[0][0], a[0][1].grad)
            #print(a[8][0], a[8][1].grad)

            # Important: add grad_clip
            nn.utils.clip_grad_norm_(local_eval_model.parameters(), max_norm=10)

            dqn.optimizer.step()
            dqn.loss += float(loss)

            '''
            # todo: 这里本来不应该有预处理
            now_states = [now_states[0].clone(), [i.clone() for i in now_states[1]]]
            experience_.put([batch[0], now_states, next_states, batch[3], zero_batches])

            if not experience_for_train_.empty():
                # print('experience_for_train_:', experience_for_train_.qsize())
                batch = experience_for_train_.get()
                reward_ = batch[0]
                now_states = batch[1]
                next_states = batch[2]
                states = batch[3]
                zero_batches = batch[4]


                q_eval = local_eval_model(now_states)


                # encode = []
                # trees = []
                # for i in next_states:
                #     encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                #     trees.append(i[1])
                # x = torch.cat(encode, dim=0)
                # now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=local_device)

                next_v = local_target_model(next_states).detach()

                for idx in zero_batches:
                    next_v[idx] = 0


                # reward_ = [batch[i][0] for i in range(len(batch))]
                # reward_ = batch[0]
                q_target = torch.tensor(reward_, device=local_device).unsqueeze(1) + next_v * dqn.gamma

                dqn.optimizer.zero_grad()

                loss = dqn.compute_loss(q_eval, q_target)
                loss.backward()

                nn.utils.clip_grad_norm_(local_eval_model.parameters(), max_norm=10)

                local_optimazor.step()
                dqn.loss += float(loss)
            '''


        else:
            time.sleep(2)

def sample(experience, experience_for_train, d):
    def transformer(x, encode, local_device):
        b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
        if encode:
            return torch.cat((encode, b))
        else:
            return b

    print("---------------------------->", "sampler")
    d = d % 4
    device = torch.device('cuda:' + str(d))
    while True:
        if not experience.empty():
            print('sample experiance size:', experience.qsize())
            # reward, now_state, next_state, state
            batch = experience.get()
            # now state
            rewards = [i[0] for i in batch]
            states = [i[3] for i in batch]
            now_states = [i[1] for i in batch]

            encode = []
            trees = []
            for i in now_states:
                encode.append(torch.tensor(i[0], dtype=torch.float32, device=device).unsqueeze(0))
                trees.append(i[1])
            x = torch.cat(encode, dim=0)
            now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=device)
            # batch[1] = [x, now_trees]

            # next_state
            all_action = []
            for i in batch:
                if i[2]:
                    all_action.extend(i[2])
            encode_ = []
            trees_ = []
            for i in all_action:
                encode_.append(torch.tensor(i[0], dtype=torch.float32, device=device).unsqueeze(0))
                trees_.append(i[1])
            x_ = torch.cat(encode_, dim=0)
            now_trees_ = prepare_trees(trees_, transformer, left_child, right_child, None, local_device=device)
            # batch[2] = [x_, now_trees_, lengths]
            experience_for_train.put([rewards, [x, list(now_trees)], [x_, list(now_trees_), [i[2] for i in batch]], states])
        else:
            time.sleep(2)


def sample_2(experience, experience_for_train, d):
    def transformer(x, encode, local_device):
        b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
        if encode:
            return torch.cat((encode, b))
        else:
            return b

    print("---------------------------->", "sampler_2")
    local_device = torch.device('cuda:' + str(d))
    while True:
        if not experience.empty():
            # print('sample experiance size:', experience.qsize())
            # reward, now_state, next_state, state, zero_batches
            batch = experience.get()
            rewards = batch[0]
            now_state = batch[1]
            states = batch[3]
            zero_batches = batch[4]

            # now state
            now_state = [now_state[0].clone(), [i.clone() for i in now_state[1]]]
            # next_state
            encode = []
            trees = []
            for i in batch[2]:
                encode.append(torch.tensor(i[0], dtype=torch.float32, device=local_device).unsqueeze(0))
                trees.append(i[1])
            x = torch.cat(encode, dim=0)
            now_trees = prepare_trees(trees, transformer, left_child, right_child, None, local_device=local_device)
            # batch[2] = [x_, now_trees_, lengths]
            experience_for_train.put([rewards, now_state, [x, list(now_trees)], states, zero_batches])
        else:
            time.sleep(2)




def act(args,
        FILE,
        experience_q,
        param_q,
        rank):
    print("---------------------------->", rank, "actor")
    d = (rank % 2)

    # d = 0
    local_device = torch.device('cuda:' + str(d))

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()

    local_model = create_model()
    local_model.to(local_device)
    # local_optim = optim.AdamW(global_model.parameters(), lr=args.learning_rate)

    if not param_q.empty():
        param = param_q.get()
        new_state_dict = OrderedDict()
        for k, v in param.items():
            name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v
        local_model.load_state_dict(new_state_dict)

    dqn = Distibut_DQN_v2(
        None,
        local_model,
        None, # local_optim,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_end=args.epsilon_end,
        path=args.path,
        batch_size=args.batch_size,
        sync_batch_size=args.sync_batch_size,
        capacity=args.capacity)


    for epoch in range(args.epoch_start, args.epoch_end):
        # random.shuffle(d_list)

        print('episode:', epoch)
        # 初始化记录loss为空表格
        dqn.all_loss = []
        dqn.cur_epoch = epoch

        for num, idx in enumerate(FILE):
            # print(num)
            if idx not in TEST_FILE:
                if not param_q.empty():
                    param = param_q.get()
                    new_state_dict = OrderedDict()
                    for k, v in param.items():
                        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                        new_state_dict[name] = v
                    local_model.load_state_dict(new_state_dict)
                # print(local_device, num)
                sql = linecache.getline('./train_plan/query_multi.sql', idx+1)
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

                    assert len(result_now[0]) == len(rl.Q_values)
                    if result_now is not None:
                        for i in range(len(result_now[0])):
                            # todo: 改成真实的 now state
                            now_state = result_now[2][i]
                            # now_state = result_now[2][i] # now_state yong zhen shi de
                            state = 1 if i != len(result_now[2]) - 1 else 0

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

                            dqn.buffer.append([reward, now_state, next_state, state])


            if num % 5000 == 0:
                # optimal_plan_7591 = "/*+ Leading(keyword movie_keyword title kind_type complete_cast comp_cast_type movie_companies movie_info info_type company_name company_type) NestLoop( keyword movie_keyword ) NestLoop( keyword movie_keyword  title) NestLoop( keyword movie_keyword  title  kind_type) NestLoop( keyword movie_keyword  title  kind_type complete_cast) NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type) NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type movie_companies) NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type movie_companies movie_info)    NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type movie_companies movie_info info_type)  NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type movie_companies movie_info info_type company_name) NestLoop( keyword movie_keyword  title  kind_type complete_cast comp_cast_type movie_companies movie_info info_type company_name company_type) IndexScan( movie_keyword ) IndexScan( title ) IndexScan( complete_cast ) IndexScan( comp_cast_type ) IndexScan( movie_companies ) IndexScan( movie_info ) IndexScan( company_name ) */"
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

                result_now = traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector)
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
                    state = 1 if i != len(result_now[2]) - 1 else 0
                    next_state = [result_now[2][i + 1]] if i != len(result_now[2]) - 1 else []
                    reward = result_now[0][i]

                    dqn.buffer.append([reward, now_state, next_state, state])

            if num % 10 == 0:
                if dqn.buffer.check_state():
                    batch = dqn.buffer.sample(dqn.batch_size)
                    experience_q.put(batch)

        dqn.update_epsilon()
        conn.reconnect()
        set_statement_timeout(EXECUTION_MAX_TIME)
        load_extension()

def create_model():
    model = nn.Sequential(
        tcnn.QueryEncoder(318),
        tcnn.BinaryTreeConv(109, 512),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.BinaryTreeConv(512, 256),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.BinaryTreeConv(256, 128),
        tcnn.TreeLayerNorm(),
        tcnn.TreeActivation(nn.ReLU()),
        tcnn.DynamicPooling(),
        tcnn.RegNorm(128)
    )
    return model



def update_target_model(model, target_model, target_model_update=1., learner_step=0):
    if target_model_update < 1.:                    # soft update
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * (1. - target_model_update) +
                                    param.data * target_model_update)
    elif learner_step % target_model_update == 0:   # hard update
        target_model.load_state_dict(model.state_dict())



class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

if __name__ == '__main__':
    # os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    # 暂时不用经验池
    print(print(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    d = 0
    # device = torch.device('cuda:' + str(d))

    # global_eval_net = create_model().to(device)
    # shared_model.share_memory()
    # optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    experience = mp.Queue()
    experience_for_train = mp.Queue()
    params = mp.Queue()

    experience_ = mp.Queue()
    experience_for_train_ = mp.Queue()


    length = len(USEABLE_FILE)
    num_processes = 2
    chunk = int(length / num_processes)

    for i in range(8):
        p_sample = mp.Process(target=sample, args=(experience, experience_for_train, i))
        p_sample.start()
        processes.append(p_sample)

    # todo: 这里本来不应该有预处理
    #for _ in range(2):
    #    p_sample = mp.Process(target=sample_2, args=(experience_, experience_for_train_, d))
    #    p_sample.start()
    #    processes.append(p_sample)

    p_train = mp.Process(target=learner, args=(args,experience_for_train, params, lock, d, num_processes, experience_, experience_for_train_))
    p_train.start()
    processes.append(p_train)


    for rank in range(0, num_processes):
        p = mp.Process(target=act, args=(args, USEABLE_FILE[chunk*rank:chunk*(rank+1)], experience, params, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()