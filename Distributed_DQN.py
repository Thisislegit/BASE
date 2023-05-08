import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from utils.Net import Net
from utils.DQN import Distibut_DQN
from utils.Arguments import parser
import math
from utils.DNNRLGOO import DNNRLGOO
from utils.Utils import *
from utils.PlanTree import *
from utils.Connection import Conn
from utils.PlanSQL import travesal_SQL
import random

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



def train():
    pass



def act(args,
        FILE,
        global_model,
        counter,
        lock,
        optimizer,
        rank):
    print("---------------------------->", rank, "learner")
    d = (rank % 3)

    local_device = torch.device('cuda:' + str(d))

    set_statement_timeout(EXECUTION_MAX_TIME)
    load_extension()

    local_model = create_model()
    local_model.to(local_device)
    # local_optim = optim.AdamW(global_model.parameters(), lr=args.learning_rate)
    update_target_model(global_model, local_model)


    dqn = Distibut_DQN(
        local_model,
        global_model,
        optimizer, # local_optim,
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
            if idx not in TEST_FILE:
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
                    try:
                        result_now = traversal_plan_tree_cost(terminate_plan['Plan'], f, query_vector)
                    except:
                        with open('plan_{}.json'.format(idx), 'w') as f:
                            json.dump(terminate_plan, f)
                        assert 0
                    rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                    for i in range(1, len(result_now[0])):
                        rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                    result_now[0] = rewards
                    # print(result_now[1])
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

                            dqn.buffer.add([reward, now_state, next_state, state])


            if num % 5000 == 0:
                sql = linecache.getline('./train_plan/query_multi.sql', idx + 1)
                sql = sqlparse.format(sql, reindent=True, keyword_case="upper")
                terminate_plan = get_hint_SQL_explain(sql,
                                                      'EXPLAIN (format json)',
                                                      str(11),
                                                      conn.conn)
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
                result_now = traversal_plan_tree_cost(terminate_plan['Plan'], 1, query_vector)
                rewards = [round(pow(result_now[0][0], 1 / NORMALIZATION), 6)]
                for i in range(1, len(result_now[0])):
                    rewards.append(round(pow(result_now[0][i] - result_now[0][i - 1], 1 / NORMALIZATION), 6))
                result_now[0] = rewards
                for i in range(len(result_now[0])):
                    now_state = result_now[2][i]
                    state = 1 if i != len(result_now[2]) - 1 else 0
                    next_state = [result_now[2][i + 1]] if i != len(result_now[2]) - 1 else []
                    reward = result_now[0][i]

                    dqn.buffer.add([reward, now_state, next_state, state])

            if num % 10 == 0:
                dqn.learn(local_device=local_device)
                with lock:
                    counter.value += 1

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    # 暂时不用经验池
    # experience = mp.Queue()
    print(print(torch.cuda.current_device()))
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device('cuda:' + str(2))
    # shared_model = Net(path=args.path, learning_rate=args.learning_rate, max_lr=args.max_lr, steps_per_epoch=args.steps_per_epoch, epoch=args.epoch_end-args.epoch_start)
    # shared_model = shared_model.model.to(torch.device('cuda'))  # 会造成很多问题
    shared_model = create_model().to(device)
    # shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=args.learning_rate)
    # optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    length = len(USEABLE_FILE)
    num_processes = 3
    chunk = int(length / num_processes)


    for rank in range(0, num_processes):
        p = mp.Process(target=act, args=(args, USEABLE_FILE[chunk*rank:chunk*(rank+1)], shared_model, counter, lock, optimizer, rank))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()