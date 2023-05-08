import os
from torch import nn
import numpy as np
from utils.Connection import all_tables
from utils.PlanTree import OPERATOR_TYPE
import torch
import random
from utils.ExperienceBuffer import ExperienceBuffer, PrioritizedReplayBuffer

# todo: 可以修改batch_size 以及batch训练时loss计算的方法！！
# torch.autograd.set_detect_anomaly(True)
class DQN:
    def __init__(
            self, target_net_model, net_model,
            capacity=50, epsilon_start=1, epsilon_end=0.1, epsilon_decay=0.85, batch_size=20, sync_batch_size=30, path='./Models/now.pth'):
        # self.buffer = ExperienceBuffer(capacity)
        self.buffer = PrioritizedReplayBuffer(capacity)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_net_model = target_net_model
        self.net_model = net_model
        self.target_net = target_net_model.model
        self.net = net_model.model
        self.count = 0
        self.learn_step_counter = 0
        self.target_replace_iter = 20
        self.sync_batch_size = sync_batch_size
        self.total_reward = []
        self.mean_reward = []
        self.optimizer = net_model.optimizer
        # self.compute_loss = nn.L1Loss()
        self.compute_loss = nn.MSELoss()
        self.gamma = 0.9
        self.loss = 0
        self.path = path
        self.all_loss = []
        self.scheduler = net_model.scheduler
        self.cur_epoch = 0

    def learn(self):
        self.learn_step_counter += 1
        # todo：第三版
        if self.buffer.check_state():
            batch, importance, indices = self.buffer.sample(self.batch_size, (self.learn_step_counter%2==0)*0.7)
            self.count += 1
            self.update_net()

            zero = [[0]*len(batch[0][1][0]), (tuple([0] * (len(all_tables) * 2 + 3)),)]
            zero_batches = []
            # now state
            now_states = [i[1] for i in batch]
            next_states = []
            # next state
            all_action = []
            for i in batch:
                if i[2]:
                    all_action.extend(i[2])
            assert len(all_action) > 0
            now_values = self.net(all_action)

            # print(now_values[0])
            cur_idx = 0
            for idx, i in enumerate(batch):
                if i[2]:
                    length = len(i[2])
                    min_idx = torch.argmin(now_values[cur_idx:cur_idx+length], dim=0).item()
                    next_states.append(i[2][min_idx])
                    cur_idx += length
                else:
                    assert i[3] == 0
                    next_states.append(zero)
                    zero_batches.append(idx)

            assert len(now_states) == len(next_states)
            self.net.train()
            q_eval = self.net(now_states)
            next_v = self.target_net(next_states).detach()
            for idx in zero_batches:
                next_v[idx] = 0
            q_target = torch.tensor([batch[i][0] for i in range(len(batch))]).unsqueeze(1) + next_v * self.gamma

            self.optimizer.zero_grad()
            error = torch.abs(q_eval - q_target).cpu().data.numpy().reshape(-1).tolist()
            # update priority
            self.buffer.set_priorities(indices, error)


            loss = self.compute_loss(q_eval, q_target)
            importance = torch.tensor(importance**(1-self.epsilon_start)).unsqueeze(1)
            loss = torch.mean(importance*loss)

            loss.backward()
            self.optimizer.step()
            self.loss += float(loss)

            # for name, parameters in self.net.named_parameters():
            #     print(name, ':', parameters.grad)
            #     break
            # a = list(self.net.named_parameters())
            # print(a[0][0], a[0][1].grad)
            # print(a[8][0], a[8][1].grad)

        # todo: 第二版
        # if self.buffer.check_state():
        #     batch = self.buffer.sample(self.batch_size)
        #     # batch = np.array(batch)
        #     self.count += 1
        #     self.update_net()
        #
        #     zero = [[0] * len(batch[0][1][0]), (tuple([0] * (len(all_tables) * 2 + 3)),)]
        #     zero_batches = []
        #     for idx, i in enumerate(batch):
        #         if not i[2]:
        #             zero_batches.append(idx)
        #
        #     now_states = [i[1] for i in batch]
        #     next_states = [i[2] if i[2] else zero for i in batch]
        #     # now_trees = prepare_trees(now_states, transformer, left_child, right_child)
        #     self.net.train()
        #     q_eval = self.net(now_states)
        #     next_v = self.target_net(next_states).detach()
        #     # print(q_eval, next_v)
        #     # print(zero_batches)
        #     for idx in zero_batches:
        #         next_v[idx] = 0
        #     # print(next_v)
        #     # print(torch.tensor([batch[i][0] for i in range(len(batch))]).unsqueeze(1))
        #     q_target = torch.tensor([batch[i][0] for i in range(len(batch))]).unsqueeze(1) + next_v * self.gamma
        #
        #     self.optimizer.zero_grad()
        #     loss = self.compute_loss(q_eval, q_target)
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        #     self.optimizer.step()
        #     self.loss += float(loss)

            # todo: 第一版
            # for i in range(len(batch)):
            #     self.count += 1
            #     self.update_net()
            #     now_state = batch[i][1]
            #     self.net.train()
            #     q_eval = self.net(now_state)
            #     if len(batch[i][2]) != 0:
            #         next_v = self.target_net(batch[i][2]).detach()
            #         q_target = batch[i][0] + next_v * self.gamma
            #     else:
            #         q_target = torch.tensor([batch[i][0]]).cuda()
            #
            #     self.optimizer.zero_grad()
            #     # print(q_eval, q_target)
            #     loss = self.compute_loss(q_eval, q_target)
            #     loss.backward()
            #     self.optimizer.step()
            #     self.loss += float(loss)



    def update_epsilon(self):
        if self.epsilon_start > self.epsilon_end:
            self.epsilon_start = self.epsilon_start * self.epsilon_decay

    def store_state(self, result_now):
        if result_now is not None:
            for i in range(len(result_now[0])):
                now_state = result_now[2][i]
                state = 1 if i != len(result_now[2]) else 0
                next_state = result_now[2][i + 1] if i != len(result_now[2]) - 1 else []
                reward = result_now[0][i]
                self.buffer.append([reward, now_state, next_state, state])

    def update_net(self):
        if self.count % self.sync_batch_size == 0:
            print(self.loss/self.count)
            # self.all_loss.append(self.loss/self.count)
            self.loss = 0
            self.count = 0
            self.target_net.load_state_dict(self.net.state_dict())
            self.scheduler.step()
            # state = {'model': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()}
            state = {'model': self.net.state_dict()}
            path = self.path.split('.')
            path[1] = path[1] + '_' + str(self.cur_epoch)
            path = '.'.join(path)
            torch.save(state, path)


