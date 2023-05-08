import torch
import torch.nn as nn
import torch.nn.functional as F
from TreeConvolution.util_1 import prepare_trees
import numpy as np
import time
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# torch.set_default_tensor_type(torch.FloatTensor)
def left_child(x):
    assert isinstance(x, tuple)

    if len(x) == 1:
        # leaf.
        return None
    return x[1]


# function to extract the right child of node
def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]


# function to transform a node into a (feature) vector,
# should be a numpy array.
def transformer(x, encode, local_device):
    b = torch.tensor(x[0], dtype=torch.float32, device=local_device)
    return torch.cat((encode, b))


# function to connect query and plan encode
def connectation(query, plan):
    b = torch.tensor(plan[0], dtype=torch.float32)
    a = torch.cat((query, b))
    goal = [a]
    if len(plan) > 1 and len(plan) < 4:
        left = connectation(query, plan[1])
        right = connectation(query, plan[2])
        goal = goal+left+right
        return goal
    else:
        # return (tuple(a),)
        # empty_node = torch.tensor([0] * len(a), dtype=torch.float32)
        return [a]


# function to get tree index
def get_index(plan, i):
    if len(plan) > 1 and len(plan) < 4:
        left_index = get_index(plan[1], i+1)
        right_index = get_index(plan[2], left_index[-1][0]+1)
        now = [i, i + 1, left_index[-1][0]+1]
        return [now]+left_index+right_index
    else:
        return [[i, 0, 0]]


def get_expanded(query,plan):
    b = torch.tensor(plan[0], dtype=torch.float32)
    a = torch.cat((query, b))
    if len(plan) > 1 and len(plan) < 4:
        left = get_expanded(query, plan[1])
        right = get_expanded(query, plan[2])
        left_node = torch.cat((query, torch.tensor(plan[1][0], dtype=torch.float32)))
        right_node = torch.cat((query, torch.tensor(plan[2][0], dtype=torch.float32)))
        goal = [a, left_node, right_node] + left + right
        return goal
    else:
        empty_node = torch.tensor([0]*len(a), dtype=torch.float32)
        return [a, empty_node, empty_node]


# conv of Binary Tree
class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BinaryTreeConv, self).__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3, bias=False)
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3, bias=False)

    def forward(self, flat_data):
        # self.local_device = next(self.weights.parameters()).device
        # self.local_device = torch.device('cuda:' + str(0))
        trees, idxes, zero_pos = flat_data
        self.local_device = trees.device

        zero_vec = torch.zeros((trees.shape[0], self.__in_channels), device=self.local_device).unsqueeze(2)
        trees = torch.cat((zero_vec, trees), dim=2)

        # add a zero vector back on
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)
        # zero_vec = torch.zeros((expanded.shape[0], self.__out_channels)).unsqueeze(2)
        # results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes, zero_pos)


# Activation of Binary Tree
class TreeActivation(nn.Module):
    def __init__(self, activation):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]),x[1], x[2])


class TreeLayerNorm(nn.Module):
    def forward(self, x):
        eps = np.finfo(np.float32).eps.item()
        data, idxes, zero_pos = x
        # a = time.time()
        # todo: 为了加速训练，这里不执行，模型会有bug
        if (zero_pos is not None) and len(set(zero_pos)) > 1:
            means = []
            stds = []
            for batch, pos in enumerate(zero_pos):
                mean = torch.mean(data[batch:batch+1, :, :pos])
                # print(mean)
                std = torch.std(data[batch:batch+1, :, :pos].clone())
                # data[batch, :, :pos] = (data[batch, :, :pos] - mean) / (std + 0.00001)
                # data = (data - mean) / (std + 0.00001)
                means.append(mean.reshape(-1).unsqueeze(1).unsqueeze(1))
                stds.append(std.reshape(-1).unsqueeze(1).unsqueeze(1))
            mean, std = torch.cat(means, dim=0), torch.cat(stds, dim=0)
            data = (data - mean) / (std + eps)
            # print(data[0, 0, :])
            # print(data)
            # b = time.time()
            # print('norm:', b-a)
            return data, idxes, zero_pos
        else:
            mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            # print(mean)
            std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            data = (data - mean) / (std + eps)
            # print(data[0, 0, :])
            # print(data)
            return (data, idxes, zero_pos)


class DynamicPooling(nn.Module):
    def forward(self, x):

        # print(torch.max(x[0], dim=2))
        # print(x[0])
        if (x[2] is not None) and len(set(x[2].tolist())) > 1:
            all = []
            for batch, pos in enumerate(x[2]):
                all.append(torch.max(x[0][batch:batch+1, :, :pos], dim=2)[0])
            return torch.cat(all, dim=0)
        else:
            return torch.max(x[0], dim=2)[0]


class DynamicPooling_Min(nn.Module):
    def forward(self, x):

        # print(torch.max(x[0], dim=2)
        if (x[2] is not None) and len(set(x[2])) > 1:
            all = []
            for batch, pos in enumerate(x[2]):
                all.append(torch.min(x[0][batch:batch+1, :, :pos], dim=2)[0])
            return torch.cat(all, dim=0)
        else:
            return torch.min(x[0], dim=2)[0]


# encode query information
class QueryEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.act = nn.ReLU()


    def forward(self, goal):
        # self.local_device = next(self.hidden1.parameters()).device
        # self.local_device = torch.device('cuda:' + str(0))
        if len(goal) == 2 and isinstance(goal[1], tuple):
            x = torch.tensor(goal[0], dtype=torch.float32, device=self.local_device)
            y = goal[1]
            x = self.act(self.hidden1(x))
            x = self.act(self.hidden2(x))
            x = self.act(self.hidden3(x))
            # a = [torch.tensor([0] * (len(y[0])+len(x)), dtype=torch.float32)]
            a = []
            a = a + connectation(x, y)
            a = torch.stack(a).unsqueeze(0).transpose(1, 2)
            b = torch.tensor(get_index(y, 1), device=self.local_device).flatten().reshape(-1, 1).unsqueeze(0)
            return a, b, None
        elif len(goal) == 2 and isinstance(goal[0], torch.Tensor):

            x = self.act(self.hidden1(goal[0]))
            x = self.act(self.hidden2(x))
            x = self.act(self.hidden3(x))
            now = goal[1]
            x = x.unsqueeze(2).expand(-1, -1, now[0].shape[2])
            now[0] = torch.cat([x, now[0]], dim=1)
            return now

        else:
            #self.local_device = torch.device('cuda:' + str(1))
            self.local_device = next(self.hidden1.parameters()).device
            encode = []
            trees = []
            for i in goal:
                encode.append(torch.tensor(i[0], dtype=torch.float32, device=self.local_device).unsqueeze(0))
                #encode.append(torch.tensor(i[0], dtype=torch.float32).unsqueeze(0))
                trees.append(i[1])
            x = torch.cat(encode, dim=0)
            x = self.act(self.hidden1(x))
            x = self.act(self.hidden2(x))
            x = self.act(self.hidden3(x))

            # a = time.time()
            now_trees = prepare_trees(trees, transformer, left_child, right_child, x, local_device=self.local_device)
            # b = time.time()
            # print(b-a)

            return now_trees


class RegNorm(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.hidden4 = nn.Linear(32, 1)
        # self.activation = nn.Softplus(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.hidden4(x)
        x = self.activation(x)
        return x