import os
import torch
import torchvision
from torch import nn
from TreeConvolution.util_2 import prepare_trees
import torch.optim as optim




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
def transformer(x, encode):
    b = torch.tensor(x[0], dtype=torch.float32)
    return torch.cat((encode, b))



class autoencoder2(nn.Module):
    def __init__(self):
        super(autoencoder2, self).__init__()
        self.hidden1 = nn.Linear(318, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.act = nn.ReLU()

        self.encoder = nn.LSTM(109,128,batch_first=True)
        self.q_net = nn.Linear(128,64)
        self.K_net = nn.Linear(128,64)
        self.sm = nn.Softmax(dim=1)


        # self.FCN1 = nn.Linear(128,256)
        # self.FCN2 = nn.Linear(256, 128)
        self.FCN1 = nn.Linear(128, 64)
        self.FCN2 = nn.Linear(64,1)


    def forward(self, goal):
        encode = []
        trees = []
        for i in goal:
            encode.append(torch.tensor(i[0], dtype=torch.float32).unsqueeze(0))
            trees.append(i[1])
        x = torch.cat(encode, dim=0)
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.hidden3(x))


        trees, idxes = prepare_trees(trees, transformer, left_child, right_child, x)
        trees = trees.permute(0, 2, 1)
        # print(idx)
        # print(trees[0, 10:, :])
        for batch, idx in enumerate(idxes):
            trees[batch, idx:, :] = 0

        output, (h_n, c_n) = self.encoder(trees)
        #计算最后一步的q值
        q = self.q_net(h_n) # [1, 2, 64]

        #计算所有时刻的K值
        K = self.K_net(output) #1*len*64
        #V值就是output
        scores = torch.bmm(K,q.permute(1, 2, 0))
        scores = self.sm(scores) #softmax转化为概率分布
        output = torch.mul(output,scores)
        output = torch.sum(output, dim=1,keepdim=True).reshape((-1, 128)) #2*1*128

        x = self.act(self.FCN1(output))
        x = self.act(self.FCN2(x))

        return x



class Net:
    def __init__(self, path='./Models/now_attention.pth', learning_rate=0.0008, steps_per_epoch=100, epoch=120, max_lr=0.001):
        self.path = path
        self.model = autoencoder2()
        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.data.normal_(0, 0.001)
                m.bias.data = torch.ones(m.bias.data.size())
        self.model.apply(init_weights)

        # for i in range(len(self.model)):
        #     print(self.model[i])
        # print(list(self.model.parameters()))
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.00001, momentum=0.9)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                                        epochs=epoch)

        if os.path.exists(self.path):
            checkpoint = torch.load(self.path)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict.keys()}
            self.model.load_state_dict(state_dict)
            # self.model.load_state_dict(checkpoint['model'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])


    def save(self):
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.path)



if __name__ == '__main__':
    model = autoencoder2()
    input = [[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0], ((1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                           0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1), ((0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1), ((0, 0,
                                                                                                                  1, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  1), ((
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),)),
                                                                                                                 ((0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 1,
                                                                                                                   1),)),
                                                                       ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                         0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),)), ((0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      1,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0),))],
             [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0], ((1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1), ((0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                                                                        0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1), ((0, 0,
                                                                                                                  1, 1,
                                                                                                                  1, 0,
                                                                                                                  0, 1,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  1, 0,
                                                                                                                  0, 0,
                                                                                                                  0, 1,
                                                                                                                  1), ((
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       1),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       1),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),)),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       1),)),
                                                                                                                       (
                                                                                                                       (
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       1,
                                                                                                                       1,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0,
                                                                                                                       0),)),
                                                                                                                 ((0, 0,
                                                                                                                   0, 1,
                                                                                                                   1, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0, 0,
                                                                                                                   0),)),
                                                                       ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),)), ((0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      1,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0,
                                                                                                                      0),))]
             ]
    model(input)
    model([input[0]])
    model([input[1]])




# model = autoencoder()
# # criterion = nn.MSELoss()
# # optimizer = torch.optim.Adam(
# #     model.parameters(), lr=learning_rate, weight_decay=1e-5)
#
#
# input = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# input = np.array(input)
# print(input.shape)
#
#
# out = model(input)
# print(out.shape)



# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = img.view(img.size(0), -1)
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch + 1, num_epochs, loss.data[0]))
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, './mlp_img/image_{}.png'.format(epoch))
#
# torch.save(model.state_dict(), './sim_autoencoder.pth')