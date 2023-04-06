import numpy as np
import torch
from torch import nn

a = torch.Tensor(range(9)).reshape(-1, 3)
dm = a.size()
# b = torch.ones(3) * 2

# c = a * b


class WFD(nn.Module):
    def __init__(self):
        super(WFD, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(3, 3))
        self.alpha_og = nn.ModuleList(
            nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, bias=True) for l in
            range(6))
        self.alpha_ng = nn.ModuleList(
            nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0, bias=True) for l in
            range(6))
        self.af = nn.ReLU(inplace=True)
        nn.init.uniform_(self.W)

    def forward(self, x):
        output = self.W * x
        return output


X = []
Y = []
w_true = torch.ones((3, 3)) * 2
# print(w_true)
# print(w_true.sum())
for i in range(200):
    x = torch.randint(1, 10, (3, 3)) * 1
    y = 2 * x
    X.append(x)
    Y.append(y.to(torch.float))
    # print(x)
    # print(y)

w = WFD()
a = w.parameters()
optimizer = torch.optim.Adam(w.parameters(), lr=0.1)
# w = torch.rand((3, 3))
# print(w)
learning_rate = 0.01
# loss = (w * x - y)^2
# dw = 2 * (w * x - y) * x
loss_mse = torch.nn.MSELoss()
loss_mse.requires_grad_()
for i in range(200):
    # dw = 2 * (w * X[i] - Y[i]) * X[i]
    # w -= learning_rate * dw
    # y_ = torch.from_numpy(w * X[i])
    # y = torch.from_numpy(Y[i])
    # y_.requires_grad=True
    # y.requires_grad=True
    # output = loss_mse(y, y_)
    # print(output)
    # output.backward()
    # loss = loss.sum()
    optimizer.zero_grad()
    output = w(X[i])
    loss = loss_mse(Y[i], output)
    loss.backward()
    optimizer.step()
    print(w.W.data)

    # print("round{}, loss = {}".format(i, output))


