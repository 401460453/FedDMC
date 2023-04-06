import torch
from torch import nn

class conv1by1(nn.Module):
    def __init__(self):
        super(conv1by1, self).__init__()
        self.W = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        output = self.W(x)
        return output


model = conv1by1()
x = torch.ones((2, 2, 2))
output = model(x)
print(1)
