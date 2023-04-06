import torch
import torch.nn as nn

class Mapper(nn.Module):
    def __init__(self):
        super.__init__()
        self.W = nn.Linear(32, 32)
    def foward(self, F):
        F_ = self.W(F)
        return F_
