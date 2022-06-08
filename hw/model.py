import torch
import torch.nn as nn
from torch.nn.functional import relu

class NN_Model(nn.Module):
    def __init__(self):
        super(NN_Model, self).__init__()
        self.l1 = nn.Linear(784, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 10)

    def forward(self, X):
        return self.l3(relu(self.l2(relu(self.l1(X)))))