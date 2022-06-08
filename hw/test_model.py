import torch
from torch.nn.functional import relu
class Test_Model(torch.nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()
        self.l1 = torch.nn.Linear(2, 3, bias=True)
        self.l1.weight = torch.nn.Parameter(torch.tensor([[0.5,-1], [2,2], [0,1]], dtype=torch.float32))
        self.l1.bias = torch.nn.Parameter(torch.tensor([1,0,0.5], dtype=torch.float32))
        
        self.l2 = torch.nn.Linear(3, 3, bias=True)
        self.l2.weight = torch.nn.Parameter(torch.tensor([[-3,2,1], [0,0.5,-1], [-2,1,0.5]], dtype=torch.float32))
        self.l2.bias = torch.nn.Parameter(torch.tensor([0,0,1], dtype=torch.float32))
        
        self.l3 = torch.nn.Linear(3, 2, bias=True)
        self.l3.weight = torch.nn.Parameter(torch.tensor([[1,1,0.5], [2,1,-1]], dtype=torch.float32))
        self.l3.bias = torch.nn.Parameter(torch.tensor([0.5,0.5], dtype=torch.float32))

    def forward(self, X):
        return self.l3(relu(self.l2(relu(self.l1(X)))))