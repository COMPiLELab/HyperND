import torch
from torch import nn
from torch.nn import functional as F
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class mlp(nn.Module):
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(mlp, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, inputs):
        x = inputs
        xw = torch.matmul(x, self.weight)

        return xw
