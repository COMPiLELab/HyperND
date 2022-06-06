import torch
import torch.nn.functional as F
from sce.mlp import mlp

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        #self.conv0 = mlp(input_dim=args.num_features, output_dim=args.output)
        self.conv1 = mlp(input_dim=args.num_features, output_dim=args.output)
        self.conv2 = mlp(input_dim=args.num_features, output_dim=args.nhid)
        self.conv3 = mlp(input_dim=args.nhid, output_dim=args.output)


    def forward(self, F_1, F_2):

        z_1 = self.conv1(F_1)
        z_2 = self.conv3(self.conv2(F_2))
        z = (z_1+z_2)/2

        return z