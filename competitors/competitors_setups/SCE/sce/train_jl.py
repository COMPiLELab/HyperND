# NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
import torch
from sce.utils import load_adj_neg, load_dataset, prepare_dataset
from sce.networks import Net
import argparse
import numpy as np
from sce.classification import classify, classify_ours
from argparse import Namespace

def train(args, adj, features, labels, idx_train, idx_val, idx_test):

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed')
    parser.add_argument('--nhid', type=int, default=512,
                        help='hidden size')
    parser.add_argument('--output', type=int, default=512,
                        help='output size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum number of epochs')
    parser.add_argument('--sample', type=int, default=5,
                        help='    ')
    parser.add_argument('--alpha', type=int, default=100000,
                        help='    ')
    parser.add_argument('--num_nodes', type=int, default=19793,
                        help='    ')
    parser.add_argument('--num_features', type=int, default=8710,
                        help='    ')

    #args = parser.parse_args()
    args = Namespace(**args)
    args.device = 'cpu'
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature, adj_normalized= prepare_dataset(adj, features)
    feature = feature.to(device)
    adj_normalized = adj_normalized.to(device)
    #feature=F.normalize(feature, p=1)
    F_1 = torch.mm(adj_normalized, feature)
    F_2 = torch.mm(adj_normalized, F_1)
    args.num_nodes = adj_normalized.shape[0]
    args.num_features = feature.shape[1]
    neg_sample = torch.from_numpy(load_adj_neg(args.num_nodes, args.sample)).float().to(device)

    model = Net(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    for epoch in range(args.epochs):

        optimizer.zero_grad()
        out = model(F_1, F_2)
        loss = args.alpha / torch.trace(torch.mm(torch.mm(torch.transpose(out, 0, 1), neg_sample), out))
        print(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    emb = model(F_1, F_2).cpu().detach().numpy()
    np.save('embedding.npy', emb)
    print(len(idx_train), len(idx_test))
    acc, pred = classify_ours(emb, labels, idx_train, idx_test)
    return pred
