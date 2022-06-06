import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle as pkl
from time import perf_counter
from argparse import Namespace

from sgc.utils import load_citation, sgc_precompute, set_seed, prepare_data
from sgc.models import get_model
from sgc.metrics import accuracy
from sgc.args import get_citation_args


# Arguments
def main(args, adj, features, labels, idx_train, idx_val, idx_test):
    #args = get_citation_args()
    args = Namespace(**args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.tuned:
        if args.model == "SGC":
            with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
                args.weight_decay = pkl.load(f)['weight_decay']
                print("using tuned weight decay: {}".format(args.weight_decay))
        else:
            raise NotImplemented

    # setting random seeds
    set_seed(args.seed, args.cuda)
    print(adj)
    data = [adj, features, labels, idx_train, idx_val, idx_test]
    adj, features, labels, idx_train, idx_val, idx_test = prepare_data(data, args.normalization, args.cuda)
    print(adj.shape, features.shape, labels.shape, idx_train.shape, idx_val.shape, idx_test.shape)
    model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

    if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))

    def train_regression(model,
                        train_features, train_labels,
                        val_features, val_labels,
                        epochs=args.epochs, weight_decay=args.weight_decay,
                        lr=args.lr, dropout=args.dropout):

        optimizer = optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
        t = perf_counter()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(train_features)
            loss_train = F.cross_entropy(output, train_labels)
            loss_train.backward()
            optimizer.step()
        train_time = perf_counter()-t

        with torch.no_grad():
            model.eval()
            output = model(val_features)
            acc_val = accuracy(output, val_labels)

        return model, acc_val, train_time

    def test_regression(model, test_features, test_labels):
        model.eval()
        return accuracy(model(test_features), test_labels), model(test_features).detach().numpy()

    if args.model == "SGC":
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                        args.epochs, args.weight_decay, args.lr, args.dropout)
        acc_test, pred = test_regression(model, features[idx_test], labels[idx_test])


    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
    return pred
