import numpy as np
import scipy.sparse as sp
import torch
import sce.load_process as load_process
import sys
import pickle as pkl
import networkx as nx


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1.0).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj)


def load_adj_neg(num_nodes, sample):
    '''
    adj_neg = np.zeros((num_nodes, num_nodes), dtype=float)
    l = np.random.randint(0, num_nodes, size=num_nodes * (sample + 1))
    t = 0
    for i in range(num_nodes):
        s = 0
        adj_neg[i, i] = sample
        while s < sample:
            if i != l[t]:
                adj_neg[i, l[t]] = -1
                s += 1
            t += 1
    '''
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    data = np.ones(num_nodes * sample)
    adj_neg = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
    return adj_neg



def load_dataset(dataset_str):
    if dataset_str == 'cora_full':
        data_name = dataset_str + '.npz'
        data_graph = load_process.load_npz_to_sparse_graph("data/{}".format(data_name))
        data_graph.to_undirected()
        data_graph.to_unweighted()
        A = data_graph.adj_matrix
        X = data_graph.attr_matrix
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(A.shape[0]) + A).toarray()).float()
        X = torch.from_numpy(X.todense()).float()
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset_str == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
        X = torch.from_numpy(features.todense()).float()

    return X, adj_normalized

def prepare_dataset(adj, features):
    adj = sp.csr_matrix(adj)
    adj_normalized = torch.from_numpy(normalize_adj(sp.eye(adj.shape[0]) + adj).toarray()).float()
    X = torch.from_numpy(features).float()


    return X, adj_normalized
