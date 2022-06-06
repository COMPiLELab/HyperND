# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
# from config import config
from argparse import Namespace
from hypergcn.model import model

def train(args, X, y, hypergraph, train, test):
    args = Namespace(**args)
    #args = config.parse()
    dataset = {'hypergraph': hypergraph, 'features': X.todense(), 'labels': y, 'n': len(y)}


    # seed
    import os, torch, numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)



    # gpu, seed
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)



    # load data
    #dataset, train, test = data.load(args)
    print("length of train is", len(train))




    # # initialise HyperGCN
    HyperGCN = model.initialise(dataset, args)



    # train and test HyperGCN
    HyperGCN = model.train(HyperGCN, dataset, train, args)
    acc, H, Z = model.test(HyperGCN, dataset, test, args)
    return float(acc), H.cpu().detach().numpy(), Z.cpu().detach().numpy()
