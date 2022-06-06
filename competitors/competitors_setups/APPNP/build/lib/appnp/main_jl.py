""" Running the APPNP model."""

import torch
import os
import sys
from appnp.appnp import APPNPTrainer
from appnp.param_parser import parameter_parser
from appnp.utils import tab_printer, graph_reader, graph_reader_from_var,\
 feature_reader, target_reader, feature_reader_from_var

from argparse import Namespace

def main(args, edges, features, target):
    """
    Parsing command line parameters, reading data, fitting an APPNP/PPNP and scoring the model.
    """
    args = Namespace(**args)
    torch.manual_seed(args.seed)
    tab_printer(args)

    edges = [list(a) for a in edges]
    graph = graph_reader_from_var(edges)
    features = feature_reader_from_var(features)
    trainer = APPNPTrainer(args, graph, features, target)
    result = trainer.fit()
    return result.detach().numpy()
